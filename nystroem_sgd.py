"""
Modern implementation for the Nyström-SGD algorithm proposed by Pfahler and Morik and
originally implemented at https://bitbucket.org/Whadup/kernelmachine/ using Python, C++ and Lapack.

This code uses pytorch and cuda for the linear algebra operations.
"""
import torch
import torch.linalg
import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
class NyströmSGD(BaseEstimator, ClassifierMixin):
    """
    The Nyström-SGD algorithm implemented as a sklearn estimator using pytorch linalg operations
    """
    def __init__(self, m, gamma=None, epochs=100, batchsize=64):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.epochs = epochs
        self.batchsize = batchsize

    def _gaussian(self, X, Y):
        XX = torch.sum(X ** 2, axis=1, keepdims=True)
        if X is Y:
            YY = XX
        else:
            YY = torch.sum(Y ** 2, axis=1, keepdims=True)
        XY = torch.matmul(X, Y.transpose(0, 1))
        d2 = -2 * XY + XX.reshape(-1, 1) + YY.reshape(1, -1)
        d2 = d2.clamp(0)
        d = torch.sqrt(d2)
        G = torch.exp(-d / (self.gamma if self.gamma is not None else torch.tensor(X.shape[1])))
        return G
    @torch.no_grad()
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        X = torch.tensor(X).cuda()
        y = torch.tensor(y).cuda()

        sample = torch.randperm(X.shape[0])[:self.m]
        X_sample = X[sample, :]

        print("Preprocessing...", end="", flush=True)
        K_nl = self._gaussian(X, X_sample)
        K_ll = K_nl[sample, :]
        Sigma, V = torch.linalg.eigh(K_ll)
        start = (Sigma <= 1e-9).sum().long().item() # get rid of numerical garbage in the lower singular values

        Sigma = Sigma[start:]
        Sigma *= torch.sqrt(torch.tensor(1.0 * self.m / X.shape[0]))

        A = torch.matmul(K_nl[:,:], V[:, start:] / Sigma.reshape(1, -1))
        Q, R = torch.linalg.qr(A)

        Lambda, U_snake = torch.linalg.eigh(torch.matmul(R, Sigma * R.transpose(0, 1)))

        start2 = (Lambda <= 1e-9).sum().long().item() # get rid of numerical garbage in the lower singular values

        Lambda = Lambda[start2:]
        U_snake = U_snake[:, start2:]
        U = torch.matmul(Q, U_snake)

        Lambda_snake = (1.0 - 1.0 * torch.sqrt(Lambda[0] / Lambda)) / (X.shape[0] * Lambda)#, torch.zeros_like(Lambda))
        Lambda_sgd = Lambda - Lambda_snake.clamp(0) * (Lambda ** 2)#torch.where(true, ... , torch.zeros_like(Lambda))

        eta = len(self.classes_) / (2.0 * Lambda[-1]) * torch.sqrt(Lambda[-1] / Lambda[1])
        print(" done.")
        w = torch.zeros((len(Lambda), len(self.classes_)), dtype=U.dtype, device=X.device)
        with tqdm.tqdm(total=self.epochs * self.batchsize * (len(U) // self.batchsize)) as pbar:
            for epoch in range(self.epochs):
                accuracy = 0
                total = 0
                for U_batch, y_batch in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(U, y), batch_size=self.batchsize, shuffle=True, drop_last=True):
                    pbar.update(len(U_batch))

                    pred = U_batch.matmul(w)
                    accuracy += (pred.argmax(dim=1) == y_batch).sum().item()
                    total += len(y_batch)

                    #Compute MSE
                    l = ((pred - torch.nn.functional.one_hot(y_batch, num_classes=len(self.classes_))) ** 2)
                    l_prime = (pred - torch.nn.functional.one_hot(y_batch, num_classes=len(self.classes_))).unsqueeze(0)

                    #Perform Update
                    gradients = (Lambda_sgd.reshape(-1, 1) * U_batch.transpose(0, 1)).unsqueeze(-1)
                    w = w - eta / self.batchsize * (l_prime* gradients).sum(1)

                    pbar.set_description(f"{l.mean().item():.4f} {accuracy / total:.4f}")

        # Store final estimator
        embed = torch.matmul(V[:, start:], torch.matmul(R, U_snake)) / Lambda.reshape(1, -1)
        self.X_sample_ = X_sample
        self.f_ = torch.matmul(embed, w)


    def predict_proba(self, X):
        X = check_array(X)
        X = torch.tensor(X).cuda()
        K = self._gaussian(X, self.X_sample_)
        return K.matmul(self.f_).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from scipy.stats import mode

    datasets = [
       "MagicTelescope", "mozilla4", "electricity", "eeg-eye-state",
        "click_prediction_small" , "PhishingWebsites", "Amazon_employee_access"
    ]
    for dataset in datasets:
        print(f"========~ {dataset} ~========")
        X, y = fetch_openml(dataset, return_X_y=True)
        # binary targets majority vs rest
        majority_class, _ = mode(y, axis=None)
        y = (y == majority_class.item()).astype(int)
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        # normalize X
        q = make_pipeline(QuantileTransformer(), MinMaxScaler(feature_range=(-1, 1)))
        X_train = q.fit_transform(X_train)
        X_test = q.transform(X_test)
        # Fit Kernel Machine
        print("svm")
        svm = SVC(C=100, gamma="auto")
        svm.fit(X_train, y_train.values)
        print(svm.score(X_test, y_test))
        kernel = NyströmSGD(10000)
        print("nyström-sgd")
        kernel.fit(X_train, y_train.values)
        print(kernel.score(X_test, y_test))
