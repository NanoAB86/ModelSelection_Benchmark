
import scipy
import time
import json
import numpy as np
from scipy import linalg
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from utils import iterative_A
from joblib import Parallel, delayed
import sklearn
from sklearn.decomposition import PCA


class LDA():
    def __init__(self, shrinkage=None, priors=None, n_components=None):
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components

    def _cov(self, X, shrinkage=-1):
        emp_cov = np.cov(np.asarray(X).T, bias=1)
        if shrinkage < 0:
            return emp_cov
        n_features = emp_cov.shape[0]
        mu = np.trace(emp_cov) / n_features
        shrunk_cov = (1.0 - shrinkage) * emp_cov
        shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
        return shrunk_cov

    def softmax(slf, X, copy=True):
        if copy:
            X = np.copy(X)
        max_prob = np.max(X, axis=1).reshape((-1, 1))
        X -= max_prob
        np.exp(X, X)
        sum_prob = np.sum(X, axis=1).reshape((-1, 1))
        X /= sum_prob
        return X

    def iterative_A(self, A, max_iterations=3):
        '''
        calculate the largest eigenvalue of A
        '''
        x = A.sum(axis=1)
        # k = 3
        for _ in range(max_iterations):
            temp = np.dot(A, x)
            y = temp / np.linalg.norm(temp, 2)
            temp = np.dot(A, y)
            x = temp / np.linalg.norm(temp, 2)
        return np.dot(np.dot(x.T, A), y)

    def _solve_eigen2(self, X, y, shrinkage):

        U, S, Vt = np.linalg.svd(np.float32(X), full_matrices=False)

        # solve Ax = b for the best possible approximate solution in terms of least squares
        self.x_hat2 = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y

        y_pred1 = X @ self.x_hat1
        y_pred2 = X @ self.x_hat2

        scores_c = -np.mean((y_pred2 - y) ** 2)
        return scores_c,

    def _solve_eigen(self, X, y, shrinkage):

        classes, y = np.unique(y, return_inverse=True)
        cnt = np.bincount(y)

        # X_ = pairwise_kernels(X, metric='linear')
        X_ = X

        means = np.zeros(shape=(len(classes), X_.shape[1]))
        np.add.at(means, y, X_)
        means /= cnt[:, None]
        self.means_ = means

        cov = np.zeros(shape=(X_.shape[1], X_.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X_[y == group, :]
            cov += self.priors_[idx] * np.atleast_2d(self._cov(Xg))
        self.covariance_ = cov

        Sw = self.covariance_  # within scatter
        if self.shrinkage is None:
            # adaptive regularization strength
            largest_evals_w = self.iterative_A(Sw, max_iterations=3)
            shrinkage = max(np.exp(-5 * largest_evals_w), 1e-10)
            self.shrinkage = shrinkage
        else:
            # given regularization strength
            shrinkage = self.shrinkage
        # print("Shrinkage: {}".format(shrinkage))
        # between scatter
        St = self._cov(X_, shrinkage=self.shrinkage)

        # add regularization on within scatter
        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - self.shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += self.shrinkage * mu

        Sb = St - shrunk_Sw  # between scatter
        # print(shrunk_Sw)
        # evals, evecs = linalg.eigh(Sb, shrunk_Sw)
        # print(np.linalg.inv(shrunk_Sw))

        evals, evecs = np.linalg.eigh(np.linalg.inv(shrunk_Sw) @ Sb)

        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
        self.idx = np.argsort(evals)[0:len(X) // 2]

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def fit(self, X, y):
        '''
        X: input features, N x D
        y: labels, N
        '''
        # X,y,y_reg=self.sample_based_on_classes(X,y,y_reg)
        self.classes_ = np.unique(y)
        # n_samples, _ = X.shape
        n_classes = len(self.classes_)

        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
        self.priors_ = np.bincount(y_t) / float(len(y))
        self._solve_eigen(X, y, shrinkage=self.shrinkage, )

        return self

    def transform(self, X):
        # project X onto Fisher Space
        X_new = np.dot(X, self.scalings_)
        return X_new  # [:, : self._max_components]

    def energy_score(self, logits):
        logits = to_torch(logits)
        return torch.logsumexp(logits, dim=-1).numpy()

    def predict_proba(self, X, y):

        logits = np.dot(X, self.coef_.T) + self.intercept_
        scores = self.softmax(logits)
        return scores

    def sample_based_on_classes(self, X, y, y_reg):
        import random
        X_new = []
        y_new = []

        labels = np.unique(y)
        mean_labels = np.zeros(len(labels))
        for label in labels:
            idx = np.where(y == label)
            X_label = X[idx]
            y_label = y[idx]
            y_label_reg = y_reg[idx]
            mean_labels[label] = np.mean(X_label)

        for label in labels:
            idx = np.where(y == label)
            X_label = X[idx]
            y_label = y[idx]
            y_label_reg = y_reg[idx]
            mean_label = np.mean(X_label)
            dist = 0
            for label_ in labels:
                if label == label_:
                    continue
                dist += np.linalg.norm(X_label - mean_labels[label_], axis=-1)
            idx = np.argsort(dist)[len(X_label) // 3:2 * len(X_label) // 3]
            if label == 0:
                X_new = X_label[idx]
                y_new = y_label[idx]
                y_new_reg = y_label_reg[idx]
            else:
                X_new = np.append(X_new, X_label[idx], axis=0)
                y_new = np.append(y_new, y_label[idx], axis=0)
                y_new_reg = np.append(y_new_reg, y_label_reg[idx], axis=0)
        idx = np.arange(len(X_new))
        random.shuffle(idx)
        return X_new[idx], y_new[idx], y_new_reg[idx]


def LDA_Score(X, y):
    n = len(y)
    num_classes = len(np.unique(y))

    prob = LDA().fit(X, y).predict_proba(X, y)  # p(y|x)
    n = len(y)
    # # # ## leep = E[p(y|x)]. Note: the log function is ignored in case of instability.
    lda_score = np.sum(prob[np.arange(n), y]) / n

    return lda_score


def to_torch(ndarray):
    from collections.abc import Sequence
    if ndarray is None: return None
    if isinstance(ndarray, Sequence):
        return [to_torch(ndarray_) for ndarray_ in ndarray if ndarray_ is not None]
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    if torch.is_tensor(ndarray): return ndarray
    raise ValueError('fail convert')

def Energy_Score(logits,percent,tail):
    logits = to_torch(logits)
    energy_score=torch.logsumexp(logits, dim=-1).numpy()
    if tail=='bot':
        chs = list(np.argsort(energy_score)[0:int(percent*10)*len(energy_score)//1000]) # #
    else:
        chs = list(np.argsort(energy_score)[-int(percent*10)*len(energy_score)//1000:])
    energy_score = energy_score[chs].mean()
    return energy_score


def _cov(X, shrinkage=-1):
    emp_cov = np.cov(np.asarray(X).T, bias=1)
    if shrinkage < 0:
        return emp_cov
    n_features = emp_cov.shape[0]
    mu = np.trace(emp_cov) / n_features
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
    return shrunk_cov


def softmax(X, copy=True):
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X


def _class_means(X, y):

    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]

    means_ = np.zeros(shape=(len(classes), X.shape[1]))
    for i in range(len(classes)):
        means_[i] = (np.sum(means, axis=0) - means[i]) / (len(classes) - 1)
    return means, means_


def split_data(data: np.ndarray, percent_train: float):
    split = data.shape[0] - int(percent_train * data.shape[0])
    return data[:split], data[split:]


def feature_reduce(features: np.ndarray, f: int = None):

    if f is None:
        return features
    if f > features.shape[0]:
        f = features.shape[0]

    return sklearn.decomposition.PCA(
        n_components=f,
        svd_solver='randomized',
        random_state=1919,
        iterated_power=1).fit_transform(features)


class TransferabilityMethod:
    def __call__(self,
                 features: np.ndarray, y: np.ndarray,
                 ) -> float:
        self.features = features
        self.y = y
        return self.forward()

    def forward(self) -> float:
        raise NotImplementedError


class PARC(TransferabilityMethod):

    def __init__(self, n_dims: int = None, fmt: str = ''):
        self.n_dims = n_dims
        self.fmt = fmt

    def forward(self):
        self.features = feature_reduce(self.features, self.n_dims)

        num_classes = len(np.unique(self.y, return_inverse=True)[0])

        if self.y.ndim == 1:
            num_classes = 100  # Assuming you have 100 classes, adjust this value accordingly
            # Check if any label index is out of bounds
            if self.y.ndim == 1:
                num_classes = np.max(self.y) + 1  # Determine the number of classes dynamically
                labels = np.eye(num_classes)[self.y]  # Convert to one-hot encoding
            else:
                labels = self.y
        #labels = np.eye(num_classes)[self.y] if self.y.ndim == 1 else self.y

        return self.get_parc_correlation(self.features, labels)

    def get_parc_correlation(self, feats1, labels2):
        scaler = sklearn.preprocessing.StandardScaler()

        feats1 = scaler.fit_transform(feats1)

        rdm1 = 1 - np.corrcoef(feats1)
        rdm2 = 1 - np.corrcoef(labels2)

        lt_rdm1 = self.get_lowertri(rdm1)
        lt_rdm2 = self.get_lowertri(rdm2)

        return scipy.stats.spearmanr(lt_rdm1, lt_rdm2)[0] * 100

    def get_lowertri(self, rdm):
        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions, 1)]


class SFDA():
    def __init__(self, shrinkage=None, priors=None, n_components=None):
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components

    def _solve_eigen(self, X, y, shrinkage):
        classes, y = np.unique(y, return_inverse=True)
        cnt = np.bincount(y)
        means = np.zeros(shape=(len(classes), X.shape[1]))
        np.add.at(means, y, X)
        means /= cnt[:, None]
        self.means_ = means

        cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X[y == group, :]
            cov += self.priors_[idx] * np.atleast_2d(_cov(Xg))
        self.covariance_ = cov

        Sw = self.covariance_  # within scatter
        if self.shrinkage is None:
            # adaptive regularization strength
            largest_evals_w = iterative_A(Sw, max_iterations=3)
            shrinkage = max(np.exp(-5 * largest_evals_w), 1e-10)
            self.shrinkage = shrinkage
        else:
            # given regularization strength
            shrinkage = self.shrinkage
        print("Shrinkage: {}".format(shrinkage))
        # between scatter
        St = _cov(X, shrinkage=self.shrinkage)

        # add regularization on within scatter
        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - self.shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += self.shrinkage * mu

        Sb = St - shrunk_Sw  # between scatter

        evals, evecs = linalg.eigh(Sb, shrunk_Sw)
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # n_samples, _ = X.shape
        n_classes = len(self.classes_)

        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
        self.priors_ = np.bincount(y_t) / float(len(y))
        self._solve_eigen(X, y, shrinkage=self.shrinkage, )

        return self

    def transform(self, X):
        # project X onto Fisher Space
        X_new = np.dot(X, self.scalings_)
        return X_new[:, : self._max_components]

    def predict_proba(self, X):
        scores = np.dot(X, self.coef_.T) + self.intercept_
        return softmax(scores)


def each_evidence(y_, f, fh, v, s, vh, N, D):
    epsilon = 1e-5
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ np.ascontiguousarray(y_)))
    for _ in range(11):
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + lam)).sum()
        # A = v @ np.diag(alpha + beta * s) @ v.transpose() # no need to compute A
        # A_inv = v @ np.diag(1.0 / (alpha + beta * s)) @ v.transpose() # no need to compute A_inv
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + epsilon)
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + epsilon)
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam
    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * (beta_de + epsilon) \
               - alpha / 2.0 * (alpha_de + epsilon) \
               - N / 2.0 * np.log(2 * np.pi)
    return evidence / N, alpha, beta, m

def truncated_svd(x):
    u, s, vh = np.linalg.svd(x.transpose() @ x)
    s = np.sqrt(s)
    u_times_sigma = x @ vh.transpose()
    k = np.sum((s > 1e-10) * 1)  # rank of f
    s = s.reshape(-1, 1)
    s = s[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh


class LogME(object):
    def __init__(self, regression=False):
        """
            :param regression: whether regression
        """
        self.regression = regression
        self.fitted = False
        self.reset()

    def reset(self):
        self.num_dim = 0
        self.alphas = []  # alpha for each class / dimension
        self.betas = []  # beta for each class / dimension
        # self.ms.shape --> [C, D]
        self.ms = []  # m for each class / dimension

    def _fit_icml(self, f: np.ndarray, y: np.ndarray):
        fh = f
        f = f.transpose()
        D, N = f.shape
        v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            evidence, alpha, beta, m = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    def _fit_fixed_point(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the arxiv 2021 paper
        "Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs"
        at https://arxiv.org/abs/2110.10545
        """
        # k = min(N, D)
        N, D = f.shape

        # direct SVD may be expensive
        if N > D:
            u, s, vh = truncated_svd(f)
        else:
            u, s, vh = np.linalg.svd(f, full_matrices=False)
        # u.shape = N x k, s.shape = k, vh.shape = k x D
        s = s.reshape(-1, 1)
        sigma = (s ** 2)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            y_ = y_.reshape(-1, 1)

            # x has shape [k, 1], but actually x should have shape [N, 1]
            x = u.T @ y_
            x2 = x ** 2
            # if k < N, we compute sum of xi for 0 singular values directly
            res_x2 = (y_ ** 2).sum() - x2.sum()

            alpha, beta = 1.0, 1.0
            for _ in range(11):
                t = alpha / beta
                gamma = (sigma / (sigma + t)).sum()
                m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
                res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
                alpha = gamma / (m2 + 1e-5)
                beta = (N - gamma) / (res2 + 1e-5)
                t_ = alpha / beta
                evidence = D / 2.0 * np.log(alpha) \
                           + N / 2.0 * np.log(beta) \
                           - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                           - beta / 2.0 * res2 \
                           - alpha / 2.0 * m2 \
                           - N / 2.0 * np.log(2 * np.pi)
                evidence /= N
                if abs(t_ - t) / t <= 1e-3:  # abs(t_ - t) <= 1e-5 or abs(1 / t_ - 1 / t) <= 1e-5:
                    break
            evidence = D / 2.0 * np.log(alpha) \
                       + N / 2.0 * np.log(beta) \
                       - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                       - beta / 2.0 * res2 \
                       - alpha / 2.0 * m2 \
                       - N / 2.0 * np.log(2 * np.pi)
            evidence /= N
            m = 1.0 / (t + sigma) * s * x
            m = (vh.T @ m).reshape(-1)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    _fit = _fit_fixed_point

    # _fit = _fit_icml

    def fit(self, f: np.ndarray, y: np.ndarray):

        if self.fitted:
            warnings.warn('re-fitting for new data. old parameters cleared.')
            self.reset()
        else:
            self.fitted = True
        f = f.astype(np.float64)
        if self.regression:
            y = y.astype(np.float64)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        return self._fit(f, y)

    def predict(self, f: np.ndarray):
        """
        :param f: [N, F], feature matrix
        :return: prediction, return shape [N, X]
        """
        if not self.fitted:
            raise RuntimeError("not fitted, please call fit first")
        f = f.astype(np.float64)
        logits = f @ self.ms.T
        if self.regression:
            return logits
        prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        # return np.argmax(logits, axis=-1)
        return prob


def LEEP(X, y, model_name='resnet18'):
    n = len(y)
    num_classes = len(np.unique(y))


    ckpt_loc = ckpt_models[model_name][1]
    fc_weight = ckpt_models[model_name][0]
    fc_bias = fc_weight.replace('weight', 'bias')
    ckpt = torch.load(ckpt_loc, map_location='cpu')
    fc_weight = ckpt[fc_weight].detach().numpy()
    fc_bias = ckpt[fc_bias].detach().numpy()

    # p(z|x), z is source label
    prob = np.dot(X, fc_weight.T) + fc_bias
    prob = softmax(prob)  # p(z|x), N x C(source)

    pyz = np.zeros((num_classes, 1000))  # C(source) = 1000
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        filter_ = np.take(prob, indices, axis=0)
        pyz[y_] = np.sum(filter_, axis=0) / n

    pz = np.sum(pyz, axis=0)  # marginal probability
    py_z = pyz / pz  # conditional probability, C x C(source)
    py_x = np.dot(prob, py_z.T)  # N x C

    # leep = E[p(y|x)]
    leep_score = np.sum(py_x[np.arange(n), y]) / n
    return leep_score



def LEEPvit(X, y, model_name='deit_tiny'):
    n = len(y)
    num_classes = len(np.unique(y))

    # read classifier
    ckpt_models = {#set path
    }

    ckpt_loc = ckpt_models[model_name][1]
    ckpt = torch.load(ckpt_loc)

    # Extract weights and bias
    if 'model' in ckpt:
        if 'norm.weight' in ckpt['model']:
            fc_weight = ckpt['model']['norm.weight'].cpu().detach().numpy()
            fc_bias = ckpt['model']['norm.bias'].cpu().detach().numpy()
        else:
            raise KeyError("Weight and bias not found in the checkpoint.")
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        if 'module.predictor.3.weight' in state_dict:
            fc_weight = state_dict['module.momentum_encoder.norm.weight'].detach().numpy()
            fc_bias = state_dict['module.momentum_encoder.norm.bias'].detach().numpy()
        else:
            raise KeyError("Weight and bias not found in the checkpoint.")
    else:
        if 'norm.weight' in ckpt:
            fc_weight = ckpt['norm.weight'].detach().numpy()
            fc_bias = ckpt['norm.bias'].detach().numpy()
        else:
            raise KeyError("Weight and bias not found in the checkpoint.")

    # Reshape fc_weight to match the number of features in X
    fc_weight = fc_weight[:, None]
    fc_weight = np.repeat(fc_weight, X.shape[1], axis=1)
    fc_weight = fc_weight.T

    # Print shapes for debugging
    print('X:', X.shape)
    print('fc_weight:', fc_weight.shape)
    print('fc_bias:', fc_bias.shape)

    # Calculate probabilities
    prob = np.dot(X, fc_weight) + fc_bias
    prob = softmax(prob)  # Apply softmax

    # Compute conditional probabilities
    pyz = np.zeros((num_classes, fc_weight.shape[1]))
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        filter_ = np.take(prob, indices, axis=0)
        pyz[y_] = np.sum(filter_, axis=0) / n

    pz = np.sum(pyz, axis=0)
    py_z = pyz / pz
    py_x = np.dot(prob, py_z.T)

    # Calculate LEEP score
    leep_score = np.sum(py_x[np.arange(n), y]) / n
    return leep_score

# Define softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def NLEEP(X, y, component_ratio=5):
    n = len(y)
    num_classes = len(np.unique(y))
    # PCA: keep 80% energy
    pca_80 = PCA(n_components=0.8)
    pca_80.fit(X)
    X_pca_80 = pca_80.transform(X)

    # GMM: n_components = component_ratio * class number
    n_components_num = component_ratio * num_classes
    gmm = GaussianMixture(n_components=n_components_num).fit(X_pca_80)
    prob = gmm.predict_proba(X_pca_80)  # p(z|x)

    # NLEEP
    pyz = np.zeros((num_classes, n_components_num))
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        filter_ = np.take(prob, indices, axis=0)
        pyz[y_] = np.sum(filter_, axis=0) / n
    pz = np.sum(pyz, axis=0)
    py_z = pyz / pz
    py_x = np.dot(prob, py_z.T)

    # nleep_score
    nleep_score = np.sum(py_x[np.arange(n), y]) / n
    return nleep_score


def LogME_Score(X, y):

    logme = LogME(regression=False)
    score = logme.fit(X, y)
    return score

class SFDA():
    def __init__(self, shrinkage=None, priors=None, n_components=None):
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components

    def _solve_eigen(self, X, y, shrinkage):
        classes, y = np.unique(y, return_inverse=True)
        cnt = np.bincount(y)
        means = np.zeros(shape=(len(classes), X.shape[1]))
        np.add.at(means, y, X)
        means /= cnt[:, None]
        self.means_ = means

        cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X[y == group, :]
            cov += self.priors_[idx] * np.atleast_2d(_cov(Xg))
        self.covariance_ = cov

        Sw = self.covariance_  # within scatter
        if self.shrinkage is None:
            # adaptive regularization strength
            largest_evals_w = iterative_A(Sw, max_iterations=3)
            shrinkage = max(np.exp(-5 * largest_evals_w), 1e-10)
            self.shrinkage = shrinkage
        else:
            # given regularization strength
            shrinkage = self.shrinkage
        #print("Shrinkage: {}".format(shrinkage))
        # between scatter
        St = _cov(X, shrinkage=self.shrinkage)

        # add regularization on within scatter
        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - self.shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += self.shrinkage * mu

        Sb = St - shrunk_Sw  # between scatter

        evals, evecs = linalg.eigh(Sb, shrunk_Sw)
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def fit(self, X, y):
        '''
        X: input features, N x D
        y: labels, N

        '''
        self.classes_ = np.unique(y)
        # n_samples, _ = X.shape
        n_classes = len(self.classes_)

        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
        self.priors_ = np.bincount(y_t) / float(len(y))
        self._solve_eigen(X, y, shrinkage=self.shrinkage, )

        return self

    def transform(self, X):
        # project X onto Fisher Space
        X_new = np.dot(X, self.scalings_)
        return X_new[:, : self._max_components]

    def predict_proba(self, X):
        scores = np.dot(X, self.coef_.T) + self.intercept_
        return softmax(scores)


def SFDA_Score(X, y):
    n = len(y)
    num_classes = len(np.unique(y))

    SFDA_first = SFDA()
    prob = SFDA_first.fit(X, y).predict_proba(X)  # p(y|x)

    # soften the probability using softmax for meaningful confidential mixture
    prob = np.exp(prob) / np.exp(prob).sum(axis=1, keepdims=True)
    means, means_ = _class_means(X, y)  # class means, outer classes means

    # ConfMix
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        y_prob = np.take(prob, indices, axis=0)
        y_prob = y_prob[:, y_]  # probability of correctly classifying x with label y
        X[indices] = y_prob.reshape(len(y_prob), 1) * X[indices] + \
                     (1 - y_prob.reshape(len(y_prob), 1)) * means_[y_]

    SFDA_second = SFDA(shrinkage=SFDA_first.shrinkage)
    prob = SFDA_second.fit(X, y).predict_proba(X)  # n * num_cls

    # leep = E[p(y|x)]. Note: the log function is ignored in case of instability.
    sfda_score = np.sum(prob[np.arange(n), y]) / n
    return sfda_score


def PARC_Score(X, y, ratio=2):
    num_sample, feature_dim = X.shape
    ndims = 32 if ratio > 1 else int(feature_dim * ratio)  # feature reduction dimension

    if num_sample > 15000:
        from utils import initLabeled
        p = 15000.0 / num_sample
        labeled_index = initLabeled(y, p=p)
        features = X[labeled_index]
        targets = X[labeled_index]
        print("data are sampled to {}".format(features.shape))

    # print("Starting PARC method...")
    method = PARC(n_dims=ndims)
    parc_score = method(features=X, y=y)
    # print("PARC method completed.")

    return parc_score

def PARC_Scorem(X, y, ratio=2, batch_size=5000):
    num_samples, feature_dim = X.shape
    ndims = 32 if ratio > 1 else int(feature_dim * ratio)  # feature reduction dimension
    parc_scores = []

    if num_samples > 15000:
        from utils import initLabeled

        # Iterate over the data in batches
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_X = X[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]

            # Sample the batch if needed
            p = min(15000.0 / batch_X.shape[0], 1.0)
            labeled_index = initLabeled(batch_y, p=p)
            features = batch_X[labeled_index]

            # Apply PARC method to the batch
            method = PARC(n_dims=ndims)
            parc_score = method(features=features, y=batch_y)
            parc_scores.append(parc_score)

        parc_score = np.mean(parc_scores)  # Aggregate scores from all batches
    else:
        # If the number of samples is not large, process the entire data at once
        method = PARC(n_dims=ndims)
        parc_score = method(features=X, y=y)

    return parc_score


def discretize_vector(vec, num_buckets=47):
    # 计算每一块中应该包含的元素数量
    num_bins = num_buckets
    bin_size = len(vec) // num_bins
    # print(bin_size)
    # 对原始向量进行排序
    sorted_vec = vec
    # print(sorted_vec)
    # print(sorted_vec[122],sorted_vec[123])
    # 初始化结果列表
    # print(sorted_vec == vec)
    result = [0] * len(vec)

    # 遍历每一块
    for i in range(num_bins):
        # 计算当前块的起始和结束位置
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size

        # 如果不是最后一块，且剩余元素数量不足以填满一整块，则将多余的元素加入到前面的块中
        if i < num_bins - 1:
            if len(vec) - end_idx < bin_size:
                end_idx = len(vec)
        else:
            # if i == num_bins - 1:
            end_index = len(vec)
        # discretized_vector[start_index:end_index] = i

        # 将当前块中的元素映射为对应的索引值
        for j in range(start_idx, end_idx):
            # print(i,j)
            result[j] = i

    return np.array(result)


def discretize_vector2(vector, num_buckets):
    min_val = np.min(vector)
    max_val = np.max(vector)
    bucket_width = (max_val - min_val) / num_buckets

    bucket_indices = ((vector - min_val) / bucket_width).astype(int)

    return bucket_indices


def coding_rate(Z, eps=1e-4):
    n, d = Z.shape
    # print(n,d)
    # print(Z.min())
    (_, rate) = np.linalg.slogdet((np.eye(d) + 1 / (n * eps) * Z.transpose() @ Z))
    return 0.5 * rate


def sort_with_index(array):
    """
    返回一个按照 array 排序后的索引数组
    """
    return np.argsort(array)


def Transrate(Z, y, eps=1e-4):
    Z = Z - np.mean(Z, axis=0, keepdims=True)
    RZ = coding_rate(Z, eps)
    RZY = 0.
    K = int(y.max() + 1)
    # print(K)
    # score = 0
    for i in range(K):
        # print(i,'i')
        tmp_Z = Z[(y == i).flatten()]
        # print(tmp_Z,i)
        RZY += coding_rate(tmp_Z, eps)
    return (RZ - RZY / K)


def Transrate_multi(Z, Y, eps=1e-4):
    RZ = coding_rate(Z, eps)
    RZY = 0.
    N, dim = Y.shape
    print(N, dim, 'y.shape')
    Y = Y.T

    def process_dim(Z, y):
        # print(y.max())
        y_perdim = y
        num_bins = 50
        y_perdim_regression = discretize_vector(y_perdim, num_bins)
        K = int(y_perdim_regression.max() + 1)
        RZY = 0
        for i in range(K):
            tmp_Z = Z[(y_perdim_regression == i).flatten()]
            RZY += coding_rate(tmp_Z, eps)
            # print((RZ - RZY / K))
        return (RZ - RZY / K)

    # n = 10   #regression2classification
    # score = 0
    results = Parallel(n_jobs=-1)(delayed(process_dim)(Z, y) for y in Y)
    total = np.sum(results)
    return total / dim


def f(x, y):
    i = int(floor(x))
    j = int(floor(y))
    x_frac = x - i
    y_frac = y - j
    h = hilbert(x_frac, y_frac)
    return h + (i + j) * (1 + sqrt(2))


def convert2T(X, Y):
    d1, N = X.shape
    d2, N = Y.shape
    T = np.zeros([d1 * d2, N * N])
    for t1 in range(d1):
        for t2 in range(d2):
            for i in range(N):
                for j in range(N):
                    index_i = t1 * d2 + t2
                    index_j = i * N + j
                    T[index_i][index_j] = X[t1][i] + Y[t2][j] * Y[t2][j]
    return T


from scipy.special import logsumexp

def softmax(x):
    # Normalize the input array to prevent numerical overflow
    x_norm = x - logsumexp(x, axis=0)
    e_x = np.exp(x_norm)
    return e_x / np.sum(e_x, axis=0)


def EMMS_optimal(Z, Y):
    N, d1 = Z.shape
    N, d2 = Y.shape
    # print(Z.shape,Y.shape)
    beta = []
    score = 0
    import numpy as np

    # 构造特征矩阵 X
    # X = np.random.rand(N, D1)

    # 对 X 在第 D1 维度上进行标准化，加入平滑项 1e-8 避免除0错误
    Z_mean = np.mean(Z, axis=0)
    Z_std = np.std(Z, axis=0)
    epsilon = 1e-8
    Z = (Z - Z_mean) / (Z_std + epsilon)

    Y_mean = np.mean(Y, axis=0)
    Y_std = np.std(Y, axis=0)
    epsilon = 1e-8
    Y = (Y - Y_mean) / (Y_std + epsilon)

    coefficients, residuals, rank, singular_values = np.linalg.lstsq(Z, Y, rcond=None)
    # residuals = np.sqrt(np.sum(residuals, axis=1))
    # print(residuals,residuals.shape,coefficients.shape)
    score = sum(residuals) / d2
    print(score)

    return 1 / (score + 0.000001)


import numpy as np


def softmax_t(x, temperature=1.0):
    """带有temperature参数的softmax函数"""
    x = np.asarray(x) / temperature
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def softmax1(x, temperature=1.0):
    """Compute softmax values for each row of x."""
    exp_x = np.exp(x / temperature)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# import numpy as np
from numpy.linalg import lstsq


def sparsemax(z):

    # sort z
    z_sorted = np.sort(z, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, z.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    # use argmax to get the index by row as .nonzero() doesn't
    # take an axis argument. np.argmax return the first index, but the last
    # index is required here, use np.flip to get the last index and
    # `z.shape[axis]` to compensate for np.flip afterwards.
    k_z = z.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(z)
    tau_sum = z_cumsum[np.arange(0, z.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    return np.maximum(0, z - tau_z)


def EMMS(Z, Y):
    x = Z
    y = Y
    N, D2, K = y.shape
    for i in range(K):
        y_mean = np.mean(y[:, :, i], axis=0)
        y_std = np.std(y[:, :, i], axis=0)
        epsilon = 1e-8
        y[:, :, i] = (y[:, :, i] - y_mean) / (y_std + epsilon)
    N, D1 = x.shape
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    epsilon = 1e-8
    x = (x - x_mean) / (x_std + epsilon)
    lam = np.array([1 / K] * K)
    w1 = 0
    lam1 = 0
    T = 0
    b = np.dot(y, lam)
    T = b
    for k in range(1):
        a = x
        w = lstsq(a, b, rcond=None)[0]
        w1 = w
        a = y.reshape(N * D2, K)
        b = np.dot(x, w).reshape(N * D2)
        lam = lstsq(a, b, rcond=None)[0]
        lam = lam.reshape(1, K)
        lam = sparsemax(lam)
        lam = lam.reshape(K, 1)
        lam1 = lam
        b = np.dot(y, lam)
        b = b.reshape(N, D2)
        T = b
    y_pred = np.dot(x, w1)
    res = np.sum((y_pred - T) ** 2) / N * D2

    return -res


import sklearn.decomposition as sd
import sklearn.mixture as sm


def gmm_estimator(features_np_all, label_np_all):
    """Estimate the GMM posterior assignment."""
    pca_model = sd.PCA(n_components=0.8)
    pca_model.fit(features_np_all)
    features_lowdim_train = pca_model.transform(features_np_all)

    num_examples = label_np_all.shape[0]
    y_classes = max([min([label_np_all.max() + 1, int(num_examples * 0.2)]),
                     int(num_examples * 0.1)])
    clf = sm.GaussianMixture(n_components=y_classes)
    clf.fit(features_lowdim_train)
    prob_np_all_gmm = clf.predict_proba(features_lowdim_train)
    return prob_np_all_gmm, features_lowdim_train


def one_hot(a):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1.
    return b


def calculate_pac_dir(features_np_all, label_np_all, alpha=1.):
    """Compute the PACTran-Dirichlet estimator."""
    prob_np_all, _ = gmm_estimator(features_np_all, label_np_all)
    #   starttime = time.time()
    label_np_all = one_hot(label_np_all)  # [n, v]
    soft_targets_sum = np.sum(label_np_all, axis=0)  # [v]
    soft_targets_sum = np.expand_dims(soft_targets_sum, axis=1)  # [v, 1]
    a0 = alpha * soft_targets_sum / np.sum(soft_targets_sum) + 1e-10

    # initialize
    qz = prob_np_all  # [n, d]
    log_s = np.log(prob_np_all + 1e-10)  # [n, d]

    for _ in range(10):
        aw = a0 + np.sum(np.einsum("BY,BZ->BYZ", label_np_all, qz), axis=0)
        logits_qz = (log_s +
                     np.matmul(label_np_all, scipy.special.digamma(aw)) -
                     np.reshape(scipy.special.digamma(np.sum(aw, axis=0)), [1, -1]))
        log_qz = logits_qz - scipy.special.logsumexp(
            logits_qz, axis=-1, keepdims=True)
        qz = np.exp(log_qz)

    log_c0 = scipy.special.loggamma(np.sum(a0)) - np.sum(
        scipy.special.loggamma(a0))
    log_c = scipy.special.loggamma(np.sum(aw, axis=0)) - np.sum(
        scipy.special.loggamma(aw), axis=0)

    pac_dir = np.sum(
        log_c0 - log_c - np.sum(qz * (log_qz - log_s), axis=0))
    pac_dir = -pac_dir / label_np_all.size
    return pac_dir


def calculate_pac_gamma(features_np_all, label_np_all, alpha=1.):
    """Compute the PAC-Gamma estimator."""
    prob_np_all, _ = gmm_estimator(features_np_all, label_np_all)
    #   starttime = time.time()
    label_np_all = one_hot(label_np_all)  # [n, v]
    soft_targets_sum = np.sum(label_np_all, axis=0)  # [v]
    soft_targets_sum = np.expand_dims(soft_targets_sum, axis=1)  # [v, 1]

    a0 = alpha * soft_targets_sum / np.sum(soft_targets_sum) + 1e-10
    beta = 1.

    # initialize
    qz = prob_np_all  # [n, d]
    s = prob_np_all  # [n, d]
    log_s = np.log(prob_np_all + 1e-10)  # [n, d]
    aw = a0
    bw = beta
    lw = np.sum(s, axis=-1, keepdims=True) * np.sum(aw / bw)  # [n, 1]

    for _ in range(10):
        aw = a0 + np.sum(np.einsum("BY,BZ->BYZ", label_np_all, qz),
                         axis=0)  # [v, d]
        lw = np.matmul(
            s, np.expand_dims(np.sum(aw / bw, axis=0), axis=1))  # [n, 1]
        logits_qz = (
                log_s + np.matmul(label_np_all, scipy.special.digamma(aw) - np.log(bw)))
        log_qz = logits_qz - scipy.special.logsumexp(
            logits_qz, axis=-1, keepdims=True)
        qz = np.exp(log_qz)  # [n, a, d]

    pac_gamma = (
            np.sum(scipy.special.loggamma(a0) - scipy.special.loggamma(aw) +
                   aw * np.log(bw) - a0 * np.log(beta)) +
            np.sum(np.sum(qz * (log_qz - log_s), axis=-1) +
                   np.log(np.squeeze(lw, axis=-1)) - 1.))
    pac_gamma /= label_np_all.size
    pac_gamma += 1.
    #   endtime = time.time()
    return pac_gamma


def calculate_pac_gauss(features_np_all, label_np_all,
                        lda_factor=1.):
    """Compute the PAC_Gauss score with diagonal variance."""
    starttime = time.time()
    nclasses = label_np_all.max() + 1
    label_np_all = one_hot(label_np_all)  # [n, v]

    mean_feature = np.mean(features_np_all, axis=0, keepdims=True)
    features_np_all -= mean_feature  # [n,k]

    bs = features_np_all.shape[0]
    kd = features_np_all.shape[-1] * nclasses
    ldas2 = lda_factor * bs  # * features_np_all.shape[-1]
    dinv = 1. / float(features_np_all.shape[-1])

    # optimizing log lik + log prior
    def pac_loss_fn(theta):
        theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

        w = theta[:features_np_all.shape[-1], :]
        b = theta[features_np_all.shape[-1]:, :]
        logits = np.matmul(features_np_all, w) + b

        log_qz = logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        xent = np.sum(np.sum(
            label_np_all * (np.log(label_np_all + 1e-10) - log_qz), axis=-1)) / bs
        loss = xent + 0.5 * np.sum(np.square(w)) / ldas2
        return loss

    # gradient of xent + l2
    def pac_grad_fn(theta):
        theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

        w = theta[:features_np_all.shape[-1], :]
        b = theta[features_np_all.shape[-1]:, :]
        logits = np.matmul(features_np_all, w) + b

        grad_f = scipy.special.softmax(logits, axis=-1)  # [n, k]
        grad_f -= label_np_all
        grad_f /= bs
        grad_w = np.matmul(features_np_all.transpose(), grad_f)  # [d, k]
        grad_w += w / ldas2

        grad_b = np.sum(grad_f, axis=0, keepdims=True)  # [1, k]
        grad = np.ravel(np.concatenate([grad_w, grad_b], axis=0))
        return grad

    # 2nd gradient of theta (elementwise)
    def pac_grad2(theta):
        theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

        w = theta[:features_np_all.shape[-1], :]
        b = theta[features_np_all.shape[-1]:, :]
        logits = np.matmul(features_np_all, w) + b

        prob_logits = scipy.special.softmax(logits, axis=-1)  # [n, k]
        grad2_f = prob_logits - np.square(prob_logits)  # [n, k]
        xx = np.square(features_np_all)  # [n, d]

        grad2_w = np.matmul(xx.transpose(), grad2_f)  # [d, k]
        grad2_w += 1. / ldas2
        grad2_b = np.sum(grad2_f, axis=0, keepdims=True)  # [1, k]
        grad2 = np.ravel(np.concatenate([grad2_w, grad2_b], axis=0))
        return grad2

    kernel_shape = [features_np_all.shape[-1], nclasses]
    theta = np.random.normal(size=kernel_shape) * 0.03
    theta_1d = np.ravel(np.concatenate(
        [theta, np.zeros([1, nclasses])], axis=0))

    theta_1d = scipy.optimize.minimize(
        pac_loss_fn, theta_1d, method="L-BFGS-B",
        jac=pac_grad_fn,
        options=dict(maxiter=100), tol=1e-6).x

    pac_opt = pac_loss_fn(theta_1d)
    endtime_opt = time.time()

    h = pac_grad2(theta_1d)
    sigma2_inv = np.sum(h) * ldas2 / kd + 1e-10
    endtime = time.time()

    if lda_factor == 10.:
        s2s = [1000., 100.]
    elif lda_factor == 1.:
        s2s = [100., 10.]
    elif lda_factor == 0.1:
        s2s = [10., 1.]

    returnv = []
    for s2_factor in s2s:
        s2 = s2_factor * dinv
        pac_gauss = pac_opt + 0.5 * kd / ldas2 * s2 * np.log(
            sigma2_inv)

        # the first item is the pac_gauss metric
        # the second item is the linear metric (without trH)
        returnv += [("pac_gauss_%.1f" % lda_factor, pac_gauss),
                    ("time", endtime - starttime),
                    ("pac_opt_%.1f" % lda_factor, pac_opt),
                    ("time", endtime_opt - starttime)]
    return returnv, theta_1d


def PAC_Score(features_np_all, label_np_all,
              lda_factor):
    """Compute the PAC_Gauss score with diagonal variance."""
    starttime = time.time()
    nclasses = label_np_all.max() + 1
    label_np_all = one_hot(label_np_all)  # [n, v]

    mean_feature = np.mean(features_np_all, axis=0, keepdims=True)
    features_np_all -= mean_feature  # [n,k]

    bs = features_np_all.shape[0]
    kd = features_np_all.shape[-1] * nclasses
    ldas2 = lda_factor * bs  # * features_np_all.shape[-1]
    dinv = 1. / float(features_np_all.shape[-1])

    # optimizing log lik + log prior
    def pac_loss_fn(theta):
        theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

        w = theta[:features_np_all.shape[-1], :]
        b = theta[features_np_all.shape[-1]:, :]
        logits = np.matmul(features_np_all, w) + b

        log_qz = logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        xent = np.sum(np.sum(
            label_np_all * (np.log(label_np_all + 1e-10) - log_qz), axis=-1)) / bs
        loss = xent + 0.5 * np.sum(np.square(w)) / ldas2
        return loss

    # gradient of xent + l2
    def pac_grad_fn(theta):
        theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

        w = theta[:features_np_all.shape[-1], :]
        b = theta[features_np_all.shape[-1]:, :]
        logits = np.matmul(features_np_all, w) + b

        grad_f = scipy.special.softmax(logits, axis=-1)  # [n, k]
        grad_f -= label_np_all
        grad_f /= bs
        grad_w = np.matmul(features_np_all.transpose(), grad_f)  # [d, k]
        grad_w += w / ldas2

        grad_b = np.sum(grad_f, axis=0, keepdims=True)  # [1, k]
        grad = np.ravel(np.concatenate([grad_w, grad_b], axis=0))
        return grad

    # 2nd gradient of theta (elementwise)
    def pac_grad2(theta):
        theta = np.reshape(theta, [features_np_all.shape[-1] + 1, nclasses])

        w = theta[:features_np_all.shape[-1], :]
        b = theta[features_np_all.shape[-1]:, :]
        logits = np.matmul(features_np_all, w) + b

        prob_logits = scipy.special.softmax(logits, axis=-1)  # [n, k]
        grad2_f = prob_logits - np.square(prob_logits)  # [n, k]
        xx = np.square(features_np_all)  # [n, d]

        grad2_w = np.matmul(xx.transpose(), grad2_f)  # [d, k]
        grad2_w += 1. / ldas2
        grad2_b = np.sum(grad2_f, axis=0, keepdims=True)  # [1, k]
        grad2 = np.ravel(np.concatenate([grad2_w, grad2_b], axis=0))
        return grad2

    kernel_shape = [features_np_all.shape[-1], nclasses]
    theta = np.random.normal(size=kernel_shape) * 0.03
    theta_1d = np.ravel(np.concatenate(
        [theta, np.zeros([1, nclasses])], axis=0))

    theta_1d = scipy.optimize.minimize(
        pac_loss_fn, theta_1d, method="L-BFGS-B",
        jac=pac_grad_fn,
        options=dict(maxiter=10), tol=1e-6).x

    pac_opt = pac_loss_fn(theta_1d)
    endtime_opt = time.time()

    h = pac_grad2(theta_1d)
    sigma2_inv = np.sum(h) * ldas2 / kd + 1e-10
    endtime = time.time()

    if lda_factor == 10.:
        s2s = [1000., 100.]
    elif lda_factor == 1.:
        s2s = [100., 10.]
    elif lda_factor == 0.1:
        s2s = [10., 1.]

    returnv = []
    for s2_factor in s2s:
        s2 = s2_factor * dinv
        pac_gauss = pac_opt + 0.5 * kd / ldas2 * s2 * np.log(
            sigma2_inv)

        # the first item is the pac_gauss metric
        # the second item is the linear metric (without trH)
        returnv += [("pac_gauss_%.1f" % lda_factor, pac_gauss),
                    ("time", endtime - starttime),
                    ("pac_opt_%.1f" % lda_factor, pac_opt),
                    ("time", endtime_opt - starttime)]
    return returnv, theta_1d


def NCE(source_labels: np.ndarray, target_labels: np.ndarray):

    C_t = int(np.max(target_labels) + 1)
    C_s = int(np.max(source_labels) + 1)
    N = len(source_labels)

    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for the joint distribution, shape [C_t, C_s]
    for s, t in zip(source_labels, target_labels):
        s = int(s)
        t = int(t)
        joint[t, s] += 1.0 / N
    p_z = joint.sum(axis=0, keepdims=True)

    p_target_given_source = (joint / p_z).T  # P(y | z), shape [C_s, C_t]
    mask = p_z.reshape(-1) != 0  # valid Z, shape [C_s]
    p_target_given_source = p_target_given_source[mask] + 1e-20  # remove NaN where p(z) = 0, add 1e-20 to avoid log (0)
    entropy_y_given_z = np.sum(- p_target_given_source * np.log(p_target_given_source), axis=1, keepdims=True)
    conditional_entropy = np.sum(entropy_y_given_z * p_z.reshape((-1, 1))[mask])

    return -conditional_entropy


def compute_bhattacharyya_distance(mu1, mu2, sigma1, sigma2):
    avg_sigma = (sigma1 + sigma2) / 2
    first_part = np.sum((mu1 - mu2) ** 2 / avg_sigma) / 8
    second_part = np.sum(np.log(avg_sigma))
    second_part -= 0.5 * (np.sum(np.log(sigma1)))
    second_part -= 0.5 * (np.sum(np.log(sigma2)))
    return first_part + 0.5 * second_part


def get_bhattacharyya_distance(per_class_stats, c1, c2, gaussian_type):
    mu1 = per_class_stats[c1]['mean']
    mu2 = per_class_stats[c2]['mean']
    sigma1 = per_class_stats[c1]['variance']
    sigma2 = per_class_stats[c2]['variance']
    if gaussian_type == 'spherical':
        sigma1 = np.mean(sigma1)
        sigma2 = np.mean(sigma2)
    return compute_bhattacharyya_distance(mu1, mu2, sigma1, sigma2)


def compute_per_class_mean_and_variance(features, target_labels, unique_labels):
    per_class_stats = {}
    for label in unique_labels:
        label = int(label)  # For correct indexing
        per_class_stats[label] = {}
        class_ids = target_labels == label
        class_features = features[class_ids]
        mean = np.mean(class_features, axis=0)
        variance = np.var(class_features, axis=0)
        per_class_stats[label]['mean'] = mean
        # Avoid 0 variance in cases of constant features with np.maximum
        per_class_stats[label]['variance'] = np.maximum(variance, 1e-4)
    return per_class_stats


# def get_gbc_score(features, target_labels, gaussian_type):
def gbc(features, target_labels, gaussian_type):
    assert gaussian_type in ('diagonal', 'spherical')
    unique_labels = np.unique(target_labels)
    per_class_stats = compute_per_class_mean_and_variance(
        features, target_labels, unique_labels)

    per_class_bhattacharyya_distance = []
    for c1 in unique_labels:
        temp_metric = []
        for c2 in unique_labels:
            if c1 != c2:
                bhattacharyya_distance = get_bhattacharyya_distance(
                    per_class_stats, int(c1), int(c2), gaussian_type)
                temp_metric.append(np.exp(-bhattacharyya_distance))
        per_class_bhattacharyya_distance.append(np.sum(temp_metric))
    gbc = -np.sum(per_class_bhattacharyya_distance)

    return gbc


def getCov(X):
    X_mean = X - np.mean(X, axis=0, keepdims=True)
    cov = np.divide(np.dot(X_mean.T, X_mean), len(X) - 1)
    return cov


def Hscore(f, Z):
    # Z=np.argmax(Z, axis=1)
    Covf = getCov(f)
    alphabetZ = list(set(Z))
    g = np.zeros_like(f)
    for z in alphabetZ:
        Ef_z = np.mean(f[Z == z, :], axis=0)
        g[Z == z] = Ef_z

    Covg = getCov(g)
    score = np.trace(np.dot(np.linalg.pinv(Covf, rcond=1e-15), Covg))
    return score

def NewEMMS(X, Y):
    N, D1 = X.shape
    N, D2, K = Y.shape

    print(N, D1, D2, K)

    # Reshape Y to have a second dimension that is a multiple of D1
    Y = Y.reshape(N, D1, -1).mean(axis=-1)

    print(X.shape, Y.shape)

    # Normalize the features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    epsilon = 1e-8
    X = (X - X_mean) / (X_std + epsilon)

    Y_mean = np.mean(Y, axis=0)
    Y_std = np.std(Y, axis=0)
    epsilon = 1e-8
    Y = (Y - Y_mean) / (Y_std + epsilon)

    # Compute the linear regression coefficients
    coefficients, _, _, _ = lstsq(Y.T, X.T, rcond=None)

    # Compute the residuals and mean squared error
    residuals = X.T - Y.T @ coefficients
    mse = np.mean(residuals ** 2)

    # Compute the correlation coefficient between X and Y
    corr_coef = np.abs(np.corrcoef(X.T, Y.T)[0, 1])  # Use absolute value to handle negative correlation

    # Compute the energy score (similar to ETran)
    logits = coefficients @ Y
    energy_score = np.mean(np.log(np.sum(np.exp(logits), axis=1)))

    # Combine the energy score and EMMS score
    w_energy = 0.9  # Weight for energy score
    w_emms = 0.1  # Weight for EMMS score

    # Combine the scores
    AMscore = (w_energy * energy_score) + (w_emms * (1 / mse))

    print(AMscore)

    return AMscore


import numpy as np

DATA_LIMIT = 45000


def LFC(target_x, target_y):
    if target_x.shape[0] > DATA_LIMIT:
        sampled_index = torch.randperm(target_x.shape[0])[:DATA_LIMIT]
        target_x = target_x[sampled_index]
        target_y = target_y[sampled_index]

    target_y = torch.tensor(target_y)  # Convert to PyTorch tensor
    y = F.one_hot(target_y)
    y = y @ y.T
    y[y == 0] = -1
    #print('shape y', y.shape)
    y = y.float()

    target_x = torch.tensor(target_x)  # Convert to PyTorch tensor
    x = target_x @ target_x.T

    y = y - torch.mean(y).item()
    x = x - torch.mean(x).item()

    return torch.sum(torch.mul(x, y)).item() / (
                torch.sqrt(torch.sum(torch.mul(x, x))).item() * torch.sqrt(torch.sum(torch.mul(y, y))).item())




def NCE1(source_label: np.ndarray, target_label: np.ndarray):
    # Compute the number of unique classes in source and target labels
    C_s = int(np.max(source_label) + 1)  # the number of source classes
    C_t = int(np.max(target_label) + 1)  # the number of target classes

    # Initialize the joint array with zeros
    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for the joint distribution

    # Iterate over source_label and target_label to populate the joint array
    N = len(source_label)
    for s, t in zip(source_label, target_label):
        # Convert negative indices to positive using modular arithmetic
        s = int(s) % C_s
        t = int(t) % C_t
        # Increment the corresponding entry in the joint array
        joint[t, s] += 1.0 / N

    # Compute conditional entropy
    p_z = joint.sum(axis=0, keepdims=True)  # shape [1, C_s]
    p_target_given_source = (joint / p_z).T
    mask = p_z.reshape(-1) != 0
    p_target_given_source = p_target_given_source[mask] + 1e-20
    entropy_y_given_z = np.sum(- p_target_given_source * np.log(p_target_given_source), axis=1,
                               keepdims=True)  # shape [C_s, 1]
    conditional_entropy = np.sum(entropy_y_given_z * p_z.reshape((-1, 1))[mask])  # scalar
    return -conditional_entropy

import torch
import torch.nn.functional as F
import numpy as np

DATA_LIMIT = 45000

def LFC(target_x, target_y):
    if target_x.shape[0] > DATA_LIMIT:
        sampled_index = torch.randperm(target_x.shape[0])[:DATA_LIMIT]
        target_x = target_x[sampled_index]
        target_y = target_y[sampled_index]

    target_y = torch.tensor(target_y)  # Convert to PyTorch tensor
    y = F.one_hot(target_y)
    y = y @ y.T
    y[y == 0] = -1
    #print('shape y', y.shape)
    y = y.float()

    target_x = torch.tensor(target_x)  # Convert to PyTorch tensor
    x = target_x @ target_x.T

    y = y - torch.mean(y).item()
    x = x - torch.mean(x).item()

    return torch.sum(torch.mul(x, y)).item() / (
                torch.sqrt(torch.sum(torch.mul(x, x))).item() * torch.sqrt(torch.sum(torch.mul(y, y))).item())


class LDA():
    def __init__(self, shrinkage=None, priors=None, n_components=None):
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components

    def _cov(self, X, shrinkage=-1):
        emp_cov = np.cov(np.asarray(X).T, bias=1)
        if shrinkage < 0:
            return emp_cov
        n_features = emp_cov.shape[0]
        mu = np.trace(emp_cov) / n_features
        shrunk_cov = (1.0 - shrinkage) * emp_cov
        shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
        return shrunk_cov

    def softmax(slf, X, copy=True):
        if copy:
            X = np.copy(X)
        max_prob = np.max(X, axis=1).reshape((-1, 1))
        X -= max_prob
        np.exp(X, X)
        sum_prob = np.sum(X, axis=1).reshape((-1, 1))
        X /= sum_prob
        return X

    def iterative_A(self, A, max_iterations=3):
        '''
        calculate the largest eigenvalue of A
        '''
        x = A.sum(axis=1)
        # k = 3
        for _ in range(max_iterations):
            temp = np.dot(A, x)
            y = temp / np.linalg.norm(temp, 2)
            temp = np.dot(A, y)
            x = temp / np.linalg.norm(temp, 2)
        return np.dot(np.dot(x.T, A), y)

    def _solve_eigen2(self, X, y, shrinkage):

        U, S, Vt = np.linalg.svd(np.float32(X), full_matrices=False)

        # solve Ax = b for the best possible approximate solution in terms of least squares
        self.x_hat2 = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ y

        y_pred1 = X @ self.x_hat1
        y_pred2 = X @ self.x_hat2

        scores_c = -np.mean((y_pred2 - y) ** 2)
        return scores_c,

    def _solve_eigen(self, X, y, shrinkage):

        classes, y = np.unique(y, return_inverse=True)
        cnt = np.bincount(y)

        # X_ = pairwise_kernels(X, metric='linear')
        X_ = X

        means = np.zeros(shape=(len(classes), X_.shape[1]))
        np.add.at(means, y, X_)
        means /= cnt[:, None]
        self.means_ = means

        cov = np.zeros(shape=(X_.shape[1], X_.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X_[y == group, :]
            cov += self.priors_[idx] * np.atleast_2d(self._cov(Xg))
        self.covariance_ = cov

        Sw = self.covariance_  # within scatter
        if self.shrinkage is None:
            # adaptive regularization strength
            largest_evals_w = self.iterative_A(Sw, max_iterations=3)
            shrinkage = max(np.exp(-5 * largest_evals_w), 1e-10)
            self.shrinkage = shrinkage
        else:
            # given regularization strength
            shrinkage = self.shrinkage
        # print("Shrinkage: {}".format(shrinkage))
        # between scatter
        St = self._cov(X_, shrinkage=self.shrinkage)

        # add regularization on within scatter
        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - self.shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += self.shrinkage * mu

        Sb = St - shrunk_Sw  # between scatter
        # print(shrunk_Sw)
        # evals, evecs = linalg.eigh(Sb, shrunk_Sw)
        # print(np.linalg.inv(shrunk_Sw))

        evals, evecs = np.linalg.eigh(np.linalg.inv(shrunk_Sw) @ Sb)

        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
        self.idx = np.argsort(evals)[0:len(X) // 2]

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def fit(self, X, y):
        '''
        X: input features, N x D
        y: labels, N
        '''
        # X,y,y_reg=self.sample_based_on_classes(X,y,y_reg)
        self.classes_ = np.unique(y)
        # n_samples, _ = X.shape
        n_classes = len(self.classes_)

        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
        self.priors_ = np.bincount(y_t) / float(len(y))
        self._solve_eigen(X, y, shrinkage=self.shrinkage, )

        return self

    def transform(self, X):
        # project X onto Fisher Space
        X_new = np.dot(X, self.scalings_)
        return X_new  # [:, : self._max_components]

    def energy_score(self, logits):
        logits = to_torch(logits)
        return torch.logsumexp(logits, dim=-1).numpy()

    def predict_proba(self, X, y):

        logits = np.dot(X, self.coef_.T) + self.intercept_
        scores = self.softmax(logits)
        return scores

    def sample_based_on_classes(self, X, y, y_reg):
        import random
        X_new = []
        y_new = []

        labels = np.unique(y)
        mean_labels = np.zeros(len(labels))
        for label in labels:
            idx = np.where(y == label)
            X_label = X[idx]
            y_label = y[idx]
            y_label_reg = y_reg[idx]
            mean_labels[label] = np.mean(X_label)

        for label in labels:
            idx = np.where(y == label)
            X_label = X[idx]
            y_label = y[idx]
            y_label_reg = y_reg[idx]
            mean_label = np.mean(X_label)
            dist = 0
            for label_ in labels:
                if label == label_:
                    continue
                dist += np.linalg.norm(X_label - mean_labels[label_], axis=-1)
            idx = np.argsort(dist)[len(X_label) // 3:2 * len(X_label) // 3]
            if label == 0:
                X_new = X_label[idx]
                y_new = y_label[idx]
                y_new_reg = y_label_reg[idx]
            else:
                X_new = np.append(X_new, X_label[idx], axis=0)
                y_new = np.append(y_new, y_label[idx], axis=0)
                y_new_reg = np.append(y_new_reg, y_label_reg[idx], axis=0)
        idx = np.arange(len(X_new))
        random.shuffle(idx)
        return X_new[idx], y_new[idx], y_new_reg[idx]



def load_score(path):
    with open(path, 'r') as f:
        score_ = json.load(f)
    time = score_['duration'] if 'duration' in score_.keys() else 0
    score = {a[0]: a[1] for a in score_.items() if a[0] != 'duration'}
    return score, time



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def NCTI_Score(X, y):
    C = np.unique(y).shape[0]
    pca = PCA(n_components=64)
    X = pca.fit_transform(X, y)
    # model_npy_feature = os.path.join('./results_f/group1/pca_feature', f'{args.model}_{args.dataset}_feature.npy')
    # np.save(model_npy_feature, X)
    temp = max(np.exp(-pca.explained_variance_[:32].sum()), 1e-10)
    print(pca.explained_variance_[:32].sum() / pca.explained_variance_.sum())

    if temp == 1e-10:
        clf = LinearDiscriminantAnalysis(solver='svd')

    else:
        clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage=float(temp))

    low_feat = clf.fit_transform(X, y)

    low_feat = low_feat - np.mean(low_feat, axis=0, keepdims=True)
    all_lowfeat_nuc = np.linalg.norm(low_feat, ord='nuc')

    low_pred = clf.predict_proba(X)
    sfda_score = np.sum(low_pred[np.arange(X.shape[0]), y]) / X.shape[0]
    print(clf.score(X, y))

    class_pred_nuc = 0
    class_low_feat = np.zeros((C, 1))
    print(class_low_feat.shape)
    for c in range(C):
        c_pred = low_pred[(y == c).flatten()]
        c_pred_nuc = np.linalg.norm(c_pred, ord='nuc')
        class_pred_nuc += c_pred_nuc
    print("all feat nuc: " + str(all_lowfeat_nuc))
    print("class res nuc: " + str((class_pred_nuc)))
    print("pred: " + str((sfda_score)))
    #return all_lowfeat_nuc, sfda_score, np.log(class_pred_nuc)
    return sfda_score


def Hscore1(features: np.ndarray, labels: np.ndarray):

    f = features
    y = labels

    def covariance(X):
        X_mean = X - np.mean(X, axis=0, keepdims=True)
        cov = np.divide(np.dot(X_mean.T, X_mean), len(X) - 1)
        return cov

    covf = covariance(f)
    C = int(y.max() + 1)
    g = np.zeros_like(f)

    for i in range(C):
        Ef_i = np.mean(f[y == i, :], axis=0)
        g[y == i] = Ef_i

    covg = covariance(g)
    score = np.trace(np.dot(np.linalg.pinv(covf, rcond=1e-15), covg))

    return score
