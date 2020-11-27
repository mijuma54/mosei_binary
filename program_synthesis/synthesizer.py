import numpy as np

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing as mp
import concurrent.futures as cf
import parmap
import cupy as cp

def random_combination(m, n, k): #m combinations from nCk
    out = []
    check = dict()
    while len(out) < 1000:#m:
        temp = np.random.choice(range(n), k, replace=False)
        if str(temp) not in check:
            out.append(np.array(temp))
            check[str(temp)] = 0
    return out


def all_pair_dist2_cuda(X1, X2, feat):
    n1 = len(X1)
    n2 = len(X2)
    nf = len(feat)
    feat = cp.array(feat)
    X1 = cp.array(X1.reshape(1, n1, -1))
    X2 = cp.array(X2.reshape(1, n2, -1))

    mat1 = cp.repeat(X1, n2, axis=0)
    mat2 = cp.repeat(X2.reshape(n2, 1, nf), n1, axis=1)

    isbow = cp.repeat(cp.repeat((feat < 2100).reshape(1, 1, nf), n1, axis=1), n2, axis=0)

    count_mat = nf - cp.sum((mat1 == 0) & (mat2 == 0) & isbow, axis=2)
    zeros = (count_mat != 0)
    #count_mat[zeros] = 1

    dist_mat = np.ones_like(count_mat)
    dist_mat[zeros] = np.sum(np.abs(mat1 - mat2), axis=2)[zeros] / count_mat[zeros]
    #dist_mat[zeros] = 1

    return dist_mat

def all_pair_dist2(X1, X2, feat):
    n1 = len(X1)
    n2 = len(X2)
    nf = len(feat)
    feat = np.array(feat)
    X1 = np.array(X1.reshape(1, n1, -1))
    X2 = np.array(X2.reshape(1, n2, -1))

    mat1 = np.repeat(X1, n2, axis=0)
    mat2 = np.repeat(X2.reshape(n2, 1, nf), n1, axis=1)

    isbow = np.repeat(np.repeat((feat < 2100).reshape(1, 1, nf), n1, axis=1), n2, axis=0)

    count_mat = nf - np.sum((mat1 == 0) & (mat2 == 0) & isbow, axis=2)
    zeros = (count_mat != 0)
    #count_mat[zeros] = 1

    dist_mat = np.ones_like(count_mat)
    dist_mat[zeros] = np.sum(np.abs(mat1 - mat2), axis=2)[zeros] / count_mat[zeros]
    #dist_mat[zeros] = 1

    return dist_mat



class dummyKneighborClassifier(object):
    def __init__(self, n):
        self.n = n
    def predict_proba(self, dist_mat, val_ground):
        dist_mat = np.array(dist_mat)
        val_ground = np.array(val_ground)

        neigh_ground = val_ground[np.argsort(dist_mat, axis=1)[:, -self.n:]]
        marginals = np.ones((dist_mat.shape[0], 2))
        marginals[:, 1] = np.sum(neigh_ground == 1, axis=1) / self.n
        marginals[:, 0] -= np.sum(neigh_ground == 1, axis=1) / self.n


        return marginals

class Synthesizer(object):
    """
    A class to synthesize heuristics from primitives and validation labels
    """

    def __init__(self, primitive_matrix, val_ground, b=0.5):
        """
        Initialize Synthesizer object
        b: class prior of most likely class
        beta: threshold to decide whether to abstain or label for heuristics
        """
        self.val_primitive_matrix = primitive_matrix
        self.val_ground = val_ground
        self.p = np.shape(self.val_primitive_matrix)[1]
        self.b = b

    def generate_feature_combinations(self, cardinality=1):
        """
        Create a list of primitive index combinations for given cardinality
        max_cardinality: max number of features each heuristic operates over
        """
        # primitive_idx = range(self.p)
        # feature_combinations = []
        #
        # for comb in itertools.combinations(primitive_idx, cardinality):
        #     feature_combinations.append(comb)

        feature_combinations = random_combination(self.p, self.p, cardinality)

        return feature_combinations

    def fit_function(self, comb, model):
        """
        Fits a single logistic regression or decision tree model
        comb: feature combination to fit model over
        model: fit logistic regression or a decision tree
        """
        X = self.val_primitive_matrix[:, comb]
        if np.shape(X)[0] == 1:
            X = X.reshape(-1, 1)

        # fit decision tree or logistic regression or knn
        if model == 'dt':
            dt = DecisionTreeClassifier(max_depth=len(comb))
            dt.fit(X, self.val_ground)
            return dt

        elif model == 'lr':
            lr = LogisticRegression()
            lr.fit(X, self.val_ground)
            return lr

        elif model == 'nn':
            # nn = KNeighborsClassifier(algorithm='auto', metric=mixed_dist)
            # nn.fit(X, self.val_ground)

            nn = dummyKneighborClassifier(20)

            return nn

    def fit_and_return(self, comb, model):
        return self.fit_function(comb, model)

    #@profile
    def generate_heuristics(self, model,min_cardinality=1, max_cardinality=1):
        """
        Generates heuristics over given feature cardinality
        model: fit logistic regression or a decision tree
        max_cardinality: max number of features each heuristic operates over
        """
        # have to make a dictionary?? or feature combinations here? or list of arrays?
        feature_combinations_final = []
        heuristics_final = []

        feature_length = 0
        for cardinality in range(min_cardinality, max_cardinality + 1):
            feature_combinations = self.generate_feature_combinations(cardinality)
            #######single-core
            # heuristics = []
            # for i, comb in enumerate(feature_combinations):
            #     heuristics.append(self.fit_function(comb, model))

            ##########with future
            # with cf.ThreadPoolExecutor(max_workers=8) as exe:
            #     future_to_comb = {exe.submit(self.fit_and_return, comb, model): comb for comb in feature_combinations}
            #     for future in cf.as_completed(future_to_comb):
            #         #comb = future_to_comb[future]
            #         heuristics.append(future.result())


            #########with parmap
            heuristics = parmap.map(self.fit_and_return, feature_combinations, model, pm_pbar=True)


            feature_combinations_final.append(feature_combinations)
            heuristics_final.append(heuristics)

        return heuristics_final, feature_combinations_final

    def beta_optimizer(self, marginals, ground):
        """
        Returns the best beta parameter for abstain threshold given marginals
        Uses F1 score that maximizes the F1 score
        marginals: confidences for data from a single heuristic
        """

        # Set the range of beta params
        # 0.25 instead of 0.0 as a min makes controls coverage better
        beta_params = np.linspace(0.25, 0.45, 10)

        f1 = []

        for beta in beta_params:
            labels_cutoff = np.zeros(np.shape(marginals))
            labels_cutoff[marginals <= (self.b - beta)] = -1.
            labels_cutoff[marginals >= (self.b + beta)] = 1.
            f1.append(f1_score(ground, labels_cutoff, average='weighted'))

        f1 = np.nan_to_num(f1)
        return beta_params[np.argsort(np.array(f1))[-1]]


    def find_dist_and_proba_and_beta(self, hf, feat, X,  ground):
        X_ = all_pair_dist2(self.val_primitive_matrix[:, feat], X[:, feat], feat)
        marginals = hf.predict_proba(X_, self.val_ground)[:, 1]
        return self.beta_optimizer(marginals, ground)

    def find_proba_and_beta(self, hf, feat, X, ground):
        marginals = hf.predict_proba(X[:, feat])[:, 1]
        return self.beta_optimizer(marginals, ground)

    #@profile
    def find_optimal_beta(self, heuristics,  X, feat_combos, ground):
        """
        Returns optimal beta for given heuristics
        heuristics: list of pre-trained logistic regression models
        X: primitive matrix
        feat_combos: feature indices to apply heuristics to
        ground: ground truth associated with X data
        """

        # beta_opt = np.zeros(len(heuristics))
        #
        # if f'{heuristics[0].__class__}' == "<class 'program_synthesis.synthesizer.dummyKneighborClassifier'>":
        #     with cf.ThreadPoolExecutor(max_workers=8) as exe:
        #         future_to_index = {exe.submit(self.find_dist_and_proba_and_beta, i[1], feat_combos[i[0]], X, ground): i[0]
        #                            for i in enumerate(heuristics)}
        #         for future in cf.as_completed(future_to_index):
        #             i = future_to_index[future]
        #             beta_opt[i] = future.result()
        #
        # else:
        #     with cf.ThreadPoolExecutor(max_workers=8) as exe:
        #         future_to_index = {exe.submit(self.find_proba_and_beta, i[1], X, feat_combos[i[0]]): i[0]
        #                            for i in enumerate(heuristics)}
        #         for future in cf.as_completed(future_to_index):
        #             i = future_to_index[future]
        #             beta_opt[i] = future.result()

        if f'{heuristics[0].__class__}' == "<class 'program_synthesis.synthesizer.dummyKneighborClassifier'>":
            ########single-core
            beta_opt = []
            for i, hf in enumerate(heuristics):
                X_ = all_pair_dist2(self.val_primitive_matrix[:, feat_combos[i]], X[:, feat_combos[i]], feat_combos[i])
                marginals = hf.predict_proba(X_, self.val_ground)[:, 1]
                beta_opt.append((self.beta_optimizer(marginals, ground)))

        else:
            #######with parmap
            beta_opt = parmap.starmap(self.find_proba_and_beta, list(zip(heuristics, feat_combos)), X, ground, pm_pbar=True)

        return beta_opt