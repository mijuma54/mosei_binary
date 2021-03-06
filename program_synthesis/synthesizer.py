import numpy as np

#from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing as mp
import concurrent.futures as cf
import parmap
import cupy as cp
from sklearn.metrics.pairwise import cosine_similarity
from numba import jit
import time

def f1_score(ground, pred):
    ground = ground.reshape(-1)
    pred = pred.reshape(-1)

    if np.sum(pred != 0) == 0:
        precision = 0
    else:
        precision = np.sum(ground == pred)/np.sum(pred != 0)
    recall = np.sum(ground == pred)/len(ground)

    if precision + recall == 0:
        return 0
    else:
        return 2*precision*recall/(precision+recall)

def f1_score_cuda(ground, pred):
    ground = cp.array(ground).reshape(-1)
    pred = cp.array(pred).reshape(-1)

    if cp.sum(pred != 0) == 0:
        precision = 0
    else:
        precision = cp.sum(ground == pred)/cp.sum(pred != 0)
    recall = cp.sum(ground == pred)/len(ground)

    if precision + recall == 0:
        return 0
    else:
        return 2*precision*recall/(precision+recall)



def random_combination(m, n, k): #m combinations from nCk
    out = []
    check = dict()
    while len(out) < 100:#m:
        temp = np.random.choice(range(n), k, replace=False)
        if str(temp) not in check:
            out.append(np.array(temp))
            check[str(temp)] = 0
    return out


def all_pair_dist_cuda(X1, X2, feat, metric='cosine'):
    if metric=='cosine':
        norm1 = cp.einsum('ij, ij->i', X1, X1)
        norm1 = cp.sqrt(norm1, norm1).reshape(-1, 1)

        norm2 = cp.einsum('ij, ij->i', X2, X2)
        norm2 = cp.sqrt(norm2, norm2).reshape(-1, 1)

        return cp.dot(X2/norm2, (X1/norm1).T)
    else:
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

        dist_mat = cp.ones_like(count_mat)
        dist_mat[zeros] = cp.sum(np.cbs(mat1 - mat2), axis=2)[zeros] / count_mat[zeros]

        return cp.asnumpy(dist_mat)

# @jit
def all_pair_dist(X1, X2, feat, metric='cosine'):
    if metric=='cosine':
        return cosine_similarity(X2, X1)
    else:
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
        nonzeros = (count_mat != 0)

        dist_mat = np.ones_like(count_mat)
        dist_mat[nonzeros] = np.sum(np.abs(mat1 - mat2), axis=2)[nonzeros] / count_mat[nonzeros]

        return dist_mat



class dummyKneighborClassifier(object):
    def __init__(self, n):
        self.n = n
    # @profile
    def predict_proba_cuda(self, dist_mat, val_ground, metric='cosine'):



        # dist_mat = cp.array(dist_mat)
        val_ground = cp.array(val_ground)
        if metric == 'cosine':
            neigh = cp.argpartition(dist_mat, self.n, axis=1)[:, :self.n]
            neigh_ground = val_ground[neigh]

        else:
            neigh_ground = val_ground[cp.argsort(dist_mat, axis=1)[:, -self.n:]]

        marginals = cp.ones((dist_mat.shape[0], 2))
        marginals[:, 1] = cp.sum(neigh_ground == 1, axis=1) / self.n

        return marginals[:, 1]
    # @jit
    def predict_proba(self, dist_mat, val_ground, metric='cosine'):
        if metric=='cosine':
            neigh_ground = val_ground[np.argpartition(dist_mat, -self.n, axis=1)[:, :self.n]]
        else:


            neigh_ground = val_ground[np.argsort(dist_mat, axis=1)[:, -self.n:]]
        marginals = np.ones((dist_mat.shape[0], 2))
        marginals[:, 1] = np.sum(neigh_ground == 1, axis=1) / self.n
        #marginals[:, 0] -= np.sum(neigh_ground == 1, axis=1) / self.n


        return marginals[:, 1]

class Synthesizer(object):
    """
    A class to synthesize heuristics from primitives and validation labels
    """

    def __init__(self, primitive_matrix, val_ground, b=0.5, cuda=False):
        """
        Initialize Synthesizer object
        b: class prior of most likely class
        beta: threshold to decide whether to abstain or label for heuristics
        """
        self.val_primitive_matrix = primitive_matrix
        self.val_ground = val_ground
        self.p = np.shape(self.val_primitive_matrix)[1]
        self.b = b
        self.cuda = cuda
        self.betas = []
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

    ## @profile
    # @jit
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



            ########with parmap
            heuristics = parmap.map(self.fit_and_return, feature_combinations, model, pm_pbar=True)





            feature_combinations_final.append(feature_combinations)
            heuristics_final.append(heuristics)

        return heuristics_final, feature_combinations_final
    # @jit
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
            f1.append(f1_score(ground, labels_cutoff))

        f1 = np.nan_to_num(f1)
        return beta_params[np.argpartition(np.array(f1), -1)[-1]]

    def beta_optimizer_cuda(self, marginals, ground):
        """
        Returns the best beta parameter for abstain threshold given marginals
        Uses F1 score that maximizes the F1 score
        marginals: confidences for data from a single heuristic
        """

        # Set the range of beta params
        # 0.25 instead of 0.0 as a min makes controls coverage better
        beta_params = cp.linspace(0.25, 0.45, 10)

        f1 = []

        for beta in beta_params:
            labels_cutoff = cp.zeros(np.shape(marginals))
            labels_cutoff[marginals <= (self.b - beta)] = -1.
            labels_cutoff[marginals >= (self.b + beta)] = 1.
            f1.append(cp.asnumpy(f1_score_cuda(ground, labels_cutoff)))


        f1 = np.nan_to_num(f1)
        return beta_params[np.argpartition(np.array(f1), -1)[-1]]


    def find_dist_and_proba_and_beta(self, hf, feat, X,  ground):
        X_ = all_pair_dist(self.val_primitive_matrix[:, feat], X[:, feat], feat)
        marginals = hf.predict_proba(X_, self.val_ground)
        beta_opt = self.beta_optimizer(marginals, ground)
        return beta_opt

    def find_proba_and_beta(self, hf, feat, X, ground):
        marginals = hf.predict_proba(X[:, feat])
        beta_opt = self.beta_optimizer(marginals, ground)
        return beta_opt

    # @profile
    def find_optimal_beta(self, heuristics,  X, feat_combos, ground):
        """
        Returns optimal beta for given heuristics
        heuristics: list of pre-trained logistic regression models
        X: primitive matrix
        feat_combos: feature indices to apply heuristics to
        ground: ground truth associated with X data
        """


        if f'{heuristics[0].__class__}' == "<class 'program_synthesis.synthesizer.dummyKneighborClassifier'>":
            ########single-core, gpu
            if self.cuda:
                beta_opt = []
                # for i, hf in enumerate(heuristics):
                #     X_.append(all_pair_dist_cuda(self.val_primitive_matrix[:, feat_combos[i]], X[:, feat_combos[i]], feat_combos[i]))
                #
                # marginals = parmap.map(heuristics[0].predict_proba, X_, self.val_ground)
                #
                # for i, hf in enumerate(heuristics):
                #     beta_opt.append((self.beta_optimizer(marginals[i], ground)))

                for i, hf in enumerate(heuristics):
                    X_ = all_pair_dist_cuda(cp.array(self.val_primitive_matrix[:, feat_combos[i]]), cp.array(X[:, feat_combos[i]]), feat_combos[i])
                    marginals = hf.predict_proba_cuda(X_, self.val_ground)
                    beta_opt.append(self.beta_optimizer_cuda(marginals, ground))
                    self.betas.append = beta_opt[-1]
            else:

                #######with parmap
                # beta_opt = parmap.starmap(self.find_dist_and_proba_and_beta, list(zip(heuristics, feat_combos)), X, ground,
                #                           pm_pbar=True)
                # self.betas = beta_opt

                #######single-core, numba
                beta_opt = []
                for i, hf in enumerate(heuristics):
                    #print(i, end='\r')
                    X_ = all_pair_dist(self.val_primitive_matrix[:, feat_combos[i]], X[:, feat_combos[i]], feat_combos[i])
                    marginals = hf.predict_proba(X_, self.val_ground)
                    beta_opt.append(self.beta_optimizer(marginals, ground))
                    self.betas.append = beta_opt[-1]
        else:
            #######with parmap
            beta_opt = parmap.starmap(self.find_proba_and_beta, list(zip(heuristics, feat_combos)), X, ground, pm_pbar=True)
            self.betas = beta_opt
        return beta_opt


    def load_optimal_beta(self, feat_idx):
        beta_opt = []
        for i in feat_idx:
            beta_opt.append(self.betas[i])
        return beta_opt