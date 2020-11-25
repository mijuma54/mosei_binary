import numpy as np
import pandas as pd
#from sklearn.metrics import f1_score

from program_synthesis.synthesizer import Synthesizer
from program_synthesis.verifier import Verifier
import concurrent.futures as cf

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

@profile
def all_pair_dist(X1, X2, feat):
    n1 = len(X1)
    n2 = len(X2)
    nf = len(feat)
    feat = np.array(feat)
    X1 = X1.reshape(1, n1, -1)
    X2 = X2.reshape(1, n2, -1)

    mat1 = np.repeat(X1, n2, axis=0)
    mat2 = np.repeat(X2.reshape(n2, 1, nf), n1, axis=1)

    isbow = np.repeat(np.repeat((feat < 2317).reshape(1, 1, nf), n1, axis=1), n2, axis=0)
    count_mat = nf - np.sum((mat1 == 0) & (mat2 == 0) & isbow, axis=2)
    zeros = np.where(count_mat == 0)
    count_mat[zeros] = 1

    dist_mat = np.sum(np.abs(mat1 - mat2), axis=2) / count_mat
    dist_mat[zeros] = 1

    return dist_mat


class HeuristicGenerator(object):
    """
    A class to go through the synthesizer-verifier loop
    """

    def __init__(self, train_primitive_matrix, val_primitive_matrix,
                 val_ground, train_ground=None, b=0.5):
        """
        Initialize HeuristicGenerator object
        b: class prior of most likely class (TODO: use somewhere)
        beta: threshold to decide whether to abstain or label for heuristics
        gamma: threshold to decide whether to call a point vague or not
        """

        self.train_primitive_matrix = train_primitive_matrix
        self.val_primitive_matrix = val_primitive_matrix
        self.val_ground = val_ground
        self.train_ground = train_ground
        self.b = b

        self.vf = None
        self.syn = None
        self.hf = []
        self.feat_combos = []
        self.feedback_idx = None
    @profile
    def apply_heuristics(self, heuristics, primitive_matrix, feat_combos, beta_opt):
        """
        Apply given heuristics to given feature matrix X and abstain by beta
        heuristics: list of pre-trained logistic regression models
        feat_combos: primitive indices to apply heuristics to
        beta: best beta value for associated heuristics
        """

        def marginals_to_labels(hf, X, beta, feat):

            if f'{hf.__class__}' == "<class 'program_synthesis.synthesizer.dummyKneighborClassifier'>":
                if self.feedback_idx is not None:
                    X_ = all_pair_dist(self.val_primitive_matrix[:, feat][self.feedback_idx], X, feat)
                else:
                    X_ = all_pair_dist(self.val_primitive_matrix[:, feat], X, feat)
                marginals = hf.predict_proba(X_, self.val_ground)[:, 1]
            else:
                marginals = hf.predict_proba(X)[:, 1]

            labels_cutoff = np.zeros(np.shape(marginals))
            labels_cutoff[marginals <= (self.b - beta)] = -1.
            labels_cutoff[marginals >= (self.b + beta)] = 1.
            return labels_cutoff

        L = np.zeros((np.shape(primitive_matrix)[0], len(heuristics)))
        for i, hf in enumerate(heuristics):
            L[:, i] = marginals_to_labels(hf, primitive_matrix[:, feat_combos[i]], beta_opt[i], feat_combos[i])

        # with cf.ThreadPoolExecutor(max_workers=8) as exe:
        #     future_to_index = {exe.submit(marginals_to_labels, i[1], primitive_matrix[:, feat_combos[i[0]]], beta_opt[i[0]], feat_combos[i[0]]): i[0]
        #                        for i in enumerate(heuristics)}
        #     for future in cf.as_completed(future_to_index):
        #         i = future_to_index[future]
        #         L[:, i] = future.result()
        return L

    @profile
    def prune_heuristics(self, heuristics, feat_combos, keep=1):
        """
        Selects the best heuristic based on Jaccard Distance and Reliability Metric
        keep: number of heuristics to keep from all generated heuristics
        """

        def calculate_jaccard_distance(num_labeled_total, num_labeled_L):
            scores = np.zeros(np.shape(num_labeled_L)[1])
            for i in range(np.shape(num_labeled_L)[1]):
                scores[i] = np.sum(np.minimum(num_labeled_L[:, i], num_labeled_total)) / np.sum(
                    np.maximum(num_labeled_L[:, i], num_labeled_total))
            return 1 - scores

        L_val = np.array([])
        L_train = np.array([])
        beta_opt = np.array([])
        max_cardinality = len(heuristics)
        for i in range(max_cardinality):
            # Note that the LFs are being applied to the entire val set though they were developed on a subset...
            beta_opt_temp = self.syn.find_optimal_beta(heuristics[i], self.val_primitive_matrix, feat_combos[i],
                                                       self.val_ground)
            L_temp_val = self.apply_heuristics(heuristics[i], self.val_primitive_matrix, feat_combos[i], beta_opt_temp)
            L_temp_train = self.apply_heuristics(heuristics[i], self.train_primitive_matrix, feat_combos[i],
                                                 beta_opt_temp)

            beta_opt = np.append(beta_opt, beta_opt_temp)
            if i == 0:
                L_val = np.append(L_val, L_temp_val)  # converts to 1D array automatically
                L_val = np.reshape(L_val, np.shape(L_temp_val))
                L_train = np.append(L_train, L_temp_train)  # converts to 1D array automatically
                L_train = np.reshape(L_train, np.shape(L_temp_train))
            else:
                L_val = np.concatenate((L_val, L_temp_val), axis=1)
                L_train = np.concatenate((L_train, L_temp_train), axis=1)

        # Use F1 trade-off for reliability
        acc_cov_scores = [f1_score(self.val_ground, L_val[:, i]) for i in range(np.shape(L_val)[1])]
        acc_cov_scores = np.nan_to_num(acc_cov_scores)

        if self.vf != None:
            # Calculate Jaccard score for diversity
            train_num_labeled = np.sum(np.abs(self.vf.L_train.T), axis=0)
            jaccard_scores = calculate_jaccard_distance(train_num_labeled, np.abs(L_train))
        else:
            jaccard_scores = np.ones(np.shape(acc_cov_scores))

        # Weighting the two scores to find best heuristic
        combined_scores = 0.5 * acc_cov_scores + 0.5 * jaccard_scores
        sort_idx = np.argsort(combined_scores)[::-1][0:keep]
        return sort_idx

    @profile
    def run_synthesizer(self, min_cardinality=1, max_cardinality=1, idx=None, keep=1, model='lr'):
        """
        Generates Synthesizer object and saves all generated heuristics
        max_cardinality: max number of features candidate programs take as input
        idx: indices of validation set to fit programs over
        keep: number of heuristics to pass to verifier
        model: train logistic regression ('lr') or decision tree ('dt')
        """
        if idx == None:
            primitive_matrix = self.val_primitive_matrix
            ground = self.val_ground
        else:
            primitive_matrix = self.val_primitive_matrix[idx, :]
            ground = self.val_ground[idx]
        self.idx = idx
        # Generate all possible heuristics
        self.syn = Synthesizer(primitive_matrix, ground, b=self.b)

        # Un-flatten indices
        def index(a, inp):
            i = 0
            remainder = 0
            while inp >= 0:
                remainder = inp
                inp -= len(a[i])
                i += 1
            try:
                return a[i - 1][remainder]  # TODO: CHECK THIS REMAINDER THING WTF IS HAPPENING
            except:
                import pdb;
                pdb.set_trace()

        # Select keep best heuristics from generated heuristics
        hf, feat_combos = self.syn.generate_heuristics(model, min_cardinality, max_cardinality)
        sort_idx = self.prune_heuristics(hf, feat_combos, keep)
        for i in sort_idx:
            self.hf.append(index(hf, i))
            self.feat_combos.append(index(feat_combos, i))

        # create appended L matrices for validation and train set
        beta_opt = self.syn.find_optimal_beta(self.hf, self.val_primitive_matrix, self.feat_combos, self.val_ground)
        self.L_val = self.apply_heuristics(self.hf, self.val_primitive_matrix, self.feat_combos, beta_opt)
        self.L_train = self.apply_heuristics(self.hf, self.train_primitive_matrix, self.feat_combos, beta_opt)

    @profile
    def run_verifier(self):
        """
        Generates Verifier object and saves marginals
        """
        ###THIS IS WHERE THE SNORKEL FLAG IS SET!!!!
        self.vf = Verifier(self.L_train, self.L_val, self.val_ground, has_snorkel=False)
        self.vf.train_gen_model()
        self.vf.assign_marginals()

    def gamma_optimizer(self, marginals):
        """
        Returns the best gamma parameter for abstain threshold given marginals
        marginals: confidences for data from a single heuristic
        """
        m = len(self.hf)
        gamma = 0.5 - (1 / (m ** (3 / 2.)))
        return gamma

    def find_feedback(self):
        """
        Finds vague points according to gamma parameter
        self.gamma: confidence past 0.5 that relates to a vague or incorrect point
        """
        # TODO: flag for re-classifying incorrect points
        # incorrect_idx = self.vf.find_incorrect_points(b=self.b)

        self.gamma = self.gamma_optimizer(self.vf.val_marginals)
        # gamma_opt = self.gamma
        vague_idx = self.vf.find_vague_points(b=self.b, gamma=self.gamma)
        incorrect_idx = vague_idx
        self.feedback_idx = list(set(list(np.concatenate((vague_idx, incorrect_idx)))))

    def evaluate(self):
        """
        Calculate the accuracy and coverage for train and validation sets
        """
        self.val_marginals = self.vf.val_marginals
        self.train_marginals = self.vf.train_marginals

        def calculate_accuracy(marginals, b, ground):
            total = np.shape(np.where(marginals != 0.5))[1]
            labels = np.sign(2 * (marginals - 0.5))
            return np.sum(labels == ground) / float(total)

        def calculate_coverage(marginals, b, ground):
            total = np.shape(np.where(marginals != 0.5))[1]
            labels = np.sign(2 * (marginals - 0.5))
            return total / float(len(labels))

        self.val_accuracy = calculate_accuracy(self.val_marginals, self.b, self.val_ground)
        self.train_accuracy = calculate_accuracy(self.train_marginals, self.b, self.train_ground)
        self.val_coverage = calculate_coverage(self.val_marginals, self.b, self.val_ground)
        self.train_coverage = calculate_coverage(self.train_marginals, self.b, self.train_ground)
        return self.val_accuracy, self.train_accuracy, self.val_coverage, self.train_coverage

    def heuristic_stats(self):
        '''For each heuristic, we want the following:
        - idx of the features it relies on
        - if dt, then the thresholds?
        '''

        def calculate_accuracy(marginals, b, ground):
            total = np.shape(np.where(marginals != 0.5))[1]
            labels = np.sign(2 * (marginals - 0.5))
            return np.sum(labels == ground) / float(total)

        def calculate_coverage(marginals, b, ground):
            total = np.shape(np.where(marginals != 0))[1]
            labels = marginals
            return total / float(len(labels))

        stats_table = np.zeros((len(self.hf), 6))
        for i in range(len(self.hf)):
            stats_table[i, 0] = int(self.feat_combos[i][0])
            try:
                stats_table[i, 1] = int(self.feat_combos[i][1])
            except:
                stats_table[i, 1] = -1.
            stats_table[i, 2] = calculate_accuracy(self.L_val[:, i], self.b, self.val_ground)
            stats_table[i, 3] = calculate_accuracy(self.L_train[:, i], self.b, self.train_ground)
            stats_table[i, 4] = calculate_coverage(self.L_val[:, i], self.b, self.val_ground)
            stats_table[i, 5] = calculate_coverage(self.L_train[:, i], self.b, self.train_ground)

        # Make table
        column_headers = ['Feat 1', 'Feat 2', 'Val Acc', 'Train Acc', 'Val Cov', 'Train Cov']
        pandas_stats_table = pd.DataFrame(stats_table, columns=column_headers)
        return pandas_stats_table
