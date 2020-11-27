import pickle
import numpy as np
from program_synthesis.heuristic_generator import HeuristicGenerator
from sklearn.model_selection import train_test_split
import warnings
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

def run_snuba(hg, ht,  n = 1, idx = None, min_cardinality = 1, max_cardinality = 1, w = 0.5, writer = None, j=None):
    validation_accuracy = []
    training_accuracy = []
    validation_coverage = []
    training_coverage = []

    training_marginals = []

    es = ['happy', 'sad', 'angry', 'surprise', 'disgus', 'fear']
    for i in range(1, n+1):
        if i  % 1 == 0:
            print(f"Running iteration: {i}")

        # Repeat synthesize-prune-verify at each iterations
        if i-1:
            hg.run_synthesizer(min_cardinality=min_cardinality, max_cardinality=max_cardinality, idx=idx, keep=1, model=ht)
        else:
            hg.run_synthesizer(min_cardinality=min_cardinality, max_cardinality=max_cardinality, idx=idx, keep=5, model=ht)
        hg.run_verifier()


        training_marginals.append(hg.vf.train_marginals)

        # Find low confidence datapoints in the labeled set
        hg.find_feedback()
        idx = hg.feedback_idx


        pos = np.where(training_marginals[-1] >= 0.5 + hg.gamma)[0]
        neg = np.where(training_marginals[-1] <= 0.5 - hg.gamma)[0]
        if len(pos) > 0:
            precision=np.sum(hg.train_ground[pos] == 1) / len(pos)
        else:
            precision = 0
        recall=np.sum(hg.train_ground[pos] == 1) / np.sum(hg.train_ground == 1)
        coverage=(len(pos) + len(neg)) / len(hg.train_ground)

        if writer is not None:
            writer.add_scalar(f'{es[j]}/Precision', precision, i)
            writer.add_scalar(f'{es[j]}/Recall', recall, i)
            writer.add_scalar(f'{es[j]}/Coverage', coverage, i)





        if idx == []:
            break
        if np.unique(hg.val_ground[idx]).shape[0] < 2:
            break
    return(training_marginals[-1])


def load_data():
    with open('data/mosei_X_stem_bag_r0.1_seed1234.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('data/mosei_y.pkl', 'rb') as f:
        y = pickle.load(f)

    with open('data/mosei_X_wordvector.pkl', 'rb') as f:
        X2 = pickle.load(f)

    glove_max = 0
    for r in X2:
        glove_max = max(glove_max, np.max(r))
    glove_min = 0
    for r in X2:
        glove_min = min(glove_min, np.min(r))
    glove_normal_factor = glove_max - glove_min

    new_X = np.zeros((len(X), X.shape[1] + 900))
    for i in range(len(X)):
        new_X[i][:X.shape[1]] = X[i]
        new_X[i][X.shape[1]:X.shape[1]+300] = np.mean(X2[i], axis=0)/glove_normal_factor
        new_X[i][X.shape[1]+300:X.shape[1] + 600] = np.max(X2[i], axis=0)/glove_normal_factor
        new_X[i][X.shape[1]+600:X.shape[1] + 900] = np.min(X2[i], axis=0)/glove_normal_factor
    return np.array(new_X), np.array(y)

def main():
    now = datetime.now()
    current_date_time = now.strftime("%m-%d-%Y_%H:%M:%S")
    warnings.filterwarnings("ignore")
    X, y = load_data()

    train_X, val_X, train_y_,  val_y_ = train_test_split(X, y, test_size=0.1, random_state=1234)

    train_ys = [np.zeros(len(train_y_)),np.zeros(len(train_y_)),np.zeros(len(train_y_)),np.zeros(len(train_y_)),np.zeros(len(train_y_)),np.zeros(len(train_y_))]
    val_ys = [np.zeros(len(val_y_)),np.zeros(len(val_y_)),np.zeros(len(val_y_)),np.zeros(len(val_y_)),np.zeros(len(val_y_)),np.zeros(len(val_y_))]
    label_count = np.zeros(6)

    for i, l in enumerate(train_y_):
        for j in range(1, 7):
            if l[j] > 0:
                train_ys[j-1][i] = 1
            else:
                train_ys[j-1][i] = -1

    for i, l in enumerate(val_y_):
        for j in range(1, 7):
            if l[j] > 0:
                val_ys[j-1][i] = 1
                label_count[j - 1] += 1
            else:
                val_ys[j-1][i] = -1



    precisions = []
    recalls = []
    coverages = []
    training_marginalss = []
    timess = []
    hgs = dict()
    for ht in ['nn','dt', 'lr']: #


        for c in range(1, 11):
            exp_name = f'{current_date_time}_{ht}_c{c}'
            writer = SummaryWriter(f'tensorboard_logs/{exp_name}')

            hgs[ht] = []
            for j in range(6):
                hgs[ht].append(HeuristicGenerator(train_X, val_X, val_ys[j], train_ys[j]))



            precision = []
            recall = []
            coverage = []
            times = []

            training_marginals = []

            for j in range(6):
                start = time.time()
                training_marginals.append(run_snuba(hgs[ht][j], ht, n=20, min_cardinality = c, max_cardinality = c, writer=writer, j=j))
                end = time.time()
                print(f'time elapsed : {end-start}')
                times.append(end-start)

                pos = np.where(training_marginals[j] >= 0.5+hgs[ht][j].gamma)[0]
                neg = np.where(training_marginals[j] <= 0.5-hgs[ht][j].gamma)[0]
                if len(pos) > 0:
                    precision.append(np.sum(train_ys[j][pos] == 1)/len(pos))
                else:
                    precision.append(0)
                recall.append(np.sum(train_ys[j][pos] == 1)/np.sum(train_ys[j] == 1))
                coverage.append((len(pos) + len(neg))/len(train_ys[j]))

            precisions.append(precision)
            recalls.append(recall)
            coverages.append(coverage)
            training_marginalss.append(training_marginals)
            timess.append(times)

            if writer is not None:
                for j, e in enumerate(['happy', 'sad', 'angry', 'surprise', 'disgust', 'fear']):
                    writer.add_scalar(f'time/{e}', times[j], j)

                writer.close()




    print()
if __name__ == '__main__':
    main()