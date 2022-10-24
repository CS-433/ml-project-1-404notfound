import numpy as np
import matplotlib.pyplot as plt


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_dataset(y, x, k_indices, k):
    """Return the split dataset for cross_validation.

    Args:
        y:          shape=(N, 1)
        x:          shape=(N, D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold

    Returns:
        x_train:    training dataset x
        x_test:     testing dataset x
        y_train:    training dataset y
        y_test:     testing dataset y
    """

    # ****************************************************
    # split the dataset into training set and testing set for cross_validation
    # ***************************************************

    test_indice = k_indices[k]
    train_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    train_indice = train_indice.reshape(-1)
    x_test = x[test_indice]
    y_test = y[test_indice]
    x_train = x[train_indice]
    y_train = y[train_indice]

    return x_train, x_test, y_train, y_test


def cross_validation_visualization(hyper_x, loss_train, loss_test, jet_num):
    """visualization the curves of the training loss and testing loss in cross_validation
    Args:
        hyper_x:    list/np.arrray, hyperparameter
        loss_train: list/np.arrray, training loss
        loss_test:  list/np.array, testing loss
        jet_num:    scalar, pri_jet_num
    """

    plt.semilogx(hyper_x, loss_train, marker=".", color="b", label="training loss")
    plt.semilogx(hyper_x, loss_test, marker=".", color="r", label="testing loss")
    plt.xlabel("lambda")
    plt.ylabel("loss")
    plt.title("cross validation for PRI_JET_NUM={}".format(jet_num))
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation" + str(jet_num))
    plt.clf()
