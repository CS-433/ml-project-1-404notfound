import numpy as np


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    t = np.clip(t, -10, 10)     # avoid overflow
    return 1 / (1 + np.exp(-t))


def linear_reg_gradient(y, tx, w):
    """Computes the gradient of linear regression at w.

    Args:
        y: numpy array of shape=(N, 1)
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, 1). The vector of model parameters.

    Returns:
        An numpy array of shape (D, 1) (same shape as w), containing the gradient of the loss at w.
    """

    N = y.shape[0]
    e = y - tx @ w

    # compute gradient vector for MSE
    grad = -1 / N * tx.T @ e

    return grad.reshape(-1, 1)


def logistic_reg_gradient(y, tx, w):
    """Computes the gradient of logistic regression at w.

    Args:
        y: numpy array of shape=(N, 1)
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, 1). The vector of model parameters.

    Returns:
        An numpy array of shape (D, 1) (same shape as w), containing the gradient of the loss at w.
    """

    N = y.shape[0]
    e = y - sigmoid(tx @ w)

    # compute gradient vector for MSE
    grad = -1 / N * tx.T @ e

    return grad.reshape(-1, 1)


def compute_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, 1)
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, 1). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    N = y.shape[0]
    e = y - tx @ w

    loss = 1 / (2 * N) * e.T @ e

    return np.float64(loss)


def compute_ce(y, tx, w):
    """Compute the cross-entropy loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """

    y_pred = sigmoid(tx @ w)
    loss = -(y.T @ np.log(y_pred) + (1 - y).T @ np.log(1 - y_pred)) / y.shape[0]

    return np.float64(loss)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: numpy array of shape=(N, 1)
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D, 1). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the last weight vector of shape (D, 1)
        loss: the corresponding mse loss
    """

    # Define parameters to store w and loss
    w = initial_w
    loss = compute_mse(y, tx, w)
    ws = [initial_w]
    losses = [loss]

    for n_iter in range(max_iters):

        # compute gradient
        grad = linear_reg_gradient(y, tx, w)

        # update w by gradient
        w = w - gamma * grad

        # compute loss
        loss = compute_mse(y, tx, w)

        # store w and loss
        ws.append(w)
        losses.append(loss)

    return ws[-1], losses[-1]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """Linear regression using stochastic gradient descent.

    Args:
        y: numpy array of shape=(N, 1)
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D, 1). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        batch_size: default 1, a scalar denoting the batch size

    Returns:
        w: the last weight vector of shape (D, 1)
        loss: the corresponding mse loss
    """

    # Define parameters to store w and loss
    w = initial_w
    loss = compute_mse(y, tx, w)
    ws = [initial_w]
    losses = [loss]

    for n_iter in range(max_iters):
        # implement stochastic gradient descent.
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):

            # compute gradient
            grad = linear_reg_gradient(y_batch, tx_batch, w)

            # update w by gradient
            w = w - gamma * grad

            # compute loss
            loss = compute_mse(y, tx, w)

            # store w and loss
            ws.append(w)
            losses.append(loss)

    return ws[-1], losses[-1]


def least_squares(y, tx):
    """Least squares regression using normal equations.
    Args:
        y: numpy array of shape (N, 1), N is the number of samples.
        tx: numpy array of shape (N, D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D, 1), D is the number of features.
        loss: scalar.
    """

    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b).reshape(-1, 1)
    loss = compute_mse(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.
    Args:
        y: numpy array of shape (N, 1), N is the number of samples.
        tx: numpy array of shape (N, D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D, 1), D is the number of features.
        loss: scalar
    """
    N, D = tx.shape
    I = np.eye(D)
    w = np.linalg.solve(tx.T @ tx + 2 * N * lambda_ * I, tx.T @ y).reshape(-1, 1)
    loss = compute_mse(y, tx, w)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD (y ∈ {0, 1}).

    Args:
        y: numpy array of shape=(N, 1)
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D, 1). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the last weight vector of shape (D, 1)
        loss: the corresponding mse loss
    """

    # Define parameters to store w and loss
    w = initial_w
    loss = compute_ce(y, tx, w)
    ws = [initial_w]
    losses = [loss]

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=y.shape[0], num_batches=1
        ):
            # compute gradient
            grad = logistic_reg_gradient(y_batch, tx_batch, w)

            # update w by gradient
            w = w - gamma * grad

            # compute loss
            loss = compute_ce(y, tx, w)

            # store w and loss
            ws.append(w)
            losses.append(loss)

    return ws[-1], losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent
    or SGD (y ∈ {0, 1}, with regularization term λ|w|2)

    Args:
        y: numpy array of shape=(N, 1)
        tx: numpy array of shape=(N, D)
        lambda_: a scalar denoting the regularization term
        initial_w: numpy array of shape=(D, 1). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the last weight vector of shape (D, 1)
        loss: the corresponding mse loss
    """

    # Define parameters to store w and loss
    w = initial_w
    loss = compute_ce(y, tx, w)
    ws = [initial_w]
    losses = [loss]

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=y.shape[0], num_batches=1
        ):
            # compute gradient
            grad = logistic_reg_gradient(y_batch, tx_batch, w)

            # update w by gradient
            w = w - gamma * (grad + 2 * lambda_ * w)

            # compute loss
            loss = compute_ce(y, tx, w)

            # store w and loss
            ws.append(w)
            losses.append(loss)

    return ws[-1], losses[-1]
