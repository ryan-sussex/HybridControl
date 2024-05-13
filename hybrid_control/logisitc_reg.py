from scipy import special


def mode_posterior(x, u, W_x, W_u, b):
    return special.softmax(W_u @ u + W_x @ x + b)