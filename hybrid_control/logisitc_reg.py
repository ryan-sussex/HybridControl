from scipy import special


def mode_posterior(x, W, b):
    return special.softmax(W @ x + b)