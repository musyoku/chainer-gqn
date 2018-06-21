import math


def gaussian_kl_divergence(mu_q, mu_p):
    diff = mu_q - mu_p
    return 0.5 * diff**2


def gaussian_negative_log_likelihood(x, mu):
    diff = x - mu
    return 0.5 * (math.log(2 * math.pi) + diff**2)
