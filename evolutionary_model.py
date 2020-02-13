from numba import njit, prange
import numpy as np
import random


# Random initiation of population states
# Number of miners n; number of pools m
def init(n, m):
    x = np.zeros(m)
    for i in range (0, n):
        x[random.randint(1, m) - 1] += 1
    return x


# Expected payoff
# Mining pool index i; population state x; args in order: hash rate in PentaHashes, number of miners, block size,
# fixed reward, fee per transaction, network delay, electricity cost per PentaHash.
@njit
def y(i, x, args):
    return (args[3] + args[4] * args[2]) / args[1] * (args[0][i])/(np.dot(args[0], x)) * np.exp(-args[5]*args[2]/args[6]) - args[7]*args[0][i]


#   Evolutionary model, asymptotically follows the characteristic system of ODEs
# Population state x; max time t_max; args in order: hash rate in PentaHashes, number of miners, block size,
# fixed reward, fee per transaction, network delay, electricity cost per PentaHash.
@njit
def evol(x, t_max, args):
    x_copy = x.copy()
    x_t = np.zeros((t_max, len(x_copy)), np.float64)
    x_t[0] = x_copy

    for t in range(0, t_max):
        for h in range(0, len(x_copy)):
            for i in range(0, np.int32(x_copy[h] * args[1])):
                j = random.randint(1, len(x_copy)) - 1
                p_ij = x_copy[j] * max([y(j, x_copy, args) - y(h, x_copy, args), 0])
                movimiento = np.random.binomial(1, p_ij) / args[1]
                x_copy[h] -= movimiento
                x_copy[j] += movimiento
        x_t[t] = x_copy

    return x_t
