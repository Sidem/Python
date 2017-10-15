from itertools import count, islice
from scipy.special import binom

def zeta(s, t=100):
    if s == 1:
        return float("inf")
    term = (1 / 2 ** (n + 1)
                * sum((-1) ** k * binom(n, k) * (k + 1) ** -s
                    for k in range(n + 1))
            for n in count(0))
    return sum(islice(term, t)) / (1 - 2 ** (1 - s))

print(zeta(-1))