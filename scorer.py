# USAGE: `python3 scorer.py <path-to-input> <path-to-output>`

import numpy as np
import sys


def score(vals, cov, betas, alphas):
    if not (np.all(betas <= alphas)):
        print("WA", betas, alphas)
        return False

    n = cov.shape[0]

    sc = np.sum(np.multiply(betas, vals))

    res = np.ones((n, n), dtype=int)
    res1 = np.multiply(res, betas)  # row-copies of beta
    res2 = res1.T
    res = np.multiply(res1, res2)
    res = np.multiply(res, cov)

    var_sum = np.sum(np.multiply(res, np.eye(n, dtype=int)))

    total_sum = np.sum(res)
    cov_sum = (total_sum - var_sum) // 2

    sc -= cov_sum + var_sum

    return sc


cov = [[]]
n = 0


def read_cov_matrix(lines):
    global cov

    cov = np.zeros((n, n), dtype=int)

    for i, line in enumerate(lines):
        if i >= n:
            break
        things = line.split(" ")
        things = list(map(int, things))

        for j in range(n):
            cov[i][j] = things[j]


def main():
    if len(sys.argv) != 3:
        print("USAGE: `python3 scorer.py <path-to-input> <path-to-output>`")
        return

    inp_file = sys.argv[1]
    out_file = sys.argv[2]

    global n

    with open(inp_file) as f:
        lines = f.read().split("\n")
        n = int(lines[0])
        inp = [list(map(int, lines[i].split(" "))) for i in range(1, n + 1)]
        alphas, vals = zip(*inp)
        alphas = np.array(alphas)
        vals = np.array(vals)

        read_cov_matrix(lines[n + 1:])

    with open(out_file) as f:
        beta_vals = f.read().split(" ")
        beta_vals = beta_vals[0:n]
        beta = np.array(list(map(int, beta_vals)))

    s = score(vals, cov, beta, alphas)
    if s < 0:
        print("Your score ( = " + str(s) + ") is negative and will be rounded up to 0")
    s = max(0, s)
    print("Final score =", s)


if __name__ == "__main__":
    main()
