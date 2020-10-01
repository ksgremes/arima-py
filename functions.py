# Reference Forecasting: Principles and Practice (HYNDMAN, ATHANASOPOULOS)
# Available at:  https://otexts.com/fpp2/
# Kae da Silva Gremes

import numpy as np


def lag(serie, h):
    # applies lag to the series
    n = len(serie)
    ret = serie.copy()
    for i in range(1, h+1):
        for j in range(n-1):
            ret[j] = ret[j+1] - ret[j]
        n = n - 1
    return(ret[0:n])


def slag(serie, H, m):
    # seasonal lag
    if(m == 1 or H == 0):
        return(serie)
    if len(serie) % m != 0:
        mat = np.zeros([(len(serie) // m) + 1 - H, m], dtype=np.float)
    else:
        mat = np.zeros([(len(serie) // m) - H, m], dtype=np.float)
    for i in range(m):
        ind = range(i, len(serie), m)
        if len(ind) == mat.shape[0] + 1:
            mat[:, i] = lag(serie[ind], H)
        else:
            mat[:-1, i] = lag(serie[ind], H)
    ret = np.zeros(mat.shape[0]*mat.shape[1])
    for i in range(mat.shape[0]):
        ret[(i*m):((i + 1)*m)] = mat[i, :]
    if len(serie) % m != 0:
        return(ret[:-(m - len(serie) % m)])
    else:
        return(ret)


def boxcox(serie, lamda):
    # boxcox transf.
    ret = serie.copy()
    if lamda != 0:
        for i in range(len(serie)):
            ret[i] = (serie[i]**lamda - 1)/lamda
        return(ret)
    else:
        return(np.log(ret))


# !! AIC, BIC ...
# autocovariance
def autocov(serie):
    # assumindo estacionariedade
    x = serie - np.mean(serie)
    ret = np.zeros(len(x))
    for i in range(len(serie)):
        for j in range(len(serie) - i):
            ret[i] += np.sum(x[j] * x[j+i])/len(x)
    return(ret)


def autocor(serie):
    ret = autocov(serie)
    for i in range(1, len(serie)):
        ret[i] = ret[i] / ret[0]
    ret[0] = 1
    return(ret)

# !! autocorrelacao parcial


if __name__ == '__main__':
    import csv
    dat = []
    fit = []
    res = []
    with open('airpass.txt', 'r') as arq:
        x = csv.reader(arq)
        for lin in x:
            dat.append(lin[0])
            fit.append(lin[1])
            res.append(lin[2])

    # fit = np.array(fit[1:len(dat)], dtype=np.float)
    # res = np.array(res[1:len(dat)], dtype=np.float)
    # dat = np.array(dat[1:len(dat)], dtype=np.float)
    # np.set_printoptions(suppress=True)
    # print(len(slag(dat[:100], 1, 12)))
    # print(lag(dat, 1))
    # print(len(lag(dat, 1)))
    # tes = slag(dat, 1, 12)
    # print(len(tes))
    # print(tes)
    sim = []
    with open('simul.txt', 'r') as arq:
        x = csv.reader(arq)
        for lin in x:
            sim.append(lin[0])
    sim = np.array(sim[1:len(sim)], dtype=np.float)
    print(sim)
    print(slag(sim, 0, 6))
