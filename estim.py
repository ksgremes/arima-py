# Reference Forecasting: Principles and Practice (HYNDMAN, ATHANASOPOULOS)
# Available at:  https://otexts.com/fpp2/
# Kae da Silva Gremes

import numpy as np
import functions as fun
import scipy.optimize as optim


def error(serie, phi=[0], d=0, the=[0], PHI=[0], D=0, THE=[0], m=0):
    # calculates error given parameters
    if(m > 0):
        if d > 0 or D > 0:
            lagged = fun.slag(serie, D, m)
            lagged = fun.lag(lagged, d)
        else:
            lagged = serie.copy()
            lagged = lagged - np.mean(lagged)  # setting the mean = 0
        errS = np.zeros(len(lagged))  # error equals 0 before x_q
        err = np.zeros(len(lagged))  # error equals 0 before x_q
        P = len(PHI)
        Q = len(THE)
        errS[:m] = lagged[:m]
        for i in range(m, len(lagged)):
            AR = range(i - m, max(i - P*m - 1, -1), -m)
            MA = range(i - m, max(i - Q*m - 1, -1), -m)
            errS[i] = (lagged[i] -
                       np.dot(PHI[:i], lagged[AR]) -
                       np.dot(THE[:i], errS[MA]))
        p = len(phi)
        q = len(the)
        for i in range(1, len(errS)):
            ar = range(i - 1, max(i - p - 1, -1), -1)
            ma = range(i - 1, max(i - q - 1, -1), -1)
            err[i] = (errS[i] -
                      np.dot(phi[:i], errS[ar]) -
                      np.dot(the[:i], err[ma]))
    else:
        if d > 0:
            lagged = fun.lag(serie, d)
        else:
            lagged = serie.copy()
            lagged = lagged - np.mean(lagged)
        err = np.zeros(len(lagged))
        p = len(phi)
        q = len(the)
        for i in range(1, len(lagged)):
            ar = range(i - 1, max(i - p - 1, -1), -1)
            ma = range(i - 1, max(i - q - 1, -1), -1)
            err[i] = (lagged[i] -
                      np.dot(phi[:i], lagged[ar]) -
                      np.dot(the[:i], err[ma]))
    return(err)


def squareSum(params, serie, p=0, d=0, q=0, P=0, D=0, Q=0, m=0):
    # for using in the optimization function
    phi = params[:p]
    the = params[p:(p + q)]
    PHI = params[(p + q):(p + q + P)]
    THE = params[-Q:]
    err = error(serie, phi, d, the, PHI, D, THE, m)
    return(np.dot(err, err))


def arima(serie, p=0, d=0, q=0, P=0, D=0, Q=0, m=0):
    # arima estimating
    # In this function, we first search for starting values with optimize.brute
    # Then, we use optimize.minimize to find the results
    if(p+q == 0):
        raise ValueError('A soma dos parametros (AR e MA) nao pode ser 0')
    elif(p + d + q + P + D + Q > len(serie)):
        raise ValueError('A soma dos parametros nao (AR, I e MA) nao pode ser \
                         maior que o numero de observacoes')
    bnds = np.array([(-1, 1)] * (p + q + P + Q))
    rets = optim.brute(squareSum, bnds,
                       args=(serie, p, d, q, P, D, Q, m), Ns=7)
    # rets = optim.minimize(squareSum, rets,
    #                       args=(serie, p, d, q, P, D, Q, m), tol=1e-1)
    if (p > 0):
        phi = rets[:p]
    else:
        phi = [0]
    if (q > 0):
        the = rets[p:(p + q)]
    else:
        the = [0]
    if (P > 0):
        PHI = rets[(p+q):(p+q+P)]
    else:
        PHI = [0]
    if (Q > 0):
        THE = rets[-Q:]
    else:
        THE = [0]
    return([phi, the, PHI, THE])


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

    fit = np.array(fit[1:len(dat)], dtype=np.float)
    res = np.array(res[1:len(dat)], dtype=np.float)
    dat = np.array(dat[1:len(dat)], dtype=np.float)
    np.set_printoptions(suppress=True)
    err = error(dat, [1.0906, -.489], 1, [-.8438])
    print(err)
    # coef = arima(dat, p=2, d=1, q=1)
    # print(coef)
    # err2 = error(dat, coef[0], 1, coef[1])
    # export = np.array([res[1:len(dat)], err, err2])
    # print(export.transpose())
    # print(np.dot(err, err), np.dot(err2, err2))
    # print('Parte sazonal')
    # err3 = error(dat, phi=[.97525], d=0, the=[-0.3255],
    #              PHI=[-.9573], THE=[.8896], D=1, m=12)
    # coef2 = arima(np.log(dat), p=1, q=1, P=1, D=1, Q=1, m=12)
    # print(coef2)
    # err4 = error(dat, phi=coef2[0], the=coef2[1],
    #              PHI=coef2[2], D=1, THE=coef2[3], m=12)
    # print(np.dot(err3, err3), np.dot(err4, err4), np.dot(res[12:], res[12:]))
    # print(len(res[12:]), len(err4))
    # exp = np.array([res[12:], err4, err3])
    # print(exp.transpose())
    # sim = []
    # with open('simul.txt', 'r') as arq:
    #     x = csv.reader(arq)
    #     for lin in x:
    #         sim.append(lin[0])
    # sim = np.array(sim[1:len(sim)], dtype=np.float)
    # coef3 = arima(sim, p=1, d=1, q=1, P=1, Q=1, m=6)
    # print(coef3)
# Series: foo
# ARIMA(1,1,1)(1,0,1)[6]
#
# Coefficients:
#          ar1      ma1    sar1     sma1
#       0.7483  -0.9547  0.3294  -0.2813
# s.e.  0.0949   0.0448  0.6519   0.6524
#
# sigma^2 estimated as 1.41:  log likelihood=-155.77
# AIC=321.53   AICc=322.18   BIC=334.51
