# Reference Forecasting: Principles and Practice (HYNDMAN, ATHANASOPOULOS)
# Available at:  https://otexts.com/fpp2/
# Kae da Silva Gremes

import numpy as np
import functions as fun
import scipy.optimize as optim
from scipy.stats import norm


class Modelo:
    def error(self, phi=[0], d=0, the=[0], PHI=[0], D=0, THE=[0], m=0):
        # calculates error given parameters
        if(m > 0):
            if d == 0:
                lagged = self.transf - np.mean(self.transf)
            else:
                lagged = self.tranfs.copy()
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
                lagged = self.transf.copy()
            else:
                lagged = self.transf - np.mean(self.transf)
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

    def squareSum(self, params, p=0, d=0, q=0, P=0, D=0, Q=0, m=0):
        # for using in the optimization function
        phi = params[:p]
        the = params[p:(p + q)]
        PHI = params[(p + q):(p + q + P)]
        THE = params[-Q:]
        err = self.error(phi, d, the, PHI, D, THE, m)
        return(np.dot(err, err))

    def estimate(self, p=0, d=0, q=0, P=0, D=0, Q=0, m=0):
        # arima estimating
        # In this function, we first search for starting values with
        # optimize.brute. Then, we use optimize.minimize to find the results
        if(p+q == 0):
            raise ValueError('A soma dos parametros (AR e MA) nao pode ser 0')
        elif(p + q + P + Q > len(self.transf)):
            raise ValueError('A soma dos parametros nao (AR, I e MA) nao pode \
                             ser maior que o numero de observacoes')
        bnds = np.array([(-1, 1)] * (p + q + P + Q))
        rets = optim.brute(self.squareSum, bnds,
                           args=(p, d, q, P, D, Q, m), Ns=7)
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

    def boxcox(self, lamda, original_data=False):
        if original_data:
            self.transf = fun.boxcox(self.serie, lamda)
        else:
            self.transf = fun.boxcox(self.transf, lamda)

    def diff(self, d=0, D=0, original_data=False):
        if original_data:
            self.transf = fun.slag(self.serie, D, self.m)
            self.transf = fun.lag(self.transf, d)
        else:
            self.transf = fun.slag(self.transf, D, self.m)
            self.transf = fun.lag(self.transf, d)

    def information(self, p=None, q=None, res=None):
        if p is None and q is None and res is None:
            p = self.p
            q = self.q
            res = self.res
        elif res is None:
            if p != 0 or q != 0:
                params = self.estimate(p=p, q=q, d=self.d)
                res = self.error(phi=params[0], d=self.d, the=params[1])
            else:
                res = self.transf

        veros = norm.pdf(res, loc=0, scale=np.std(res))
        logveros = sum(np.log(veros))
        k = 0 if self.d > 0 else 1
        params = p + q + k + 1
        n = len(self.transf)

        AIC = -2*logveros + 2*(params)
        AICc = AIC + ((2*(params)*(params + 1)) / (n - params - 1))
        BIC = AIC + (np.log(n) - 2)*(params)
        return({'AIC': AIC, 'AICc': AICc, 'BIC': BIC})

    def modSelect(self, criteria='AICc'):
        Atual = self.information(p=0, q=0, res=self.transf)[criteria]
        maisBaixo = False
        p = 0
        q = 0
        while maisBaixo is False:
            maisP = self.information(p=p+1, q=q)[criteria]
            maisQ = self.information(p=p, q=q+1)[criteria]
            menosP = self.information(p=p-1, q=q)[criteria] if p != 0 else 1e10
            menosQ = self.information(p=p, q=q-1)[criteria] if q != 0 else 1e10
            menor = min([Atual, maisP, maisQ, menosP, menosQ])
            if menor == maisP:
                Atual = maisP
                p += 1
            elif menor == maisQ:
                Atual = maisQ
                q += 1
            elif menor == menosP:
                Atual = menosP
                p -= 1
            elif menor == menosQ:
                Atual = menosQ
                q -= 1
            elif menor == Atual:
                maisBaixo = True
        return([p, q, Atual])

    def __init__(self, serie, p=0, d=0, q=0, P=0, D=0, Q=0, m=0,
                 transformation=None):
        self.serie = serie
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.m = m

        if transformation is None:
            self.transf = self.serie.copy()
        elif transformation == 'log':
            self.transf = np.log(self.serie)
        elif 'boxcox' in transformation:
            self.transf = fun.boxcox(self.serie, transformation['boxcox'])

        if D > 0:
            self.transf = fun.slag(self.transf, D, self.m)
        if d > 0:
            self.transf = fun.lag(self.transf, d)

        if self.p == 0 and self.q == 0:
            self.p, self.q, _ = self.modSelect()

        params = self.estimate(p=self.p, d=self.d, q=self.q,
                               P=self.P, D=self.D, Q=self.Q,
                               m=self.m)
        self.phi = params[0]
        self.the = params[1]
        if(m > 0):
            self.PHI = params[2]
            self.THE = params[3]
        else:
            self.PHI = [0]
            self.THE = [0]
        self.res = self.error(phi=self.phi, d=self.d, the=self.the,
                              PHI=self.PHI, D=self.D, THE=self.THE,
                              m=self.m)


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
    # mod = Modelo(dat, p=2, d=1, q=1)
    # print(mod.res)
    # print(mod.phi, mod.the)
    # print(mod.p, mod.q, mod.d, mod.P, mod.Q, mod.D, mod.m)
    modlog = Modelo(dat, d=1, transformation='log')
    print(modlog.phi, modlog.the)
    print(modlog.res)
