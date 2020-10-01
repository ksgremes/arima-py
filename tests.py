# Reference Forecasting: Principles and Practice (HYNDMAN, ATHANASOPOULOS)
# Available at:  https://otexts.com/fpp2/
# Kae da Silva Gremes

from scipy.stats import chi2 as chisq
import functions as fun


def ljungbox(modelo, lagmax=0, conf=.95):
    # Ljung-box test
    n = len(modelo.transf)
    autocor = fun.autocor(modelo.res)
    if lagmax == 0:
        lagmax = n//5
    soma = 0
    for i in range(1, lagmax + 1):
        soma += (autocor[i]**2) / (n - i)
    Q = n * (n + 2) * soma
    critical = chisq.ppf(conf, lagmax)
    pval = 1 - chisq.cdf(Q, lagmax)
    print("Região crítica: [{}, +Inf]\n \
          Estatistica calculada: {}\n \
          Valor-p: {}".format(critical, Q, pval))
    return([Q, critical, pval])


if __name__ == '__main__':
    import csv
    import numpy as np
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
    import modelo
    mod = modelo.Modelo(dat, p=2, d=1, q=1, transformation='log')
    print(mod.phi, mod.the)
    stats = ljungbox(mod)
