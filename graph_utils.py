# Reference: Forecasting: Principles and Practice (HYNDMAN, ATHANASOPOULOS)
# Available at:  https://otexts.com/fpp2/
# Kae da Silva Gremes

# !! -> coisas pra fazer
from matplotlib.figure import Figure
import numpy as np
import functions as fun
import matplotlib
matplotlib.use('TkAgg')


def splot(serie):
    # series plot
    img = Figure(figsize=(6, 6))
    fig = img.add_subplot(111)
    fig.plot(range(1, len(serie)+1), serie, 'b-')
    # fig.xlabel('Tempo')
    # fig.ylabel('Valor observado')
    # p.savefig
    return(img)


def seasonplot(serie, m):
    # !! cores
    # seasonal plot
    img = Figure(figsize=(6, 6))
    fig = img.add_subplot(111)
    for i in range(len(serie)//m):  # number of seasons
        fig.plot(range(m), serie[(i*m):(i+1)*m], '--')
    #  last season (if incomplete)
    r = len(serie) % m
    if r != 0:
        fig.plot(range(r), serie[-r:], 'r--')
    fig.ylabel('Valor observado')
    if m == 12:
        xlab = 'Mes'
    elif m == 4:
        xlab = 'Trimestre'
    elif m == 52:
        xlab = 'Semana'
    elif m == 7:
        xlab = 'Dia'
    else:
        xlab = 'Indice'
    fig.xlabel(xlab)
    # season ticks on y-axis
    leg = range(0, len(serie), m)
    box = fig.get_position()
    fig.set_position([box.x0, box.y0, box.width * .8, box.height])
    fig.legend(range(2000, 2000+len(leg)), bbox_to_anchor=(1, 0.5),
               loc='center left')
    return(img)


def polarplot(serie, m):
    #  !! cores
    img = Figure(fisize=(6, 6))
    fig = img.add_subplot(111, projection='polar')
    angles = np.linspace(0, 2*np.pi, m, endpoint=False)  # setting angles
    angles = np.append(angles, 0)
    for i in range(len(serie)//m):
        fig.plot(angles, np.append(serie[(i*m):(i+1)*m], serie[i*m]), '-')
    r = len(serie) % m
    if r != 0:
        fig.plot(angles[0:r], serie[-r:], '-')
    leg = range(0, len(serie), m)
    box = fig.get_position()
    fig.set_position([box.x0, box.y0, box.width * .8, box.height])
    fig.legend(range(2000, 2000+len(leg)), bbox_to_anchor=(1, .5),
               loc='center left')
    return(img)


def subseriesplot(serie, m):
    # subseries seasonal plot
    img = Figure(figsize=(6, 6))
    fig = img.add_subplot(111)
    if len(serie) % m != 0:
        mat = np.zeros([m, (len(serie) // m) + 1], dtype=np.float)
    else:
        mat = np.zeros([m, (len(serie) // m)], dtype=np.float)
    for i in range(m):
        try:
            mat[i, :] = serie[range(i, len(serie), m)]
        except ValueError:
            mat[i, :-1] = serie[range(i, len(serie), m)]
            mat[i, -1] = np.nan
    # creating indexes
    x = np.zeros(mat.shape)
    for i in range(m):
        if len(serie) % m != 0:
            x[i, :] = np.linspace(i-.5, i+.5, (len(serie) // m) + 1,
                                  endpoint=False)
        else:
            x[i, :] = np.linspace(i-.5, i+.5, (len(serie) // m),
                                  endpoint=False)
    x = x + (.5 - x[0, -1])/2  # to align with the ticks
    #  drawing mean lines
    meas = np.zeros(m)
    for i in range(m):
        meas[i] = np.nanmean(mat[i, :])
    for i in range(m):
        fig.plot(x[i, :], mat[i, :], 'b-')
        fig.plot(x[i, :], np.repeat(meas[i], len(x[i, :])), 'k--')
    fig.xticks(range(0, m), range(1, m+1))  # !! customize label
    return(img)


def lagplot(serie, qt=2, m=0):
    # qt: quantity of plots per axe (2x2 = 4 plots)
    img = Figure(figsize=(6, 6))
    for i in range(qt**2):
        fig = img.add_subplot(qt, qt, i + 1)
        fig.plot(fun.lag(serie, i + 1), serie[:-(i+1)])
    return(img)
    #  Continue: include seasonality


def acfplot(serie):
    # autocorrelation plot !!cores
    img = Figure(figsize=(6, 6))
    fig = img.subplot(111)
    acf = fun.autocor(serie)
    fig.bar(range(len(serie)//5), acf[:len(serie)//5], .1, color='b')
    fig.plot(range(len(serie)//5), np.repeat(1/(len(serie)**(1/2)),
                                             len(serie)//5), 'r--')
    fig.plot(range(len(serie)//5), np.repeat(-1/(len(serie)**(1/2)),
                                             len(serie)//5), 'r--')
    fig.plot(range(len(serie)//5), np.repeat(0, len(serie)//5), 'k-')
    fig.xlabel('lag')
    fig.ylabel('Autocorrelacao')
    return(img)


if __name__ == '__main__':
    import csv
    dat = []
    with open('airpass.txt', 'r') as arq:
        x = csv.reader(arq)
        for lin in x:
            dat.append(lin[0])
    dat = np.array(dat, dtype=np.float)
    # splot(dat)
    # splot(fun.boxcox(dat, .3))
    # splot(np.log(dat))
    # splot(fun.lag(dat, 1))
    # seasonplot(dat, 12)
    # subseriesplot(dat, 12)
    # polarplot(dat, 12)
    acfplot(dat)
    acfplot(fun.lag(dat, 1))
    # acfplot(fun.lag(np.log(dat), 1))
    # lagplot(dat, qt = 2, m = 12)
