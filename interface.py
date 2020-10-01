# Reference Forecasting: Principles and Practice (HYNDMAN, ATHANASOPOULOS)
# Available at:  https://otexts.com/fpp2/
# Kae da Silva Gremes

# import modelo as mod
import graph_utils as gu
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Interface(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        self.master.title('Arima-Py')
        self.fram = tk.Frame(self.master)
        self.grid(padx=5, pady=5, ipadx=5, ipady=5)
        self.fram.place(rely=.5, relx=.5, anchor="center", relwidth=1)
        self.fram.grid_columnconfigure(1, weight=1)
        self.fram.grid_rowconfigure(1, weight=1)
        self.preenche()

    def preenche(self):
        self.buts = tk.Frame(self.fram)  # Botoes de acao (direita)
        self.info = tk.Frame(self.fram)  # Informações
        self.dado = tk.Frame(self.fram)  # Dados

        # Botoes
        tk.Button(self.buts, text='Importar',
                  command=self.importar).grid(row=0, column=0, sticky='new',
                                              padx=5, pady=5)
        tk.Button(self.buts, text='Estimar',
                  command=self.estimar).grid(row=1, column=0, sticky='new',
                                             padx=5, pady=5)

        bExport = tk.LabelFrame(self.buts, text='Exportar')
        bExport.grid_columnconfigure(0, weight=1)
        bExport.grid(row=2, column=0, sticky='new', padx=5, pady=5)
        tk.Button(bExport, text='Dados',
                  command=lambda: self.exportar('dados')).grid(
                      row=0, column=0, padx=5, pady=5, sticky='EW')
        tk.Button(bExport, text='Saída de texto',
                  command=lambda: self.exportar('texto')).grid(
                      row=1, column=0, padx=5, pady=5, sticky='EW')
        tk.Button(bExport, text='Gráficos',
                  command=lambda: self.exportar('graf')).grid(
                      row=2, column=0, padx=5, pady=5, sticky='EW')

        bGraf = tk.LabelFrame(self.buts, text='Gráficos')
        bGraf.grid_columnconfigure(0, weight=1)
        bGraf.grid(row=3, column=0, sticky='new', padx=5, pady=5)
        tk.Button(bGraf, text='Série',
                  command=lambda: self.menugrafs('splot')).grid(
                      row=0, column=0, padx=5, pady=5, sticky='EW')
        tk.Button(bGraf, text='Sazonal',
                  command=lambda: self.menugrafs('seasonplot')).grid(
                      row=1, column=0, padx=5, pady=5, sticky='EW')
        tk.Button(bGraf, text='Polar',
                  command=lambda: self.menugrafs('polar')).grid(
                      row=2, column=0, padx=5, pady=5, sticky='EW')
        tk.Button(bGraf, text='Subséries',
                  command=lambda: self.menugrafs('subseries')).grid(
                      row=3, column=0, padx=5, pady=5, sticky='EW')
        tk.Button(bGraf, text='Lags',
                  command=lambda: self.menugrafs('lag')).grid(
                      row=4, column=0, padx=5, pady=5, sticky='EW')
        tk.Button(bGraf, text='Correlações',
                  command=lambda: self.menugrafs('corr')).grid(
                      row=5, column=0, padx=5, pady=5, sticky='EW')

        bTransf = tk.LabelFrame(self.buts, text='Transformações')
        bTransf.grid_columnconfigure(0, weight=1)
        bTransf.grid(row=4, column=0, sticky='new', padx=5, pady=5)
        tk.Button(bTransf, text='Diferenciação',
                  command=lambda: self.menutrans('dif')).grid(
                      row=0, column=0, padx=5, pady=5, sticky='EW')
        tk.Button(bTransf, text='Log',
                  command=lambda: self.menutrans('log')).grid(
                      row=1, column=0, padx=5, pady=5, sticky='EW')
        tk.Button(bTransf, text='Box-Cox',
                  command=lambda: self.menutrans('box')).grid(
                      row=1, column=0, padx=5, pady=5, sticky='EW')

        tk.Button(self.buts, text='Fechar',
                  command=self.fechar).grid(row=999, column=0, sticky='new',
                                            padx=5, pady=5)

        # Informações
        self.pars = tk.LabelFrame(self.info, borderwidth=1,
                                  text='Parâmetros')

        tk.Label(self.pars, text='P').grid(
            row=0, column=0, padx=5, pady=5, sticky='E')
        self.parP = tk.Entry(self.pars, width=2)
        self.parP.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self.pars, text='D').grid(
            row=1, column=0, padx=5, pady=5, sticky='E')
        self.parD = tk.Entry(self.pars, width=2)
        self.parD.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self.pars, text='Q').grid(
            row=2, column=0, padx=5, pady=5, sticky='E')
        self.parQ = tk.Entry(self.pars, width=2)
        self.parQ.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(self.pars, text="Phi").grid(
            row=0, column=2, padx=5, sticky='E')
        self.phi = tk.Label(self.pars, text=0)
        self.phi.grid(row=0, column=3, padx=5, pady=5)

        tk.Label(self.pars, text="D").grid(
            row=1, column=2, padx=5, sticky='E')
        self.diff = tk.Label(self.pars, text=0)
        self.diff.grid(row=1, column=3, padx=5, pady=5)

        tk.Label(self.pars, text="Theta").grid(
            row=2, column=2, padx=5, sticky='E')
        self.theta = tk.Label(self.pars, text=0)
        self.theta.grid(row=2, column=3, padx=5, pady=5)

        self.pars.grid(row=0, column=0, sticky='news',
                       ipady=5, ipadx=5, padx=5, pady=5)

        self.transframe = tk.LabelFrame(
            self.info, text="Transformações utilizadas")
        self.transframe.grid(row=1, column=0, padx=5, pady=5)
        self.transf = tk.Label(self.transframe, text='Nenhuma')
        self.transf.grid(row=0, column=0, columnspan=6, padx=5, pady=5)

        # Dados
        self.dados = tk.Text(self.dado, wrap='word')
        self.montaTexto()
        self.dados.grid(row=0, column=0, ipady=5, ipadx=5, sticky='news')

        # Grafico
        plt = gu.splot(np.array(range(100)))
        self.img = FigureCanvasTkAgg(plt, master=self.fram)

        self.img.get_tk_widget().grid(row=0, column=0, sticky='nse', rowspan=2,
                                      ipady=5, ipadx=5, padx=15, pady=15)
        # Monta a janela
        self.info.grid(row=0, column=1, sticky='ns',
                       ipady=5, ipadx=5, padx=5, pady=5)
        self.dado.grid_columnconfigure(0, weight=1)
        self.dado.grid(row=1, column=1, sticky='news',
                       ipady=5, ipadx=5, padx=5, pady=5)
        self.buts.grid(row=0, column=2, sticky='nsw', rowspan=2,
                       ipady=5, ipadx=5, padx=5, pady=5)

    def montaTexto(self, dados=None, transf=None,
                   estim=None, res=None, pred=None):
        if dados is None and estim is None and transf is None and \
                res is None and pred is None:
            string = ("Bem vindo de volta! \n\n" +
                      "Esse é o Arima-Py, o sistema que estima seus modelos " +
                      "ARIMA (todo programado em Python!) \n\n" +
                      "Para começar, por que não importa uma série clicando " +
                      "no botão 'Importar'?")
        else:
            string = ("t \t Valor \t Transf. \t Estim. \t Resid. \n")
            string += "--------------------------------------------- \n"
            for i in range(min(len(dados), 10)):
                string += "{} \t {:.3f} \n".format(i, dados[i])
            if len(dados) > 10:
                string += "\n Com mais {} linhas. \n".format(len(dados) - 10)
        self.dados.insert(tk.END, "\n" + string + "\n")
        self.dados.see('end')

    def exportar(self, opt=10):
        pass

    def importar(self):
        self.montaTexto(dados=np.array(range(10)), transf=np.array(range(9)))

    def estimar(self):
        pass

    def menugrafs(self):
        pass

    def menutrans(self):
        pass

    def fechar(self):
        self.master.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    root.wm_attributes('-zoomed', True)
    app = Interface(root)
    root.mainloop()
