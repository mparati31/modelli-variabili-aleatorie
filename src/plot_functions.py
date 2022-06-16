import matplotlib.pyplot as plt

def plot_pmf(x, y, y_top=1.1):
    plt.vlines(x, 0, y, lw=2)
    plt.plot(x, y, 'o', markersize=8, color='r', mec='k')
    plt.ylim(0, y_top)
    plt.xticks(x)
    plt.xlabel('$x$')
    plt.ylabel('$p_X$')
    plt.title('Funzione di massa di probabilità')


def plot_pdf(x, y, x_l, x_r, y_top=1.1):
    plt.step(x, y, lw=2)
    plt.xlim(x_l, x_r)
    plt.xlabel('$x$')
    plt.ylim(0, y_top)
    plt.ylabel('$f_X$')
    plt.title('Funzione di densità di probabilità')


def plot_cdf_discr(x, y, xl, xr, xticks):
    plt.step(x, y, lw=2)
    plt.xlim(xl, xr)
    plt.xticks(xticks)
    plt.xlabel('$x$')
    plt.ylim(0, 1.1)
    plt.ylabel('$F_X$')
    plt.title('Funzione di ripartizione')


def plot_mean(x, y, xp, yp, xlabel, y_top=1.1):
    plt.plot(x, y, lw=2)
    plt.plot(xp, yp, 'o', markersize=8, color='r', mec='k', label='$p={}$'.format(xp))
    plt.xlabel(xlabel)
    plt.ylim(0, y_top)
    plt.ylabel('$E[X]$')
    plt.title('Valore atteso')
    plt.legend()


def plot_var(x, y, xp, yp, xlabel, y_top=1.1):
    plt.plot(x, y, lw=2)
    plt.plot(xp, yp, 'o', markersize=8, color='r', mec='k', label='$p={}$'.format(xp))
    plt.xlabel(xlabel)
    plt.ylim(0, y_top)
    plt.ylabel(r'$\operatorname{Var}(X)$')
    plt.title('Varianza')
    plt.legend()


def plot_cdf_cont(x, y, x_l, x_r, y_top=1.1):
    plt.step(x, y, lw=2)
    plt.xlim(x_l, x_r)
    plt.xlabel('$x$')
    plt.ylim(0, y_top)
    plt.ylabel('$F_X$')
    plt.title('Funzione di ripartizione')
