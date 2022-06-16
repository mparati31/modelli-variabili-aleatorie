import matplotlib.pyplot as plt
import numpy as np

from src.plot_functions import *
from scipy.stats import (
    bernoulli,
    binom,
    hypergeom,
    expon,
    norm,
    poisson,
    uniform
)

def plot_bern(p):
    X = bernoulli(p)
    support = X.support()
    
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    
    plt.sca(axes[0][0])
    plot_pmf(support, X.pmf(support))
    plt.xlim(-1, 2)

    plt.sca(axes[0][1])
    x = np.linspace(-1, 2, 1000)
    plot_cdf_discr(x, X.cdf(x), -.1, 1.1, support)

    plt.sca(axes[1][0])
    x = np.linspace(0, 1, 100)
    plot_mean(x, x, p, X.mean(), '$p$')

    plt.sca(axes[1][1])
    Var = list(map(lambda p: bernoulli.var(p), x))
    plot_var(x, Var, p, X.var(), '$p$', .27)

    plt.show()


def plot_binom(n, p):
    X = binom(n, p)
    support = np.arange(0, n+1)
    
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    
    plt.sca(axes[0][0])
    plot_pmf(support, X.pmf(support))

    plt.sca(axes[0][1])
    x = np.linspace(-1, n+2)
    plot_cdf_discr(x, X.cdf(x), -.5, n+.5,  support)

    plt.sca(axes[1][0])
    x = np.linspace(0, 1, 100)
    E = list(map(lambda p: binom.mean(n, p), x))
    plot_mean(x, E, p, X.mean(), '$p$', 20)

    plt.sca(axes[1][1])
    Var = list(map(lambda p: binom.var(n, p), x))
    plot_var(x, Var, p, X.var(), '$p$', 5.2)

    plt.show()


def plot_unif_discr(n):
    support = np.arange(1, n+1)

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))

    plt.sca(axes[0])
    p = list(map(lambda _: 1/n, support))
    plot_pmf(support, p)

    plt.sca(axes[1])
    x = np.arange(-1, n+2)
    F = list(map(lambda i: i/n if i <= n else 0 if i <=0 else 1, x))
    plot_cdf_discr(x, F, -.5, n+.5, np.concatenate(([0], support)))
    plt.xticks(x, x+1)
    plt.xlim(-.5, n-1+.5)

    plt.show()


def plot_geom(p):
    support = np.arange(15)

    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    
    plt.sca(axes[0][0])
    pf = list(map(lambda i: p*(1-p)**i, support))
    plot_pmf(support, pf)

    plt.sca(axes[0][1])
    x = np.arange(-2, 30)
    F = list(map(lambda xp: 1-(1-p)**(xp+1), x))
    plot_cdf_discr(x+1, F, -.5, 14.1, support)
    
    plt.sca(axes[1][0])
    x = np.linspace(0, 1, 1000)[1:]
    E = list(map(lambda p: (1-p)/p, x))
    plot_mean(x, E, p, (1-p)/p, '$p$', 100)

    plt.sca(axes[1][1])
    Var = list(map(lambda p: (1-p)/p**2, x))
    plot_var(x, Var, p, (1-p)/p**2, '$p$', 500)

    plt.show()


def plot_poisson(lambd):
    X = poisson(lambd)
    support = np.arange(25)

    plot_pmf(support, X.pmf(support))

    plt.show()


def plot_hyper(N=10, M=10, n=15):
    X = hypergeom(N+M, N, n)
    support = np.arange(n+1)
    
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.7))
    
    plt.sca(axes[0])
    plot_pmf(support, X.pmf(support))

    plt.sca(axes[1])
    x = np.arange(-2, n+3)
    plot_cdf_discr(x+1, X.cdf(x), -.5, n+.5, support)

    plt.show()


def plot_unif_cont(ab, I):
    a, b = ab
    x1, x2 = I
    X = uniform(a, b-a)

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    plt.sca(axes[0])
    x = np.linspace(-1, 11, 1000)
    plot_pdf(x, X.pdf(x), -1, 11)
    plt.plot([X.mean(), X.mean()], [0, X.pdf(x[len(x)//2])], linestyle='--', color='m', label='$E[X]$')
    x = np.linspace(x1, x2, 1000)
    plt.fill_between(x, 0, X.pdf(x), color='r', alpha=.25, label='$P(x=I)$')
    plt.legend()

    plt.sca(axes[1])
    x = np.linspace(-1, 11, 1000)
    plot_cdf_cont(x, X.cdf(x), -1, 11)

    plt.show()


def plot_exp(lambd):
    X = expon(scale=1/lambd)
    support = np.linspace(0, 10, 1000)
    
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    
    plt.sca(axes[0][0])
    plot_pdf(support, X.pdf(support), 0, 10, 2.1)

    plt.sca(axes[0][1])
    plot_cdf_cont(support, X.cdf(support), 0, 10)

    plt.sca(axes[1][0])
    x = np.linspace(0, 2, 100)[1:]
    E = list(map(lambda lambd_p: expon(scale=1/lambd_p).mean(), x))
    plot_mean(x, E, lambd, X.mean(), '$p$', 50)

    plt.sca(axes[1][1])
    x = np.linspace(0, 2, 100)[1:]
    E = list(map(lambda lambd_p: expon(scale=1/lambd_p).var(), x))
    plot_var(x, E, lambd, X.var(), '$p$', 200)

    plt.show()


def plot_gauss(mu, sigma):
    X = norm(mu, sigma)
    support = np.linspace(-10, 10, 10000)
    
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    
    plt.sca(axes[0])
    plt.plot([mu-X.std(), mu-X.std()], [0, X.pdf(mu-X.std())], linestyle='--', color='g', label='$\mu-\sigma_X$')
    plt.plot([mu+X.std(), mu+X.std()], [0, X.pdf(mu+X.std())], linestyle='--', color='g', label='$\mu+\sigma_X$')
    plot_pdf(support, X.pdf(support), -10, 10, 2.1)
    plt.plot([mu, mu], [0, X.pdf(mu)], linestyle='--', color='m', label='$E[X]$')
    plt.legend()

    plt.sca(axes[1])
    plot_cdf_cont(support, X.cdf(support), -10, 10)

    plt.show()
