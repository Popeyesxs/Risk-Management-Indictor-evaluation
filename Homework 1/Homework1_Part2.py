from scipy.stats import norm
import math
import numpy as np
import matplotlib.pyplot as plt

class BSOption:
    def __init__(self, s, x, t, T, sigma, rf):
        self.s = s
        self.x = x
        self.t = t
        self.sigma = sigma
        self.rf = rf
        self.T = T

    def d1(self):
        f = (self.rf + (self.sigma ** (2)) / 2) * (self.T - self.t)
        return (1 / (self.sigma * ((self.T - self.t) ** (0.5)))) * (np.log(self.s / self.x) + f)

    def d2(self):
        d1 = self.d1()
        return d1 - self.sigma * ((self.T - self.t) ** (0.5))

    def nd1(self):
        d1 = self.d1()
        return norm.cdf(d1)

    def nd2(self):
        d2 = self.d2()
        return norm.cdf(d2)

    def call_value(self):
        nd1 = self.nd1()
        nd2 = self.nd2()
        f1 = nd1 * self.s
        f2 = nd2 * self.x * math.e ** (-self.rf * (self.T - self.t))
        return f1 - f2

    def delta(self):
        return self.nd1()

    def gamma(self):
        return norm.pdf(self.d1()) /(self.sigma * self.s * (self.T - self.t)**0.5)

    def theta(self):
        f = (self.rf + (self.sigma ** (2)) / 2) * (self.T - self.t)
        return -(norm.pdf(self.d1()) * self.s *self.sigma)/(2*((self.T-self.t)**0.5)) - self.x * self.rf * np.exp(-self.rf * (self.T- self.t)) * norm.cdf(self.d2())

def full_loss(S_0, t, t_delta, x, lamda,K,T,sigma,r,M):
    Call_delta = BSOption(S_0 * np.exp(x), K, t+t_delta, T, sigma, r).call_value()
    Call = BSOption(S_0, K, t, T, sigma, r).call_value()
    loss = -M*(lamda * S_0 * (np.exp(x) - 1) - (Call_delta - Call))
    return loss

def linearized_loss(M,theta,t_delta,x):
    return M*theta*t_delta

def quadratic_loss(M,theta,t_delta,gamma,S_0,x):
    return M*(theta*t_delta+0.5*gamma*(np.exp(x)-1)**2*S_0**2)

if __name__ == "__main__":
    mu = 0.15475
    sigma = 0.2214
    r = 0.0132
    t = 0
    T = 0.25
    t_delta = 10 / 252
    S_0 = 158.12
    K = 170
    M = 100
    N = 1000000
    delta = BSOption(S_0,K,t,T,sigma,r).delta()
    gamma = BSOption(S_0, K, t, T, sigma, r).gamma()
    theta = BSOption(S_0, K, t, T, sigma, r).theta()
    lamda =  BSOption(S_0,K,t,T,sigma,r).nd1()
    x = np.random.normal(mu * t_delta, sigma * np.sqrt(t_delta), N)
    full_loss = full_loss(S_0, t, t_delta, x, lamda, K, T, sigma, r, M)
    linearized_loss = linearized_loss(M,theta,t_delta,x)
    quadratic_loss = quadratic_loss(M, theta, t_delta, gamma, S_0, x)
    fig, ax = plt.subplots()
    ax.hist(full_loss, bins=100, density=True, range=(-100, 1300))
    ax.set(title='Full losses simulations')
    fig1, ax1 = plt.subplots()
    ax1.hist(linearized_loss, bins=100, density=True, range=(-100, 1300))
    ax1.set(title='Linearized losses simulations')
    fig2, ax2 = plt.subplots()
    ax2.hist(quadratic_loss, bins=100, density=True, range=(-100, 1300))
    ax2.set(title='Quadratic losses simulations')
