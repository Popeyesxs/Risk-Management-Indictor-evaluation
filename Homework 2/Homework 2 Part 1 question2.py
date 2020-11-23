from scipy.stats import norm
import math
import numpy as np
import pandas as pd

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

def one_day_VaR(h_t,x,S_0,call_price_t,call_price_tdelta,N):
    """
    :param h_t: delta
    :param x: the strike price
    :param t
    :param t_delta
    :param S_0
    :return: the one day VaR over [t,t + t_delta]
    """
    loss_list = -h_t * S_0*(np.exp(x) - 1) - call_price_t + call_price_tdelta
    loss_list.sort()
    return 100 * loss_list[math.ceil(N*0.95)]

def VaR_Time_Aggregation(mu,S_0, Kappa, t, t_delta, T, sigma, r,k,simulations = 25000):
    S = np.zeros([simulations,k+1])
    V_T_k = np.zeros([simulations,k+1])
    h_T = np.zeros([simulations,k+1])
    Y_T = np.zeros([simulations,k+1])
    dayloss = np.zeros([simulations,k])
    S[:, 0] = S_0
    h_T[:,0] = BSOption(S_0, Kappa, t, T, sigma, r).delta()
    V_T_k[:,0] = np.multiply(h_T[:,0],S[:, 0]) - BSOption(S_0, Kappa, t, T, sigma, r).call_value()
    for i in range(simulations):
        for j in range(k):
            t_time = (j+1) * t_delta
            S[i][j+1] = S[i][j]*np.exp(np.random.normal((mu) * t_delta, sigma * np.sqrt(t_delta)))
            V_T_k[i][j+1] = h_T[i][j] * S[i][j+1] + (Y_T[i][j])*np.exp(r * t_delta) - BSOption(S[i][j+1], Kappa, t_time, T, sigma, r).call_value()
            dayloss[i][j] =100 * (V_T_k[i][j] - V_T_k[i][j+1])
            h_T[i][j+1] =  BSOption(S[i][j+1], Kappa, t_time, T, sigma, r).delta()
            Y_T[i][j+1] = V_T_k[i][j+1] - (h_T[i][j+1] * S[i][j+1] - BSOption(S[i][j+1], Kappa, t_time, T, sigma, r).call_value())
    return dayloss



if __name__ == "__main__":
    mu = 0.15475
    sigma = 0.2214
    r = 0.0132
    t = 0
    T = 0.25
    t_delta = 1 / 252
    S_0 = 158.12
    Kappa = 170
    calls_M = 100
    N = 25000
    simulations = 25000
    k = 10
    x = np.random.normal(mu * t_delta, sigma * np.sqrt(t_delta), N)
    h_t = BSOption(S_0, Kappa, t, T, sigma, r).delta()
    call_price_t = BSOption(S_0, Kappa, t, T, sigma, r).call_value()
    call_price_tdelta = BSOption(S_0*np.exp(x), Kappa, t+t_delta, T, sigma, r).call_value()
    VaR_one_day = one_day_VaR(h_t, x, S_0, call_price_t, call_price_tdelta, N)
    VaR_one_day_K = (10**0.5) * VaR_one_day
    day_loss_aggregation =  VaR_Time_Aggregation(mu,S_0, Kappa, t, t_delta, T, sigma, r,k,simulations = 25000)
    aggregation_VaR_sample = day_loss_aggregation.sum(axis = 1)
    aggregation_VaR_sample.sort()
    aggregation_VaR = aggregation_VaR_sample[math.floor(0.95 * 25000)]
    VaR1 = [VaR_one_day,VaR_one_day_K,aggregation_VaR]
    df = pd.DataFrame(VaR1).T
    df.columns = ['VaR_one_day','VaR_Kday','aggregation_VaR']
    print(df)