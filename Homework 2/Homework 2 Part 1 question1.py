import pandas as pd
import numpy as np
import math
from scipy.stats import norm
#1 VaR for a Portfolio of Apple and Amozon
#(a) Estimate the mean vector and covariance matrix for the daily log returns using EWMA.

def EWMA(mu,covariance,lamda,theta,log_return):
    """
    :param mu_delta: the initial mean
    :param covariance_Delta: the initial variance
    :param lamda
    :param theta
    :param log_return: log return rate
    :return: the estimation of mean and variance
    """
    N = log_return.shape[0]
    mu_delta = np.zeros_like(mu)
    covariance_Delta = np.zeros_like(covariance)
    for i in range(N):
        covariance_Delta = theta * covariance_Delta + (1 - theta) * np.multiply(np.matrix( log_return.iloc[i,:] - mu_delta ).T,np.matrix(log_return.iloc[i,:] - mu_delta))
        mu_delta = mu_delta * lamda + (1- lamda) *  log_return.iloc[i,:]
    return np.array(mu_delta),np.array(covariance_Delta)

if __name__ == "__main__":
    df = pd.read_csv('/Users/Popeye/Desktop/MF 731/Homework 2/StockData.csv')
    Stock_Data = df.iloc[:,1:]
    Stock_Data.index = df['Date']
    S = Stock_Data.iloc[::-1]
    S1 = np.log(S / S.shift(1))
    log_return = S1.iloc[1:]
    N = log_return.shape[0]
    mu = log_return.mean()
    covariance = log_return.cov()
    lamda = 0.97
    theta = 0.97
    market_capitalization = [2225.98,1725.48]
    weight = np.divide(market_capitalization,sum(market_capitalization))
    mu_sample, covariance_sample = EWMA(mu, covariance, lamda, theta, log_return)

    #b Estimate the VaR, for a 95% confidence, of the market cap weighted portfolio in three different way.
    #use the empirical distribution
    portfolio = 1000000
    emp_full_losses = (-portfolio * np.dot(np.exp( log_return)-1,weight))
    emp_full_losses.sort()
    VaR_emp_full = emp_full_losses[math.ceil(501 * 0.95) - 1]

    emp_lin_losses = -portfolio * np.dot(log_return,weight)
    emp_lin_losses.sort()
    emp_lin_losses = emp_lin_losses[math.ceil(501 * 0.95) - 1]

    emp_quad_losses = (-portfolio * np.dot(log_return + 0.5 * log_return**2, weight))
    emp_quad_losses.sort()
    VaR_emp_quad= emp_quad_losses[math.floor(501 * 0.95)]

    Simulation_N = 100000
    x = np.random.multivariate_normal(mu_sample, covariance_sample, Simulation_N,)
    #use the full method
    full_sim_ewma_losses = (-portfolio * np.dot((np.exp(x) - 1),weight))
    full_sim_ewma_losses.sort()
    VaR_full = full_sim_ewma_losses[math.ceil(Simulation_N * 0.95) - 1]

    #use the linear method

    variance = np.dot(weight, np.dot(covariance_sample , (weight)).T)
    VaR_lin = -portfolio * np.dot(weight, mu_sample) + portfolio * (variance)**(0.5) * norm.ppf(0.95)

    #use the quadratic method
    quad_sim_ewma_losses = (-portfolio * np.dot((x + 0.5 * x**2),weight))
    quad_sim_ewma_losses.sort()
    VaR_quad = quad_sim_ewma_losses[math.ceil(Simulation_N * 0.95) - 1]

    VaR = np.array([VaR_emp_full,emp_lin_losses,VaR_emp_quad,VaR_full,VaR_lin,VaR_quad])
    df1 = pd.DataFrame(VaR).T
    df1.columns = ['VaR_emp_full','VaR_emp_lin','VaR_emp_quad','Simulation_VaR_full','Simulation_VaR_lin','Simulation_VaR_quad']
    print(df1.T)







