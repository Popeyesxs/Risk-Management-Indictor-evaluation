import pandas as pd
import numpy as np
import math

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

def spectral_risk_measure(gamma,N):
    spect_vec = np.zeros(N)
    for i in range(N):
        spect_vec[i] =(np.exp(gamma*(i+1)/N) - np.exp(gamma*i/N))/(np.exp(gamma) - 1)
    return spect_vec

def time_aggregated_simulation(mu,covariance,N,K,lamda,theta,mkt_cap_wgt,portfolio_value):
    Kday_loss = np.zeros(N)
    for i in range(N):
        mu_delta = mu
        covariance_Delta = covariance
        log_ret = np.zeros([K,4])
        for j in range(K):
            log_ret[j] = np.random.multivariate_normal(mu_delta, covariance_Delta)
            covariance_Delta = theta * covariance_Delta + (1 - theta) * np.multiply(np.matrix(log_ret[j] - mu_delta).T, np.matrix(log_ret[j] - mu_delta))
            mu_delta = mu_delta * lamda + (1 - lamda) * log_ret[j]
        Kday_loss[i] = -portfolio_value*(np.prod(np.dot((np.exp(log_ret)),mkt_cap_wgt)) - 1)
    return Kday_loss



if __name__ == "__main__":
    alpha = .99
    gamma = 30
    M = 100
    lamda =.94
    theta=.97
    K=10
    N=50000
    df = pd.read_csv('/Users/Popeye/Desktop/MF 731/Homework 2/Prices.csv')
    Stock_Data = df.iloc[:, 1:]
    Stock_Data.index = df.iloc[:, 0]
    S1 = np.log(Stock_Data / Stock_Data.shift(1))
    log_return = S1.iloc[1:]
    mkt_cap = [97.39, 158.2, 179.01, 417.97]
    mkt_cap_wgt =np.divide(mkt_cap,sum(mkt_cap))
    portfolio_value = 1000000
    mu_init = log_return.iloc[:M].mean()
    Sigma_init = log_return.iloc[:M].cov()
    mu_sample, sigma_sample = EWMA(mu_init, Sigma_init, lamda, theta, log_return)
    x = np.random.multivariate_normal(mu_sample, sigma_sample,N)

    #VaR by simulation
    loss_sim = -portfolio_value*(np.dot(np.exp(x) - 1 , mkt_cap_wgt))
    loss_sim.sort()
    VaR_sim = loss_sim[math.floor(alpha * N)]

    Kday_VaR_sim = VaR_sim * (K**0.5)

    Kday_ES_sim = (K ** 0.5) * (1/(N*(1-alpha)))*(np.sum(loss_sim[math.floor(N*alpha):N]) +(math.floor(N*alpha)-N*alpha)*loss_sim[math.floor(N*alpha)])

    spec_vec = spectral_risk_measure(gamma,N)

    Kday_spect_var = K ** (0.5) * np.dot(spec_vec,loss_sim.T)

    loss_aggregated = time_aggregated_simulation(mu_sample, sigma_sample, N, K,lamda,theta,mkt_cap_wgt,portfolio_value)

    loss_aggregated.sort()
    loss_aggregated_VaR = loss_aggregated[math.ceil(alpha * N) + 1]

    loss_aggregated_VaR_ES = (1/(N*(1-alpha)))*(np.sum(loss_aggregated[math.floor(N*alpha)+1:N]) +(math.floor(N*alpha)-N*alpha)*loss_aggregated[math.floor(N*alpha)])

    loss_aggregated_VaR_spec = np.dot(spec_vec,loss_aggregated)

    print('K day VaR: %.2f' %loss_aggregated_VaR)
    print('sqrt(K) times one day VaR: %.2f' %Kday_VaR_sim)
    print('K day ES:: %.2f' %loss_aggregated_VaR_ES)
    print('sqrt(K) times one day ES: %.2f' %Kday_ES_sim)
    print('K day Spectral: %.2f' %loss_aggregated_VaR_spec)
    print('sqrt(K) times one day Spectral: %.2f' %Kday_spect_var)



