import pandas as pd
import datetime as dt
import numpy as np
from scipy.stats import norm
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
    for i in range(N):
        covariance = theta * covariance + (1 - theta) * np.multiply(np.matrix( log_return.iloc[i,:] - mu ).T,np.matrix(log_return.iloc[i,:] - mu))
        mu = mu * lamda + (1- lamda) *  log_return.iloc[i,:]
    return np.array(mu),np.array(covariance)

def stress_testing(mu,sigma,K,Simulation_M,dollar_position,Square_root_Linear_loss,Regulatory_capital):

    shock_Kday_port_loss = np.zeros(Simulation_M)
    shock_sum_stock_return = np.zeros([Simulation_M, 2])
    shock_Kday_VaR_exceed = np.zeros(Simulation_M)
    shock_Kday_VaR_reg_exceed = np.zeros(Simulation_M)
    Kday_port_loss = np.zeros(Simulation_M)
    sum_stock_return = np.zeros([Simulation_M, 2])
    Kday_VaR_exceed = np.zeros(Simulation_M)
    Kday_VaR_reg_exceed = np.zeros(Simulation_M)

    for i in range(Simulation_M):
        shock_sim_mu_hat = mu
        shock_sim_Sigma_hat = sigma
        sim_mu_hat = mu
        sim_Sigma_hat = sigma
        shock_sim_log_ret = np.zeros([K,2])
        sim_log_ret =  np.zeros([K,2])
        shock_sim_log_ret[0][1] = shock_sim_mu_hat[1] - 5 * np.sqrt(shock_sim_Sigma_hat[1][1])
        shock_sim_correlation = shock_sim_Sigma_hat[0][1] / (shock_sim_Sigma_hat[1][1] * shock_sim_Sigma_hat[0][0])**0.5
        shock_sim_log_ret[0][0] = shock_sim_mu_hat[0] + shock_sim_correlation * np.sqrt(shock_sim_Sigma_hat[0][0]/shock_sim_Sigma_hat[1][1])*\
                                  (shock_sim_log_ret[0][1] - shock_sim_mu_hat[1]) + np.random.normal(0,1) * np.sqrt(shock_sim_Sigma_hat[0][0] *(1- shock_sim_correlation**2))
        sim_log_ret[0] = np.random.multivariate_normal(sim_mu_hat, sim_Sigma_hat)

        for j in np.arange(1,K):
            shock_sim_Sigma_hat = theta * shock_sim_Sigma_hat + (1 - theta) * np.matrix(shock_sim_log_ret[j-1] - shock_sim_mu_hat).T*\
                                                                        np.matrix(shock_sim_log_ret[j-1] - shock_sim_mu_hat)
            shock_sim_mu_hat = shock_sim_mu_hat * lamda + (1 - lamda) * shock_sim_log_ret[j-1]

            sim_Sigma_hat = theta * sim_Sigma_hat + (1 - theta) * \
                np.matrix(sim_log_ret[j - 1] - sim_mu_hat).T* \
                np.matrix(sim_log_ret[j - 1] - sim_mu_hat)
            sim_mu_hat = sim_mu_hat * lamda + (1 - lamda) * sim_log_ret[j - 1]
            shock_sim_log_ret[j] = np.random.multivariate_normal(shock_sim_mu_hat, shock_sim_Sigma_hat)
            sim_log_ret[j] = np.random.multivariate_normal(sim_mu_hat, sim_Sigma_hat)
        shock_sum_stock_return[i] = np.sum(shock_sim_log_ret,axis =0)
        shock_Kday_port_loss[i] =-1 *  np.dot(dollar_position,shock_sum_stock_return[i])
        sum_stock_return[i] = np.sum(sim_log_ret,axis =0)
        Kday_port_loss[i] =-1 * np.dot( dollar_position,sum_stock_return[i])
        shock_Kday_VaR_exceed[i] = (shock_Kday_port_loss[i]>Square_root_Linear_loss)
        shock_Kday_VaR_reg_exceed[i] = (shock_Kday_port_loss[i]>Regulatory_capital)
        Kday_VaR_exceed[i] = Kday_port_loss[i] >Square_root_Linear_loss
        Kday_VaR_reg_exceed[i] = Kday_port_loss[i] > Regulatory_capital

    return shock_sum_stock_return,shock_Kday_port_loss,sum_stock_return,Kday_port_loss,shock_Kday_VaR_exceed,shock_Kday_VaR_reg_exceed,Kday_VaR_exceed,Kday_VaR_reg_exceed




if __name__ == '__main__':
    df = pd.read_csv('/Users/popeye/Desktop/MF 731/Homework 3 /MSFT_AAPL_Log_Returns.csv',header=None)
    time = pd.DataFrame(np.zeros_like(df.iloc[:, 0]))
    for i in range(len(df)):
        b2 =dt.datetime.fromordinal(int(df.iloc[i, 0]+ 693594))
        b2 = str(b2)[:10]
        time.iloc[i, 0] = dt.datetime.strptime(b2,'%Y-%m-%d').date()
    log_return = pd.DataFrame(df.iloc[:,1:])
    log_return.index = time.iloc[:,0]
    log_return.columns = ['MSFT','AAPL']
    mkt_cap = np.array([448.77, 575.11])
    mkt_cap_wgt = np.divide(mkt_cap, sum(mkt_cap))
    por = 1000000
    alpha = 0.95
    K = 10
    M = 100
    lamda = 0.97
    theta = 0.97
    Simulation_M = 50000
    dollar_position = por * mkt_cap_wgt
    init_mu = 0
    init_sigma = np.zeros([2,2])
    #1 Estimate the mean vector and covariance
    mu_sample, covariance_sample = EWMA(init_mu, init_sigma, lamda, theta, log_return)
    #2 Estimate VaR
    Linear_loss = np.dot(-dollar_position,mu_sample) + np.sqrt(np.dot(np.dot(dollar_position,covariance_sample), dollar_position)) * norm.ppf(0.95)

    Square_root_Linear_loss = K**0.5 * Linear_loss

    Regulatory_capital = 3 * Square_root_Linear_loss

    shock_sum_stock_return, shock_Kday_port_loss, sum_stock_return, Kday_port_loss, shock_Kday_VaR_exceed, shock_Kday_VaR_reg_exceed, Kday_VaR_exceed, Kday_VaR_reg_exceed = \
        stress_testing(mu_sample, covariance_sample, K, Simulation_M, dollar_position, Square_root_Linear_loss, Regulatory_capital)

    shock_Kday_port_loss.sort()
    Kday_port_loss.sort()
    shock_avg_port_loss = np.mean(shock_Kday_port_loss)
    shock_port_VaR_Kday = shock_Kday_port_loss[math.ceil(Simulation_M * alpha)]
    shock_num_Kday_VaR_exceed = np.sum(shock_Kday_VaR_exceed)
    shock_num_Kday_VaR_reg_exceed = np.sum(shock_Kday_VaR_reg_exceed)
    avg_port_loss = np.mean(Kday_port_loss)
    port_VaR_Kday = Kday_port_loss[math.ceil(Simulation_M * alpha)]
    num_Kday_VaR_exceed = np.sum(Kday_VaR_exceed)
    num_Kday_VaR_reg_exceed = np.sum(Kday_VaR_reg_exceed)

    print('Confidence: %0.2f' %alpha)
    print('Number of days: %0.0f' %K)
    print('Initial one day VaR: %10.2f'% Linear_loss)
    print('Initial %i day VaR: %10.2f'% (K, Square_root_Linear_loss))
    print('3x Initial %i day VaR: %10.2f'% (K, Regulatory_capital))
    print('%0.0f day VaR (no shock, shock): %0.2f %0.2f\n'
    %(K, port_VaR_Kday, shock_port_VaR_Kday))
    print('Average %i day loss: (no shock, shock) %10.2f %10.2f\n'
    %(K, avg_port_loss, shock_avg_port_loss))
    print('Pct exceedances for %0.0f day VaR over inital %i day VaR (no shock, shock):  %10.2f %10.2f\n'
    %(K, K, 100 *num_Kday_VaR_exceed / Simulation_M, 100 * shock_num_Kday_VaR_exceed / Simulation_M))
    print('Pct exceedances for %i day VaR over 3x inital %i day VaR (no shock, shock): %10.2f %10.2f\n'
    %(K, K, 100 * num_Kday_VaR_reg_exceed / Simulation_M, 100 * shock_num_Kday_VaR_reg_exceed / Simulation_M))
    print('\n')






