import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import datetime as dt


#1 Components risk measures for an equity portfolio.
def Componenet_risk_measure(mu,covariance,lamda,theta,log_return,M,mkt_cap,alpha):
    """
    The component risk measure for an equity portfolio
    :param mu: the mean of portfolio
    :param covariance: the covariance of portfolio
    :param lamda: EWMA parameter
    :param theta: EWMA parameter
    :param log_return: The generation of historical data
    :param M: Trail days
    :param mkt_cap: The investment weight
    :param alpha: The confidence
    :return: Percent component
    """
    N = log_return.shape[0]
    mu_delta = mu
    covariance_Delta = covariance
    Var_lin = np.zeros(N - M+ 1)
    ES_lin =np.zeros(N - M+ 1)
    mar_Var_lin =np.zeros([N - M+ 1,5])
    mar_ES_lin =np.zeros([N - M+ 1,5])
    percent_Var_com =np.zeros([N - M+ 1,5])
    percent_ES_com =np.zeros([N - M+ 1,5])
    percent_contribution = np.zeros([N - M+ 1,5])
    for i in np.arange(M,N+1):
        covariance_Delta = theta * covariance_Delta + (1 - theta) * np.matrix( log_return.iloc[i-1,:] - mu_delta ).T*np.matrix(log_return.iloc[i-1,:] - mu_delta)
        mu_delta = mu_delta * lamda + (1- lamda) *  log_return.iloc[i-1,:]
        Var_lin[i-M] = -np.dot(mkt_cap.T, mu_delta) + np.sqrt(np.dot(np.dot(mkt_cap, covariance_Delta).T,mkt_cap)) * norm.ppf(alpha)
        ES_lin[i-M] =  -np.dot(mkt_cap.T, mu_delta) + np.sqrt(np.dot(np.dot(mkt_cap.T, covariance_Delta),mkt_cap)) * (1/(1-alpha)) * norm.pdf(norm.ppf(alpha))
        mar_Var_lin[i-M] = -mu_delta + np.divide(np.dot(mkt_cap.T, covariance_Delta),np.sqrt(np.dot(np.dot(mkt_cap.T, covariance_Delta),mkt_cap)))* norm.ppf(alpha)
        mar_ES_lin[i-M] = -mu_delta + np.divide(np.dot(mkt_cap.T, covariance_Delta),np.sqrt(np.dot(np.dot(mkt_cap.T, covariance_Delta),mkt_cap)))* (1/(1-alpha)) * norm.pdf(norm.ppf(alpha))
        percent_Var_com[i-M] =np.divide(np.multiply(mkt_cap, mar_Var_lin[i-M]),Var_lin[i-M]) * 100
        percent_ES_com[i-M] =np.divide(np.multiply(mkt_cap, mar_ES_lin[i-M]),ES_lin[i-M]) * 100
        percent_contribution[i-M] = np.divide(np.multiply(mkt_cap,np.dot(covariance_Delta,mkt_cap)),np.dot(np.dot(mkt_cap.T, covariance_Delta),mkt_cap))* 100
    return percent_Var_com,percent_ES_com,percent_contribution


def graphs_plot(data,time,title = 'Component Value at Risk'):
    df = pd.DataFrame(data)
    df.columns = ['Walmart','Target','Costco','Citigroup','JP Morgan']
    df.index = time.iloc[0:,0]
    df.plot()
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('/Users/popeye/Desktop/MF 731/Homework 3 /Five_Stock_Prices.csv')
    Stock_Data = df.iloc[:, 1:]
    S1 = np.log(Stock_Data / Stock_Data.shift(1))
    log_return = S1.iloc[1:]
    mkt_cap = np.array([3000000, 3000000, 3000000, 3000000, 3000000])
    mkt_cap_wgt =np.divide(mkt_cap,sum(mkt_cap))
    M = 50
    N = len(log_return)
    alpha = 0.99
    lamda = 0.94
    theta = 0.96
    mu = log_return.iloc[:M].mean()
    covariance = log_return.iloc[:M].cov()
    percent_com_Var,percent_com_ES,percent_contribution = Componenet_risk_measure(mu, covariance, lamda, theta, log_return, M, mkt_cap, alpha)
    time = pd.DataFrame(np.zeros_like(df.iloc[M:, 0]))
    for i in np.arange(M,N+1):
        b2 =dt.datetime.fromordinal(int(df.iloc[i, 0]+ 693594))
        b2 = str(b2)[:10]
        time.iloc[i - M, 0] = dt.datetime.strptime(b2,'%Y-%m-%d').date()

    graphs_plot(percent_com_Var, time)
    graphs_plot(percent_com_ES, time,'Component Expected Shortfall')
    graphs_plot(percent_contribution, time,'Percentage Contribution to Variance')

