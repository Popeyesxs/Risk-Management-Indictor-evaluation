import pandas as pd
import numpy as np
import math
from scipy.stats import norm

def Empirical_distribution_method(N,M,emp_return_data):
    """
    (a) the empirical distribution method
    :param N: the return data points
    :param M: 4 years
    :return:the total number of exceedances for Empirical distribution method
    """
    var_data = np.zeros(N - M)
    exceed_data = np.zeros(N - M)
    for i in range(N-M):
        emp_data = np.array(emp_return_data[i:i+M]).T
        emp_data.sort()
        var_data[i] =  emp_data[0][math.floor(M*0.95)]
        exceed_data[i] = emp_return_data.iloc[i+M] > var_data[i]
    return sum(exceed_data)

def Normal_log_return_method(N,M,ewma_return_data,labda,theta):
    """
    (b) the normal log return method
    :param N: the return data points
    :param M: 4 years
    :return:the total number of exceedances for the normal log return method
    """
    mu_delta = ewma_return_data.iloc[0:M-1].mean()
    sigma_delta = ewma_return_data.iloc[0:M-1].std()
    var_data = np.zeros(N - M)
    exceed_data = np.zeros(N - M)
    for i in range(N - M):
        mu_delta = labda * mu_delta + (1 - labda) * ewma_return_data.iloc[i + M - 1]
        sigma_delta  =  (theta * sigma_delta**2 + (1- theta) * (ewma_return_data.iloc[i + M - 1] - mu_delta)**2)**0.5
        var_data[i] = 1-(np.exp(mu_delta+sigma_delta* norm.ppf(1 - 0.95)))
        exceed_data[i] = (1- np.exp(ewma_return_data.iloc[i + M]) )> var_data[i]
    return sum(exceed_data)


if __name__ == "__main__":
    df = pd.read_csv('/Users/Popeye/Desktop/MF 731/Homework 2/SP_Prices.csv',header=None)
    Stock_Data = df.iloc[:,1:]
    Stock_Data.index = df.iloc[:,0]
    Stock_Data.columns = ['SP_price']
    S1 = np.log(Stock_Data / Stock_Data.shift(1))
    log_return = S1.iloc[1:]
    N = len(log_return)
    alpha = 0.95
    beta = 0.05
    M = 1010
    labda =0.97
    theta = 0.97
    emp_return_data = -(np.exp(log_return) - 1)
    ewma_return_data = log_return
    mean_value = (N - M) * (1 - alpha)
    CI_Low = (N - M) * (1 - alpha) - norm.ppf(1 - beta / 2) * ((N - M) * alpha * (1 - alpha)) ** (0.5)
    CI_High = (N - M) * (1 - alpha) + norm.ppf(1 - beta / 2) * ((N - M) * alpha * (1 - alpha)) ** (0.5)
    print('Mean: %.2f' %mean_value)
    print('Low confidence interval: %.2f' %CI_Low)
    print('High confidence interval: %.2f' %CI_High)
    print('Empirical number of exceedences: %.2f' %Empirical_distribution_method(N,M,emp_return_data))
    print('EWMA number of exceedences: %.2f' %Normal_log_return_method(N,M,ewma_return_data,labda,theta))
