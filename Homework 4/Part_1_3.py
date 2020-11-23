from scipy.stats import norm
import math
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import genextreme as gev
from scipy.stats import genpareto as gpd
import matplotlib.pyplot as plt

def EWMA(mu,covariance,lamda,theta,log_return):

    N = log_return.shape[0]
    for i in range(1, N-499):
        covariance = (theta * covariance**2 + (1 - theta) * np.dot(( log_return.iloc[499 + i,:] - mu ),(log_return.iloc[499 + i,:] - mu)))**0.5
        mu = mu * lamda + (1- lamda) *  log_return.iloc[499 + i,:]
    return np.array(mu),np.array(covariance)

def EMP_VaR(alpha, sort_port_loss):
    N = len(sort_port_loss)
    emp_var_table = np.zeros_like(alpha)
    for i in range(len(alpha)):
        emp_var_table[i] = sort_port_loss[math.floor(alpha[i] * N)]
    return emp_var_table

def EWMA_VaR(ewma_mu,ewma_covariance, alpha,port_value ):
    return -port_value * ewma_mu + ewma_covariance*abs(port_value)*norm.ppf(alpha)

def block_maximuns(num_blocks,blocks_size, port_loss):
    max_in_blocks = np.zeros(num_blocks)
    for i in range(num_blocks):
        max_in_blocks[i] = np.max(port_loss.iloc[blocks_size*i + 1:blocks_size*i+blocks_size,:])
    return max_in_blocks

def GEV_VaR(gev_values,blocks_size,alpha):
    if (-gev_values[0])<0.00001:
        gev_var = gev_values[1] - gev_values[2] * np.log(-blocks_size*np.log(alpha))
    else:
        gev_var = gev_values[1] - (gev_values[2]/(-gev_values[0])) * (1-(-blocks_size*np.log(alpha))**(gev_values[0]))
    return gev_var

def GPD_VaR(gpd_values,u_value,cdf_u_value,alpha):
    if gpd_values[0] < 0.00001:
        gpd_var = u_value + gpd_values[2]*np.log((1-cdf_u_value)/(1-alpha))
    else:
        gpd_var = u_value + (gpd_values[2]/gpd_values[0])*(((1-cdf_u_value)/(1-alpha))**(gpd_values[0])-1)
    return gpd_var

def see_crash(VaR,crash_loss,delta_alpha):
    if np.max(VaR) < crash_loss:
        see_crash = 'No'
        alpha_index = len(VaR)
    else:
        see_crash = 'Yes'
        alpha_index = np.min(np.where(VaR >= crash_loss))
    c_alpha = low_alpha + (alpha_index - 1) * delta_alpha
    return see_crash,c_alpha,alpha_index


if __name__ == '__main__':
    df = pd.read_csv('/Users/popeye/Desktop/MF 731/Homework 4/SP500_Log_Returns.csv',header=None)
    time = pd.DataFrame(np.zeros_like(df.iloc[:, 0]))
    for i in range(len(df)):
        b2 = dt.datetime.fromordinal(int(df.iloc[i, 0] + 693594))
        b2 = str(b2)[:10]
        time.iloc[i, 0] = dt.datetime.strptime(b2, '%Y-%m-%d').date()
    log_return = pd.DataFrame(df.iloc[:, 1:])
    log_return.index = time.iloc[:,0]
    port_value = 1000000
    port_loss = -port_value * log_return
    sort_port_loss = np.array(port_loss)[:,0].T
    sort_port_loss.sort()
    M =500
    lamda = theta = 0.97
    mu = np.mean(log_return.iloc[:499,])
    covariance = np.std(log_return.iloc[:499,])
    ewma_mu,ewma_covariance = EWMA(mu,covariance,lamda,theta,log_return)

    blocks_size = 125
    num_blocks = math.floor(log_return.shape[0]/blocks_size)
    max_in_blocks = block_maximuns(num_blocks,blocks_size, port_loss)
    gev_values = gev.fit(max_in_blocks)

    alpha_thresh = 0.95
    u_value = sort_port_loss[math.floor(log_return.shape[0] * alpha_thresh)]
    cdf_u_value = math.ceil(log_return.shape[0] * alpha_thresh) / log_return.shape[0]
    sort_data = port_loss[port_loss > u_value].dropna()- u_value
    y_data = np.array(sort_data)[:, 0].T
    gpd_values = gpd.fit(y_data)

    high_alpha = 0.9999
    low_alpha = 0.99
    delta_alpha = 0.000099
    alpha = np.arange(low_alpha,high_alpha,delta_alpha)

    emp_var_table = EMP_VaR(alpha, sort_port_loss)
    ewma_var_table = EWMA_VaR(ewma_mu[0],ewma_covariance[0], alpha,port_value )
    gev_var_table = GEV_VaR(gev_values,blocks_size,alpha)
    gpd_var_table = GPD_VaR(gpd_values,u_value,cdf_u_value,alpha)

    crash_log_return = -0.099452258
    crash_loss = -port_value * crash_log_return
    emp_pre = see_crash(emp_var_table, crash_loss, delta_alpha)
    ewma_pre = see_crash(ewma_var_table, crash_loss, delta_alpha)
    gev_pre = see_crash(gev_var_table, crash_loss, delta_alpha)
    gpd_pre = see_crash(gpd_var_table, crash_loss, delta_alpha)

    plt.plot(alpha, emp_var_table, label='EMP_VaR',linestyle="-")
    plt.plot(alpha, ewma_var_table, label='EWMA_VaR',linestyle="--")
    plt.plot(alpha, gev_var_table, label='GEV_VaR',linestyle="-.")
    plt.plot(alpha, gpd_var_table, label='GPD_VaR',linestyle=":")
    plt.title('VaR by the alpha from 0.99 to 0.9999')
    plt.legend()
    plt.show()

    print('Crash Loss: %0.2f' %crash_loss)
    print('Empirical VaR see Crash? %s' %emp_pre[0])
    print('Closest Empirical VaR alpha: %1.6f' %emp_pre[1])
    print('Closest Empirical VaR: %10.2f' %emp_var_table[emp_pre[2]-1])
    print('EWMA VaR see Crash? %s' % ewma_pre[0])
    print('Closest EWMA VaR alpha: %1.6f' % ewma_pre[1])
    print('Closest EWMA VaR: %10.2f' % ewma_var_table[ewma_pre[2] - 1])
    print('GEV VaR see Crash? %s' %gev_pre[0])
    print('Closest GEV VaR alpha: %1.6f' %gev_pre[1])
    print('Closest GEV VaR: %10.2f' %gev_var_table[gev_pre[2]-1])
    print('GPD VaR see Crash? %s' %gpd_pre[0])
    print('Closest GPD VaR alpha: %1.6f' %gpd_pre[1])
    print('Closest GPD VaR: %10.2f' %gpd_var_table[gpd_pre[2]-1])

    print('alpha = 0.9999 Empirical VaR: %10.2f' %emp_var_table[emp_pre[2]-1])
    print('alpha = 0.9999 EWMA VaR: %10.2f' % ewma_var_table[ewma_pre[2] - 1])
    print('alpha = 0.9999 GEV VaR: %10.2f' %gev_var_table[gev_pre[2]-1])
    print('alpha = 0.9999 GPD VaR: %10.2f' %gpd_var_table[gpd_pre[2]-1])

    print('The Highest VaR is generated by the empirical method,\n\
and the loss when X = -0.99452258 can not be predicted by these methods')