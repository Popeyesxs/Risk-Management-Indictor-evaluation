from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import genpareto
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import math
from scipy.stats import genpareto as gpd
import matplotlib.pyplot as plt



def compound_poisson_distribution(lamda, num_values,mu,sigma):
    poisson_dis = np.random.poisson(lamda, num_values)
    compound_pos = np.zeros(num_values)
    for i in range(num_values):
        if poisson_dis[i]!= 0:
            compound_pos[i] =  np.sum(np.random.lognormal(mu,sigma,poisson_dis[i]))
    return compound_pos

def emp_cdf_tail(sample_data,cdf_values):
    emp_cdf_dis = ECDF(sample_data)
    emp_cdf_tail = np.zeros(len(cdf_values))
    for i in range(len(cdf_values)):
        emp_cdf_tail[i] = 1 - np.max(emp_cdf_dis.y[emp_cdf_dis.x<= cdf_values[i]])
    return emp_cdf_tail


if __name__ == '__main__':
    lamda = 100
    mu = 0.1
    sigma = 0.4
    M = 1000000
    alpha_0 = 0.99
    num_values = M
    mean_N = lamda
    mean_X = np.exp(mu+sigma ** 2 / 2)
    var_N = lamda
    var_X = np.exp(2 * mu+sigma ** 2) * (np.exp(sigma **2)-1)
    mean_X_cube=np.exp(3 * mu+9 * sigma ** 2 / 2)
    mean_X_sq = np.exp(2 * mu+2 * sigma ** 2)
    mean_SN = mean_N * mean_X
    var_SN = mean_N * var_X + (mean_X) ** 2 * var_N
    skew_SN = (mean_X_cube) * ( lamda * (mean_X_sq) ** 3) ** (-1 / 2)

    gamma_alpha = 4 * (skew_SN) ** (-2)
    gamma_beta = (gamma_alpha / (lamda *mean_X_sq)) ** (1 / 2)
    gamma_k = lamda *mean_X - gamma_alpha / gamma_beta
    sort_comp_poisson_rnd = compound_poisson_distribution(lamda, num_values,mu,sigma)
    sort_comp_poisson_rnd.sort()
    alpha_low = .99
    alpha_high = .99999
    low_val = sort_comp_poisson_rnd[math.floor(num_values * alpha_low)]
    high_val = sort_comp_poisson_rnd[math.floor(num_values * alpha_high)]
    mu_gp = low_val
    sample_data = compound_poisson_distribution(lamda, num_values, mu, sigma)
    data_gp = sample_data[sample_data>mu_gp] - mu_gp
    gpd_value = gpd.fit(data_gp)

    cdf_values = np.arange(low_val,high_val+(high_val-low_val)/1000,(high_val-low_val)/1000)

    norm_cdf_tail = 1 - norm.cdf(cdf_values,mean_SN,(var_SN)**(1/2))
    gamma_cdf_tail = 1 - gamma.cdf(cdf_values-gamma_k,gamma_alpha,scale=1/gamma_beta)
    GP_cdf_tail = 1 - (genpareto.cdf(cdf_values - low_val, gpd_value[0],scale = gpd_value[2]) * 0.01 + 0.99)
    emp_cdf_tails = emp_cdf_tail(sample_data,cdf_values)

    plt.loglog(cdf_values, norm_cdf_tail, label='CLT')
    plt.loglog(cdf_values, gamma_cdf_tail, label='GAMMA')
    plt.loglog(cdf_values, GP_cdf_tail, label='GP')
    plt.loglog(cdf_values, emp_cdf_tails, label='EMP')
    plt.title('LOG-LOG plot of 1-F_SN vs x')
    plt.legend()
    plt.show()
