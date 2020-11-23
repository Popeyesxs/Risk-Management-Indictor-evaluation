from scipy.stats import norm
import math
import numpy as np

def loss(M,lamda,S,mu,sigma,v,zeta):
    losses = np.zeros(M)
    normal_samples = np.random.randn(M,2)
    for i in range(M):
        losses[i] = -lamda*S*(np.exp(mu + sigma*normal_samples[i,0])*(1-(1/2)*(v+zeta*normal_samples[i,1]))-1)

    return losses



if __name__ == '__main__':
    lamda = 100
    S0 = 59
    mu = 0
    sigma = 0.4/(252**0.5)
    v = 0.002
    zeta = 0.0008
    k =3
    M = 1000000
    alpha = 0.99
    losses = loss(M, lamda, S0, mu, sigma, v, zeta)
    losses.sort()
    sim_liq_var = losses[math.floor(alpha*M)]
    th_var = lamda * S0 * (1 - np.exp(mu + sigma * norm.ppf(1 - alpha)))
    th_lc = (1 / 2) * lamda * S0 * (v + k * zeta)

    print('Confidence: %10.3f' %alpha)
    print('Simulated Liquidty VaR: %10.2f' %sim_liq_var)
    print('Theoretical VaR: %10.2f' %th_var)
    print('Simulated Liquidity Cost: %10.2f' %(sim_liq_var - th_var))
    print('Simulated Percentage Liquidity VaR Increase: %10.2f'%(100 * ((sim_liq_var / th_var) - 1)))
    print('Industry Approximate Liquidity VaR: %10.2f' %(th_var + th_lc))
    print('Industry Approximate Liquidity Cost: %10.2f'%th_lc)
    print('Industry Approximate Percentage Liquidity VaR Increase: %10.2f'%(100 * (th_lc / th_var)))



