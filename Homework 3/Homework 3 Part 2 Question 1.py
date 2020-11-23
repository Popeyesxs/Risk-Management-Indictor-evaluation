from BlackScholes import *
import numpy as np


def h_t(S0,K,t,sigma,r):
    return BSEuroCallOption(S0,K,t,sigma,r).delta() + BSEuroPutOption(S0,K,t,sigma,r).delta()

def V_t(S0,K,t,sigma,r):
    return BSEuroCallOption(S0,K,t,sigma,r).value() + BSEuroPutOption(S0,K,t,sigma,r).value() - h_t(158.12,K,0.25,.2214,r) * S0

def scenario_analysis(S0,K,delta,sigma,sigma_delta,sigma_delta_w,r,X,X_w,T):
    loss = np.zeros([len(X),len(X)])
    loss_weight = np.zeros([len(X),len(X)])
    for i in range(len(X)):
        for j in range(len(sigma_delta)):
            loss[i][j] = V_t(S0*np.exp(X[i]),K,T-5*delta,sigma*sigma_delta[j],r)-V_t(S0,K,T,sigma,r)
            loss_weight[i][j] = loss[i][j] * X_w[i] * sigma_delta_w[j]

    return loss,loss_weight

if __name__ == '__main__':
    r = 0.0132
    mu = .15475
    sigma = .2214
    T = .25
    Delta = 1 / 252
    S0 = 158.12
    K = 160
    t = 0
    X = [-.2, -.1, -.05, .05, .1, .2]
    X_w = [.5, .75, 1, 1, .75, .5]
    sigma_delta = [.5, .75, 1.25, 1.5, 1.75, 2]
    sigma_delta_w = [.5, 1.25, .75, .75, 1.25, .5]
    loss,loss_weight = scenario_analysis(S0, K, Delta, sigma, sigma_delta, sigma_delta_w, r, X, X_w,T)
    m = np.argmax(loss)
    r, c = divmod(m, loss.shape[1])
    print('Worst Case Scenario Risk Measure %.4f' %loss[r][c])
    print('Worst Case Log Return %.4f' %X[r])
    print('Worst Case Beta %.4f' %sigma_delta[c])

    m2 = np.argmax(loss_weight)
    i, j = divmod(m2, loss.shape[1])
    print('Weighted Worst Case Scenario Risk Measure %.4f' % loss_weight[i][j])
    print('Weighted Worst Case Log Return %.4f' %X[i])
    print('Weighted Worst Case Beta %.4f' % sigma_delta[j])
    print('Weighted Worst Case Loss %.4f' %loss[i][j])