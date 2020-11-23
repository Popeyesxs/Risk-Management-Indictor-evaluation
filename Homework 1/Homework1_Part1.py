import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from arch import arch_model


# Change the y to percentage
def to_percent(temp,position):
    return '%1.2f'%(100*temp) + '%'

#4
# (a) The moving average method with a look-back period of n = 100 days
def MA(S,periods,delta):
    X_t = np.log(S / S.shift(1))
    X_t.columns = ['Return rate']
    MA = []
    for i in range(len(S.iloc[periods:,])):
        volatility = (((X_t.iloc[i:periods+i,0])**2).mean() / delta)**0.5
        MA += [volatility]
    return MA

# (b) the exponentially weighted moving average method with lambda = 0.94 and lambda = 0.97
def EWMA(S,periods,delta,lamda):
    X_t = np.log(S / S.shift(1))
    X_t.columns = ['Return rate']
    EWMA = []
    for i in range(len(S.iloc[periods:, ])):
        if i == 0:
            EWMA += [((X_t.iloc[i:periods+i,0])**2).mean()]
        else:
            volatility = EWMA[i-1] * lamda + (1 - lamda) *( X_t.iloc[periods+i, 0]**2)
            EWMA += [volatility]
    return np.sqrt(np.divide(EWMA,delta))


#plot the figure of question 4
def plot_curve():
    # Buliding a DataFrame to including MA and EWMA
    df1 = pd.DataFrame(MA(S, 100,1/252))
    df1.columns = ['MA(100 days)']
    df1['EWMA(0.94)'] = EWMA(S,100,1/252, 0.94)
    df1['EWMA(0.97)'] = EWMA(S, 100, 1 / 252, 0.97)
    df1.index = S_A_volatility.index
    plt.rcParams['figure.figsize'] = (12.0, 5.0)
    df1.plot()
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gcf().autofmt_xdate()
    plt.grid(axis="y")
    plt.show()

#5 GARCH Volatility Estimates
def fit_data_coefficient(S):
    X_t = np.log(S / S.shift(1))
    X_t.columns = ['Return rate']
    X_period = X_t.loc['2015-08-04':'2019-07-31']
    garch11 = arch_model(X_period, vol='GARCH', p=1, q = 1, dist='Normal',mean= 'zero')
    res = garch11.fit(update_freq=5)
    return res.params


def simulate_garch(dates, coefficient, no_samples, starting_val,starting_std):
    omega = coefficient[0]
    alpha = coefficient[1]
    beta = coefficient[2]
    n = dates.shape[0]
    epsilon = np.zeros((n, no_samples))
    epsilon[0, :] = starting_val
    sigma2 = np.zeros((n, no_samples))
    sigma2[0, :] = starting_std**2
    for i in range(0, no_samples):
        for k in range(1, n):
            sigma2[k,i] = omega + alpha*(epsilon[k-1,i]**2) + beta*sigma2[k-1,i]
            epsilon[k, i] = np.random.normal() * np.sqrt(sigma2[k,i])
    return(pd.DataFrame(data=sigma2, index=dates.index))

#plot the figure of question 5
def plot_curve2():
    coefficient = fit_data_coefficient(S)
    X_t = np.log(S / S.shift(1))
    X_t.columns = ['Return rate']
    X_period = X_t.loc['2016-07-01':'2017-06-30']
    X_t.columns = ['Return rate']
    Initial_std = X_t.loc['2019-01-31':'2019-07-31'].std()
    Initial_Xt = X_t.loc['2019-07-31']
    dates1 = X_t.loc['2019-07-31':'2020-07-13'].index
    dates_est = pd.DataFrame(index=dates1)
    simulation = simulate_garch(dates_est, coefficient, 1, Initial_Xt, Initial_std)
    # Buliding a DataFrame to including MA and EWMA
    df2 = pd.DataFrame(MA(S, 100,1/252))
    df2.columns = ['MA(100 days)']
    df2['EWMA(0.94)'] = EWMA(S,100,1/252, 0.94)
    df2['EWMA(0.97)'] = EWMA(S, 100, 1 / 252, 0.97)
    df2.index = S_A_volatility.index
    df2['GARCH(1,1)']= np.sqrt(simulation * 252)
    df2.fillna(0)
    plt.rcParams['figure.figsize'] = (12.0, 5.0)
    df2.plot()
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gcf().autofmt_xdate()
    plt.grid(axis="y")
    plt.show()

def plot_curve3():
    coefficient = fit_data_coefficient(S)
    X_t = np.log(S / S.shift(1))
    X_t.columns = ['Return rate']
    X_period = X_t.loc['2016-07-01':'2017-06-30']
    X_t.columns = ['Return rate']
    Initial_std = X_t.loc['2019-01-31':'2019-07-31'].std()
    Initial_Xt = X_t.loc['2019-07-31']
    dates1 = X_t.loc['2019-07-31':'2020-07-13'].index
    dates_est = pd.DataFrame(index=dates1)
    no_samples = 50
    simulation = simulate_garch(dates_est, coefficient, no_samples, Initial_Xt, Initial_std)
    annu_simulation = np.sqrt(simulation * 252)
    df1 = pd.DataFrame(MA(S, 100, 1 / 252))
    df1.columns = ['MA(100 days)']
    df1['EWMA(0.94)'] = EWMA(S, 100, 1 / 252, 0.94)
    df1['EWMA(0.97)'] = EWMA(S, 100, 1 / 252, 0.97)
    df1.index = S_A_volatility.index
    plt.rcParams['figure.figsize'] = (12.0, 5.0)
    plt.plot(dates1,annu_simulation,linestyle=":")
    plt.plot(dates1,df1['2019-07-31':'2020-07-13'],label = ['MA(100 days)','EWMA(0.94)','EWMA(0.97)'])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gcf().autofmt_xdate()
    plt.grid(axis="y")
    plt.show()

if __name__ == "__main__":
    df = pd.read_excel('/Users/Popeye/Desktop/MF 731/Homework 1/SPData.xlsx')
    #a look_back period of 100 days
    periods = 100
    delta = 1/252
    SPD_DATA = df.iloc[:,1:]
    SPD_DATA.index = df.iloc[:,0]
    S = SPD_DATA.iloc[::-1]
    S_A_volatility = S.iloc[100:,]
    coefficient = fit_data_coefficient(S)
    X_t = np.log(S / S.shift(1))
    X_t.columns = ['Return rate']
    X_period = X_t.loc['2016-07-01':'2017-06-30']
    X_t.columns = ['Return rate']
    plot_curve()
    plot_curve2()
    plot_curve3()



