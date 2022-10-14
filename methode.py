# This file is used to create figures with multiple curves in one



#%% import packages
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels as sm
import statsmodels.formula.api as smf
import fitting
#%%

# A method that uses input data to fit historic NPV to historic expansion --> gives us the function fun

def methode(data, colour, label):
    # %% split data to x and y
    x = data.iloc[:,:1]
    y = data.iloc[:,1:2]
    # scatter plot datapoints
    plt.scatter(x.to_numpy().tolist(),y.to_numpy().tolist())
    #plt.show()
    # transform x & y to ndarray and reshape array
    x = x.to_numpy().reshape(1,7)
    y = y.to_numpy().reshape(1,7)
    # fit function with polyfit (x, y, degree)
    fit = np.polyfit(x[0,:],y[0,:],3)
    print(fit)
    # create function from fit
    fun = np.poly1d(fit)
    # plot function
    poly = np.linspace(-15000, 4000, 1000)
    plt.plot(x, y, 'o', color = colour) 
    plt.plot(poly, fun(poly), '- -', color = colour, label= label)
    plt.ylim(0,120000)
    plt.xlabel('Net Present Value')
    plt.ylabel('Cumulated No. of HSS')
    plt.legend(loc='upper left')
   
# A method to calculate the r_squared value to control the values given by smf

def r_squared(data):
    # %% split data to x and y
    x = data.iloc[:,:1]
    y = data.iloc[:,1:2]
    # transform x & y to ndarray and reshape array
    x = x.to_numpy().reshape(1,7)
    y = y.to_numpy().reshape(1,7)
    # hier bestimmen wir y^, also die geschätzen werte für y
    y_dach = fitting.fun(x)
    # hier den mittelwert von y bestimmen
    y_strich = np.mean(y)
    # hier nenner und zähler bestimmen
    zähler = y-y_dach
    nenner = y-y_strich
    # hier nenner und zähler jeweils quadrieren
    zähler_squared = zähler**2
    nenner_squared = nenner**2
    # summen der quadrierten nenner und zähler bilden
    sum_nenner_squared = np.sum(nenner_squared[0][:])
    sum_zähler_squared = np.sum(zähler_squared[0][:])
    # r_squared bestimmen
    r_squared = 1- (sum_zähler_squared/sum_nenner_squared)
    # r_squared printen
    print(r_squared)

# To view individual datapoints: enter fun(future NPV) in jupyter console

# load all datasets
#%% import data
low10 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/low_10%.csv',sep = ";")
low4 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/low_4%.csv',sep = ";")
low2 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/low_2%.csv',sep = ";")
mod10 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/moderate_10%.csv',sep = ";")
mod4 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/moderate_4%.csv',sep = ";")
mod2 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/moderate_2%.csv',sep = ";")
high10 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/high_10%.csv',sep = ";")
high4 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/high_4%.csv',sep = ";")
high2 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/high_2%.csv',sep = ";")


#%% run methode with all wanted datasets
methode(high10, 'g', 'scenario high (10%)')
methode(high4, 'g', 'scenario high (4%)')
methode(high2, 'g', 'scenario high (2%)')
methode(low10, 'r', 'scenario low (10%)')
methode(low4, 'r', 'scenario low (4%)')
methode(low2, 'r', 'scenario low (2%)')
methode(mod10, 'b', 'scenario mod (10%)')
methode(mod10, 'b', 'scenario mod (4%)')
methode(mod10, 'b', 'scenario mod (2%)')

# plt.show and plt. savefig to save and reset graphs
plt.savefig('anzahl_all.pdf')
plt.show()
# %% hier überprüfen und statistische results bekommen
results = smf.ols(formula='y~fun(x)', data=high10)
r_squared(high10)


