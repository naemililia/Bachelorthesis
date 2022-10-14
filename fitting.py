# This file is used to create a function that fits the historic NPV to the historic expansion of HSS


#%% import packages
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels as sm
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score
import scipy.stats 
import methode
import openpyxl 



#%%
# load all datasets (Expansion in No. of HSS)
low10 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/low_10%.csv',sep = ";")
low4 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/low_4%.csv',sep = ";")
low2 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/low_2%.csv',sep = ";")
mod10 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/moderate_10%.csv',sep = ";")
mod4 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/moderate_4%.csv',sep = ";")
mod2 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/moderate_2%.csv',sep = ";")
high10 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/high_10%.csv',sep = ";")
high4 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/high_4%.csv',sep = ";")
high2 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/high_2%.csv',sep = ";")

# load all datasets (Expansion in net nominal power)
nn_low10 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/nettonennleistung/low 10%.csv',sep = ";")
nn_low4 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/nettonennleistung/low 4%.csv',sep = ";")
nn_low2 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/nettonennleistung/low 2%.csv',sep = ";")
nn_mod10 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/nettonennleistung/moderate 10%.csv',sep = ";")
nn_mod4 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/nettonennleistung/moderate 4%.csv',sep = ";")
nn_mod2 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/nettonennleistung/moderate 2%.csv',sep = ";")
nn_high10 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/nettonennleistung/high 10%.csv',sep = ";")
nn_high4 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/nettonennleistung/high 4%.csv',sep = ";")
nn_high2 = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/nettonennleistung/high 2%.csv',sep = ";")

#%% import yearly NPVs
npv_yearly = pd.read_csv('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/npv_interpoliert.csv',sep = ";")
#%% Split into columns --> to be entered into the function fun (fitted by polyfit) 
npv_high2 = npv_yearly.iloc[:,:1]
npv_high4 = npv_yearly.iloc[:,1:2]
npv_high10 = npv_yearly.iloc[:,2:3]
npv_mod2 = npv_yearly.iloc[:,3:4]
npv_mod4 = npv_yearly.iloc[:,4:5]
npv_mod10 = npv_yearly.iloc[:,5:6]
npv_low2 = npv_yearly.iloc[:,6:7]
npv_low4 = npv_yearly.iloc[:,7:8]
npv_low10 = npv_yearly.iloc[:,8:9]

#%% choose a dataset from above and call it data
data = mod4

# Use input data to fit historic NPV to historic expansion --> gives us the function fun

# %% split data to x and y
x = data.iloc[:,:1]
y = data.iloc[:,1:2]
#%% scatter plot datapoints
plt.scatter(x.to_numpy().tolist(),y.to_numpy().tolist())
plt.show()
#%% transform x & y to ndarray and reshape array
x = x.to_numpy().reshape(1,7)
y = y.to_numpy().reshape(1,7)
#%% fit function with polyfit (x, y, degree) --> calculates parameters
fit = np.polyfit(x[0,:],y[0,:],3)
print(fit)
#%% create function from fit --> makes a function from the parameters
fun = np.poly1d(fit)
#%% plot function
poly = np.linspace(-17000, 4000, 100)
plt.plot(x, y, '.', poly, fun(poly), '-')
plt.ylim(0,60000)

# To view individual datapoints (y-values --> value of expansion): enter fun(future NPV) in jupyter console



#%% Save plot
plt.savefig('random.pdf')
plt.show()

#%% Residual errors (e =y -y_predicted --> e = y - fun(x))
error = y - fun(x)
print(error)
plt.scatter(x,error)

#%% Use smf to get summary of statistical values. The model that is evaluated shows how well y is represented by fun(x)
x_neu = high2.iloc[:,:1]
y_neu = high2.iloc[:,1:2]
results = smf.ols(formula='y_neu ~ fun(x_neu)', data= high2).fit()
results.summary()

