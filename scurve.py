# This file is used to create the s-curve on the basis of the priorly calculated NPV curve


#%% import packages 
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#%% load data 
ydata = pd.read_excel('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/Ausbau in HSS yearly.xlsx', sheet_name='Base low 2', usecols='O')
xdata = pd.read_excel('C:/Users/Naemi/Documents/KIT/Semester/Vertiefungsstudium/Bachelorarbeit/Regression/Ausbau in HSS yearly.xlsx', sheet_name='Base low 2', usecols='A')


#%% define S-curve 
# L is the potential
# x0 is the point in time in which the truning point is reached
# k is a growth factor
# b is the minimum value of the function
def sigmoid (x,L,x0,k,b):
    y = L / (1+ np.exp(-k*(x-x0)))+b
    return y

#%% make a parameter guess
# turning point at 10% of potential: L = 1/5*Potential
# turning point at 25% of potential: L = 1/2*Potential
# turning point at 50% of potential: L = 1*Potential
# k: trial und error until best r_squared is reached
guess = [1221782/2, 2034, 0.17, 428]
#%% guess as input for sigmoid function
n = len(xdata)
y = np.empty(n)
for i in range(n):
    y[i] =sigmoid(xdata.values[i],guess[0], guess[1], guess[2], guess[3])

# plot curves
# the curve 'via NPV' corresponds to the previously calculated curve via the interpolated NPVs and the polyfit function
plt.plot(xdata,ydata, '-', color = 'green', label= 'via NPV')
plt.plot(xdata,y,'.', color = 'red', label = 's-curve')
plt.xlabel('Years')
plt.ylabel('Cumulated No. of HSS')
plt.legend(loc='upper left')
plt.savefig('s_curve_25.pdf')

# transform values into dataframe so they can be exported
y = pd.DataFrame(y)
#print(y)
#y.to_excel('l2_10.xlsx')


# give R squared value
# the 'relevant' values describe the number of years up to the turning point
from sklearn.metrics import r2_score
relevant_ydata = ydata[0:13]
relevant_y = y[0:13]
print('R^2: ', r2_score(relevant_ydata,relevant_y))







