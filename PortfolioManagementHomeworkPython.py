
# coding: utf-8

# # Programming Assignment 1
# # MATH-UA 250 Mathematics of Finance
# # Portfolio Management


import sys
sys.version
sys.version_info






#get_ipython().magic(u'matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# #  File Contents
# Returns to the following asset classes are contained in the file
#  * "MSCI WORLD"
#  * "MSCI AC WORLD"    
#  * "MSCI EUROPE"
#  * "MSCI EM"
#  * "MSCI EAFE"        
#  * "MSCI PACIFIC"
#  * "MSCI USA"
#  * "Treasury.Bond.10Y"
#  * "Treasury.Bill.90D"
# 
# 
#  <span style="color:blue">Our set of risky assets for the analysis is  MSCI EUROPE, MSCI USA , MSCI PACIFIC, Treasury.Bond.10Y </span>
# 


#
# read in the data

inpath  = "/Users/fengjiayi/Documents/MathFinance/"
outpath = "/Users/fengjiayi/Documents/MathFinance/"

infile  = "ReturnsPortfolios.csv"

print(inpath+infile)
indata = pd.read_csv( inpath + infile)


# check data

indata.head(5)


# divide risky assets and risky free assets that we want to analysis

RiskyAsset     = ["MSCI EUROPE","MSCI USA","MSCI PACIFIC","Treasury.Bond.10Y"]
RiskFreeAsset  = "Treasury.Bill.90D"



# 
# print out first 3 rows and all columns of risky assets
indata.loc[1:3,RiskyAsset]


# 
# ## <span style="color:blue">Part 1: Risky Assets Analysis</span>
# 
#  Create a table calculating a-h for the risky assets:
# 1. mean
# 2.	median
# 3.	standard deviations 
# 4.	skew 
# 5.	kurtosis
# 6.  return - risk ratio 
# 7.  plot the assets classes on a return - risk graph   
# 8.  Write up a paragraph comparing the statistics of the risky assets
# 


mean = indata[RiskyAsset].mean()
median = indata[RiskyAsset].median()
std = indata[RiskyAsset].std()
skew = indata[RiskyAsset].skew()
kurtosis = indata[RiskyAsset].kurt()
rrr = indata[RiskyAsset].mean()/indata[RiskyAsset].std()
d = {"mean":mean, "median":median,"std":std,"skew":skew,"kurtosis":kurtosis,"rrr":rrr}
table = pd.DataFrame(d)
print(table)
rrr.plot(kind = 'bar')
plt.show()

# 
# ## <span style="color:blue">Part 2:  Distributions </span>
# Plot the distributions for each asset class
# 
# Compare the distribution


indata["MSCI WORLD"].plot.hist(stacked=True, bins=20)
plt.show()
indata["MSCI AC WORLD"].plot.hist(stacked=True, bins=20)
plt.show()
indata["MSCI EUROPE"].plot.hist(stacked=True, bins=20)
plt.show()
indata["MSCI EM"].plot.hist(stacked=True, bins=20)
plt.show()
indata["MSCI EAFE"].plot.hist(stacked=True, bins=20)
plt.show()
indata["MSCI PACIFIC"].plot.hist(stacked=True, bins=20)
plt.show()
indata["MSCI USA"].plot.hist(stacked=True, bins=20)
plt.show()
indata["Treasury.Bond.10Y"].plot.hist(stacked=True, bins=20)
plt.show()
indata["Treasury.Bill.90D"].plot.hist(stacked=True, bins=20)
plt.show()




# ## <span style="color:blue"> Part 3: Covariance and Correlation Matrices  </span>
# 
#  1.	Calculate the covariance matrix 
#  2. Calculate the correlation matrix
#  3.	Verify the covariance matrix is non-singular
#  4.	Verify the covariance matrix is symmetric and positive definite
# 
CovMatrix = indata.cov()
CorrMatrix = indata.corr()
non_singular = np.linalg.matrix_rank(CovMatrix) == CovMatrix.shape[0]
Symmetric = ((CovMatrix.transpose() == CovMatrix).all()).all()
PosDef = (np.linalg.eigvals(CovMatrix)>0).all()
print(CovMatrix)
print(CorrMatrix)
print("non-singularity: ",non_singular)
print("Symmetric: ",Symmetric)
print("Positive Definite: ", PosDef)



# ## <span style="color:blue">Part 4: Calculate the weights, returns and risks for the following portfolios     </span>
#  1.	Minimum variance portfolio
#  2. Portfolio with expected return of 9% and minimum variance
#  3. Efficicent frontier (calculate the weights, returns and risk of at least 7 portfolios)
#  4. Plot the portfolios and asset classes on a graph
#  5. Plot an equal weighted portolio on the graph
#  6. What is the expected return and risk of the minimum variance portfolio
#   
u = np.array([1,1,1,1])
CovMRisky = indata[RiskyAsset].cov()
CovMR_inv = np.linalg.inv(CovMRisky)
#minimum variance portfolio
w_mvp = np.divide(np.dot(u,CovMR_inv),np.dot(np.dot(u,CovMR_inv),u.transpose()))
miu_mvp = np.dot(w_mvp,mean.transpose())
std_mvp = np.sqrt(np.dot(np.dot(w_mvp,CovMRisky),w_mvp.transpose()))
print("weight of mvp: ",w_mvp)
print("return of mvp: ",miu_mvp)
print("risk of mvp: ",std_mvp)
#minimum variance portfolio with return = 0.09
a = np.dot(np.dot(mean,CovMR_inv),mean.transpose())
b = np.dot(np.dot(u,CovMR_inv),mean.transpose())
c = np.dot(np.dot(mean,CovMR_inv),u.transpose())
d = np.dot(np.dot(u,CovMR_inv),u.transpose())
M = np.array([[a,b],[c,d]])
Minv = np.linalg.inv(M)
miu = 0.09/12
Coef_array = np.dot(Minv,[miu,1])
w_009 = np.add(np.dot(np.dot(Coef_array[0],mean),CovMR_inv),np.dot(np.dot(Coef_array[1],u),CovMR_inv))
print("minimum variance portforlio with return = 0.09: ",w_009)
# efficient frontier
def MVP(E):
    lamda_array = np.dot(Minv,[E,1])
    w_E = np.add(np.dot(np.dot(lamda_array[0],mean),CovMR_inv),np.dot(np.dot(lamda_array[1],u),CovMR_inv))
    print("weight for: ",E,"is: ",w_E)
    risk_E = np.sqrt(np.dot(np.dot(w_E,CovMRisky),w_E.transpose()))
    print("risk is: ",risk_E)
    print("return is: ",E)
    plt.plot(risk_E,E,'b.')
    
plt.plot(std_mvp,miu_mvp,'g^')
MVP(0.09/12)
MVP(0.1/12)
MVP(0.11/12)
MVP(0.12/12)
MVP(0.13/12)
MVP(0.14/12)
MVP(0.15/12)
MVP(0.16/12)

#equal weight
w_eq = 0.25*u
std_eq = np.sqrt(np.dot(np.dot(w_eq,CovMRisky),w_eq.transpose()))
miu_eq = np.dot(w_eq,mean.transpose())
plt.plot(std_eq,miu_eq,'r.')
plt.show()
# ## <span style="color:blue">Part 5: Calculate the beta (t-stat, R-square) of each the risk asset class to the following 2 'market' portfolios   </span>
# 
#   Market Portfolio 1: "MSCI AC WORLD"    
#   Market Portfolio 2: "MSCI USA"
#   
#   Compare the betas for each asset class
# 
# Assume R = 0, beta = miuV-miuM
R = indata[RiskFreeAsset].mean()
w_m = np.divide(np.dot(mean-np.dot(R,u),CovMR_inv),np.dot(np.dot(mean-np.dot(R,u),CovMR_inv),u.transpose()))
miu_m = np.dot(w_m,mean.transpose())
beta_MSCIACWORLD = (indata["MSCI AC WORLD"].mean()-R)/(miu_m-R)
beta_MSCIUSA = (indata["MSCI USA"].mean()-R)/(miu_m-R)
print("beta of MSCI AC WORLD is: ", beta_MSCIACWORLD)
print("beta of MSCI USA is: ", beta_MSCIUSA)

