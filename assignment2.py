import pandas as pd
from numpy.linalg import solve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# transform the variables 
#---------------------------------------------
# Load the dataset
data = pd.read_csv('.csv file')
data # I'm checking if dataset has been loaded correctly

# Clean the DataFrame by removing the row with transformation codes
data_cleaned = data.drop(index=0)
data_cleaned.reset_index(drop=True, inplace=True)
data_cleaned['sasdate'] = pd.to_datetime(data_cleaned['sasdate'], format='%m/%d/%Y')
data_cleaned

# Extract transformation codes
transformation_codes = data.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

# 'transformation_codes' has the variable’s name (Series) and its transformation 
# (Transformation_Code). There are six possible transformations
#to see them-->
transformation_codes
## - `transformation_code=1`: no trasformation
## - `transformation_code=2`: $\Delta x_t$
## - `transformation_code=3`: $\Delta^2 x_t$
## - `transformation_code=4`: $log(x_t)$
## - `transformation_code=5`: $\Delta log(x_t)$
## - `transformation_code=6`: $\Delta^2 log(x_t)$
## - `transformation_code=7`: $\Delta (x_t/x_{t-1} - 1)$

# Function to apply transformations based on the transformation code
#   1) We're defining 'apply_transformation' function

def apply_transformation(series, code):
    if code == 1:
        # No transformation
        return series
    elif code == 2:
        # First difference
        return series.diff()
    elif code == 3:
        # Second difference
        return series.diff().diff()
    elif code == 4:
        # Log
        return np.log(series)
    elif code == 5:
        # First difference of log
        return np.log(series).diff()
    elif code == 6:
        # Second difference of log
        return np.log(series).diff().diff()
    elif code == 7:
        # Delta (x_t/x_{t-1} - 1)
        return series.pct_change()
    else:
        raise ValueError("Invalid transformation code")

#   2) Applying the transformations to each column in df_cleaned based on transformation_codes
for series_name, code in transformation_codes.values:
    data_cleaned[series_name] = apply_transformation(data_cleaned[series_name].astype(float), float(code))


data_cleaned = data_cleaned[2:]     # We drop the first two observations of the dataset
data_cleaned.reset_index(drop=True, inplace=True) #We reset the index, now  first 
                                                # observation has index 0
data_cleaned.head()

# The data are ready to be used



# LET'S DEVELOP OUR AR(7) MODEL

# Dependent variable INDPRO
Y = data_cleaned['INDPRO']
Y

# The AR(7) likelihood

# DEFINING FUNCTIONS 
import numpy as np
import scipy 
def unconditional_ar_mean_variance(c, phis, sigma2): # 'c' = the constant term of the AR(7) model, 
                                                     # 'phis' = vector containing autoregressive coefficients of order 1 to 7
                                                     # 'sigma2' = the variance of the process.
    ## The length of phis is p
    p = len(phis)
    A = np.zeros((p, p))                             # 'A' = tridiagonal matrix in which the phis coefficients are on the 
                                                     # first row and the main diagonal contains 1.
    A[0, :] = phis
    A[1:, 0:(p-1)] = np.eye(p-1)
    ## Check for stationarity: checking whether all absolute values of the eigenvalues of A are less than 1. 
    # If this condition is met, the process is stationary and the stationary variable is set to True, otherwise to False.
    eigA = np.linalg.eig(A)
    if all(np.abs(eigA.eigenvalues)<1):
        stationary = True
    else:
        stationary = False
    # Create the vector b, a vector of zeros
    b = np.zeros((p, 1))
    b[0, 0] = c
    
    # Compute the mean using matrix algebra (Stationary mean)
    I = np.eye(p)
    mu = np.linalg.inv(I - A) @ b 
    
    # Solve the discrete Lyapunov equation
    Q = np.zeros((p, p))
    Q[0, 0] = sigma2
    #Sigma = np.linalg.solve(I - np.kron(A, A), Q.flatten()).reshape(7, 7)
    Sigma = scipy.linalg.solve_discrete_lyapunov(A, Q)
    
    return mu.ravel(), Sigma, stationary


from scipy.stats import norm
from scipy.stats import multivariate_normal
import numpy as np

# Create the lagged matrix: 
    # Empty matrix with n rows and max_lag columns
def lagged_matrix(Y, max_lag=7):
    n = len(Y)
    lagged_matrix = np.full((n, max_lag), np.nan)   

    # Fill each column with the appropriately lagged data:
    for lag in range(1, max_lag + 1):
        lagged_matrix[lag:, lag - 1] = Y[:-lag]
    return lagged_matrix


# CONDITIONAL log-likelihood:
def cond_loglikelihood_ar7(params, y):
    c = params[0] 
    phi = params[1:8]
    sigma2 = params[8]

    # Compute the unconditional mean and variance:
    mu, Sigma, stationary = unconditional_ar_mean_variance(c, phi, sigma2)

    # Check that at phis the process is stationary, return -Inf if it is not:
    if not(stationary):
        return -np.inf
    
    # The distribution of 
    # y_t|y_{t-1}, ..., y_{t-7} ~ N(c+\phi_{1}*y_{t-1}+...+\phi_{7}y_{t-7}, sigma2)

    # Create lagged matrix:
    X = lagged_matrix(y, 7)
            # Matrix containing the lagged values of y up to the seventh lag;
    yf = y[7:]
            # yf contains the original time series from the seventh observation onward,
    Xf = X[7:,:]
            # Xf will be the corresponding lagged matrix.
    
    # Compute the conditional log-likelihood:
    loglik = np.sum(norm.logpdf(yf, loc=(c + Xf@phi), scale=np.sqrt(sigma2)))
    return loglik


# UNCONDITIONAL log-likelihood:
def uncond_loglikelihood_ar7(params, y):
    # The unconditional loglikelihood is the unconditional "plus" the density of the
    # first p (7 in our case) observations
    cloglik = cond_loglikelihood_ar7(params, y)

    # Calculate initial
    # y_1, ..., y_7 ~ N(mu, sigma_y)
    c = params[0] 
    phi = params[1:8]
    sigma2 = params[8]

    # Compute the unconditional mean and variance:
    mu, Sigma, stationary = unconditional_ar_mean_variance(c, phi, sigma2)

    # Checking for stationarity:
    if not(stationary):
        return -np.inf
    
    mvn = multivariate_normal(mean=mu, cov=Sigma, allow_singular=True)
    uloglik = cloglik + mvn.logpdf(y[0:7])
    return uloglik
    

# OPTIMIZATION TO ESTIMATE THE PARAMETERS

# Define y:
y = Y

# Ordinary Least Squares:
X = lagged_matrix(y, 7)
yf = y[7:]
Xf = np.hstack((np.ones((len(yf),1)), X[7:,:]))

# Estimate the parameters and the variance: 
beta = np.linalg.solve(Xf.T@Xf, Xf.T@yf)
beta        # To see the estimates 
sigma2_hat = np.mean((yf - Xf@beta)**2)

# They are concatenated into a single vector:
params = np.hstack((beta, sigma2_hat))

# Negative value of the conditional log-likelihood:
def cobj(params, y):
    # Compute the value of the objective function
    value = -cond_loglikelihood_ar7(params, y)
    
    # Handle invalid values
    if np.isnan(value):
        #  If the value is invalid, return a large value to indicate an error
        return 1e12
    else:
        # Otherwise, return the computed value
        return value

# Minimize the conditional log-likelihood using the L-BFGS-B algorithm:
results1 = scipy.optimize.minimize(cobj, params, args = y, method='L-BFGS-B')
results1

# We can see that the values of result.x are equal to the OLS parameters

## Not the conditional

def uobj(params, y): 
    return - uncond_loglikelihood_ar7(params,y)

bounds_constant = tuple((-np.inf, np.inf) for _ in range(1))
bounds_phi = tuple((-1, 1) for _ in range(7))
bounds_sigma = tuple((0,np.inf) for _ in range(1))
bounds = bounds_constant + bounds_phi + bounds_sigma

## L-BFGS-B support bounds
results2 = scipy.optimize.minimize(uobj, results1.x, args = y, method='L-BFGS-B', bounds = bounds)
results2
