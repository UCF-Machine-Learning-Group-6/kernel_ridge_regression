import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.kernel_ridge import KernelRidge
import random
import warnings as ws


# Ignore all Underflow Warnings : Value close to Zero - No Accuracy Errors
np.seterr(under='ignore')
sp.seterr(under='ignore')
ws.simplefilter('ignore')


def kfold(dataset, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=random.randint(1, 10))
    return kf

# Load the CSV file
dataset = pd.read_csv('wineQualityReds.csv')
percent = 0.80
dataset_cv = dataset[ : int(percent * dataset.shape[0])]
dataset_main = dataset[int(percent * dataset.shape[0]) : ]

# Get Cross Validation K-Fold
folds = kfold(dataset, 5)

# Get train and test set splits for Cross Validation and Main Testing
for train_index, test_index in folds.split(dataset_cv):
    x_train_cv = np.asarray([[row['fixed.acidity'], row['volatile.acidity'], row['citric.acid'], 
                              row['residual.sugar'], row['chlorides'], row['free.sulfur.dioxide'], 
                              row['total.sulfur.dioxide'], row['density'], row['pH'], row['sulphates'], 
                              row['alcohol']] for index, row in dataset_cv.iloc[train_index].iterrows()])
    x_test_cv = np.asarray([[row['fixed.acidity'], row['volatile.acidity'], row['citric.acid'], 
                             row['residual.sugar'], row['chlorides'], row['free.sulfur.dioxide'], 
                             row['total.sulfur.dioxide'], row['density'], row['pH'], row['sulphates'], 
                             row['alcohol']] for index, row in dataset_cv.iloc[train_index].iterrows()])
    y_train_cv = np.asarray([[row['quality']] for index, row in dataset_cv.iloc[train_index].iterrows()])
    y_test_cv = np.asarray([[row['quality']] for index, row in dataset_cv.iloc[train_index].iterrows()])

x_test_main = np.asarray([[row['fixed.acidity'], row['volatile.acidity'], row['citric.acid'], 
                           row['residual.sugar'], row['chlorides'], row['free.sulfur.dioxide'], 
                           row['total.sulfur.dioxide'], row['density'], row['pH'], row['sulphates'], 
                           row['alcohol']] for index, row in dataset_main.iterrows()])
y_test_main = np.asarray([[row['quality']] for index, row in dataset_main.iterrows()])


#Kernel Hyper Parameters
#Kernel    alpha    degree    gamma    coef0
#         1e3:1e-3  3/poly   rbf/poly   poly 
#linear    above    none      none      none
#poly      above    poly      poly      poly
#rbf/gauss above    none      rbf       none

# Tuning for Hyper Parameters
tx = time.time()
compare_lin = []
compare_poly = []
compare_rbf =[]
for k in ['linear', 'poly', 'rbf']:
    for a in [100.0, 50.0, 10.0, 5.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]:
        for d in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
            for g in [1.0, 2.0, 3.0, 4.0, 5.0]:
                for c in [1.0, 2.0, 3.0, 4.0, 5.0]:
                    krr = KernelRidge(kernel=k, alpha=a, degree=d, gamma=g, coef0=c)
                    t = time.time()
                    krr.fit(x_train_cv, y_train_cv)
                    krr_fit = time.time() - t
                    #print('Time Taken to Fit the Model : ' + str(krr_fit) + 'Secs')
                    t = time.time()
                    y_krr = krr.predict(x_test_cv)
                    krr_predict = time.time() - t
                    #print('Time Taken to Predict using the Fitted Model : ' + str(krr_predict) + 'Secs')
                    acc_score = r2_score(y_test_cv, y_krr) * 100
                    #print('Accuracy Score of the Model : ' + str(acc_score))
                    if k == 'linear':
                        compare_lin.append([a, krr_fit, krr_predict, acc_score])
                    elif k == 'poly':
                        compare_poly.append([a, d, (1/(2*(g**2))), c, krr_fit, krr_predict, acc_score])
                    elif k == 'rbf':
                        compare_rbf.append([a, (1/(2*(g**2))), krr_fit, krr_predict, acc_score])
                    else:
                        pass
print('\nTime Taken to Tune Hyper Parameter Values = ' + str((time.time()-tx)/60) + ' Mins')


# Saving All Values obtained from Hyper Parameter Tuning
compare_lin = np.array(compare_lin)
df = pd.DataFrame(compare_lin)
df.to_csv("linear_results_sklearn.csv")
compare_poly = np.array(compare_poly)
df = pd.DataFrame(compare_poly)
df.to_csv("polynomial_results_sklearn.csv")
compare_rbf = np.array(compare_rbf)
df = pd.DataFrame(compare_rbf)
df.to_csv("gaussian_results_sklearn.csv")


# Fetching Hyperparameters based on Max Accuracy Score and Return Kernel
# Linear
lin_max = np.where(compare_lin[:, 3] == np.amax(compare_lin[:, 3]))
lin_alpha = compare_lin[lin_max[0]][0][0]
# Polynomial
poly_max = np.where(compare_poly[:, 6] == np.amax(compare_poly[:, 6]))
poly_alpha = compare_poly[poly_max[0]][0][0]
poly_deg = compare_poly[poly_max[0]][0][1]
poly_gamma = compare_poly[poly_max[0]][0][2]
poly_coef = compare_poly[poly_max[0]][0][3]
# RBF/Gaussian
rbf_max = np.where(compare_rbf[:, 4] == np.amax(compare_rbf[:, 4]))
rbf_alpha = compare_rbf[rbf_max[0]][0][0]
rbf_gamma = compare_rbf[rbf_max[0]][0][1]


# Generate Prediction for selected Hyperparameters
#Linear
print('\nLinear Kernel with Hyperparameters : Alpha = ' + str(lin_alpha))
krr = KernelRidge(kernel='linear', alpha=lin_alpha)
t = time.time()
krr.fit(x_test_main, y_test_main)
krr_fit = time.time() - t
print('Time Taken to Fit the Model : ' + str(krr_fit) + ' Secs')
t = time.time()
y_krr = krr.predict(x_test_main)
krr_predict = time.time() - t
print('Time Taken to Predict using the Fitted Model : ' + str(krr_predict) + ' Secs')
acc_score = r2_score(y_test_main, y_krr) * 100
print('Accuracy Score of the Model : ' + str(acc_score))
plt.figure('Linear Kernel')
plt.plot([x for x in range(0, len(y_test_main))], y_test_main, 'b.')
plt.plot([x for x in range(0, len(y_test_main))], y_krr, 'r-')
plt.title('Linear Kernel - Y_test vs. Y_pred')
plt.legend(['Y_test', 'Y_pred'])
#Polynomial
print('\nPolynomial Kernel with Hyperparameters : Alpha = ' + str(poly_alpha) + 
      ', Degree = ' + str(poly_deg) + ', Gamma = ' + str(poly_gamma) + 
      ', Coeff = ' + str(poly_coef))
krr = KernelRidge(kernel='poly', alpha=poly_alpha, degree=poly_deg, gamma=poly_gamma, coef0=poly_coef)
t = time.time()
krr.fit(x_test_main, y_test_main)
krr_fit = time.time() - t
print('Time Taken to Fit the Model : ' + str(krr_fit) + ' Secs')
t = time.time()
y_krr = krr.predict(x_test_main)
krr_predict = time.time() - t
print('Time Taken to Predict using the Fitted Model : ' + str(krr_predict) + ' Secs')
acc_score = r2_score(y_test_main, y_krr) * 100
print('Accuracy Score of the Model : ' + str(acc_score))
plt.figure('Polynomial Kernel')
plt.plot([x for x in range(0, len(y_test_main))], y_test_main, 'b.')
plt.plot([x for x in range(0, len(y_test_main))], y_krr, 'r-')
plt.title('Polynomial Kernel - Y_test vs. Y_pred')
plt.legend(['Y_test', 'Y_pred'])
#Gaussian
print('\nGaussian Kernel with Hyperparameters : Alpha = ' + str(rbf_alpha) + 
      ', Gamma = 1/(2*Sigma^2) = ' + str(rbf_gamma))
krr = KernelRidge(kernel='rbf', alpha=rbf_alpha, gamma=rbf_gamma)
t = time.time()
krr.fit(x_test_main, y_test_main)
krr_fit = time.time() - t
print('Time Taken to Fit the Model : ' + str(krr_fit) + ' Secs')
t = time.time()
y_krr = krr.predict(x_test_main)
krr_predict = time.time() - t
print('Time Taken to Predict using the Fitted Model : ' + str(krr_predict) + ' Secs')
acc_score = r2_score(y_test_main, y_krr) * 100
print('Accuracy Score of the Model : ' + str(acc_score))
plt.figure('Gaussian Kernel')
plt.plot([x for x in range(0, len(y_test_main))], y_test_main, 'b.')
plt.plot([x for x in range(0, len(y_test_main))], y_krr, 'r-')
plt.title('Gaussian Kernel - Y_test vs. Y_pred')
plt.legend(['Y_test', 'Y_pred'])


# Reset Underflow Warnings : Value close to Zero - No Accuracy Errors
np.seterr(under='warn')
sp.seterr(under='warn')
ws.resetwarnings()


# End of File