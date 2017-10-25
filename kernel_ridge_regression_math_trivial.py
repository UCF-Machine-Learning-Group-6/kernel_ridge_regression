import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error


def load_csv(filename):
    dataset = pd.read_csv(filename)
    return dataset

def linear(X, Xprime):
    return np.dot(np.transpose(X), Xprime)

def polynomial(X, Xprime, gamma, r, M):
    return (np.dot(np.dot(gamma, np.transpose(X)), Xprime) + r) ** M

def gauss(X, Xprime, sigma):
    X = X - Xprime
    return np.exp(-(np.linalg.norm(X) ** 2) / (2 * (sigma) ** 2))

def regr_lin(params):
    x_train, y_train = params[0], params[1]
    K = np.empty([len(x_train), len(x_train)])
    lambda_identity = np.identity(len(x_train))
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            K[i, j] = linear(x_train[i], x_train[j])
    theInverse = np.linalg.inv(K + lambda_identity)
    w = np.dot(theInverse, y_train)
    return w

def regr_poly(params):
    x_train, y_train, gamma, r, M = params[0], params[1], params[2], params[3], params[4]
    K = np.empty([len(x_train), len(x_train)])
    lambda_identity = np.identity(len(x_train))
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            K[i, j] = polynomial(x_train[i], x_train[j], gamma, r, M)
    theInverse = np.linalg.inv(K + lambda_identity)
    w = np.dot(theInverse, y_train)
    return w

def regr_gauss(params):
    x_train, y_train, sigma = params[0], params[1], params[2]
    K = np.empty([len(x_train), len(x_train)])
    lambda_identity = np.identity(len(x_train))
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            K[i, j] = gauss(x_train[i], x_train[j], sigma)
    theInverse = np.linalg.inv(K + lambda_identity)
    w = np.dot(theInverse, y_train)
    return w

def kfold(dataset, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=random.randint(1, 10))
    return kf

def train_test_split_cv(dataset):
    for train_index, test_index in folds.split(dataset):
        x_train = np.asarray([[row['density']] for index, row in dataset.iloc[train_index].iterrows()])        
        x_test = np.asarray([[row['density']] for index, row in dataset.iloc[test_index].iterrows()])
        y_train = np.asarray([[row['alcohol']] for index, row in dataset.iloc[train_index].iterrows()])
        y_test = np.asarray([[row['alcohol']] for index, row in dataset.iloc[test_index].iterrows()])
    return x_train, x_test, y_train, y_test

def test_sets_main(dataset):
    x_test = np.asarray([[row['density']] for index, row in dataset.iterrows()])
    y_test = np.asarray([[row['alcohol']] for index, row in dataset.iterrows()])
    return x_test, y_test

def pred_lin(params):
    
    dataset = params[0]
    set_type = params[1]
    
    if set_type == 'cv':
        x_train, x_test, y_train, y_test = train_test_split_cv(dataset)
        w = regr_lin([x_train, y_train])
        predicted = []
        for i in range(len(x_test)):
            k = np.empty(len(x_train))
            for j in range(len(x_train)):
                k[i] = linear(x_train[j], x_test[i])
            prediction = np.dot(k, w)
            predicted.append(prediction)
        predicted = np.asarray(predicted)
        mse = mean_squared_error(y_test, predicted)
        acc_score = r2_score(y_test, predicted)
        acc_score = (acc_score/(np.floor(np.log10(acc_score)) + 1)) * 100
    
    elif set_type == 'main':
        x_test, y_test = test_sets_main(dataset)
        w = regr_lin([x_test, y_test])
        predicted = []
        for i in range(len(x_test)):
            k = np.empty(len(x_test))
            for j in range(len(x_test)):
                k[i] = linear(x_test[j], x_test[i])
            prediction = np.dot(k, w)
            predicted.append(prediction)
        predicted = np.asarray(predicted)
        mse = mean_squared_error(y_test, predicted)
        acc_score = r2_score(y_test, predicted)
        acc_score = (acc_score/(np.floor(np.log10(acc_score)) + 1)) * 100
    
    print ('Lin :' + str(mse))
    return acc_score, predicted

def pred_poly(params):
    
    dataset, gamma, r, M = params[0], params[2], params[3], params[4]
    set_type = params[1]
    
    if set_type == 'cv':
        x_train, x_test, y_train, y_test = train_test_split_cv(dataset)
        w = regr_poly([x_train, y_train, gamma, r, M])
        predicted = []
        for i in range(len(x_test)):
            k = np.empty(len(x_train))
            for j in range(len(x_train)):
                k[i] = polynomial(x_train[j], x_test[i], gamma, r, M)
            prediction = np.dot(np.transpose(k), w)
            predicted.append(prediction)
        predicted = np.asarray(predicted)
        mse = mean_squared_error(y_test, predicted)
        acc_score = r2_score(y_test, predicted)
        acc_score = (acc_score/(np.floor(np.log10(acc_score)) + 1)) * 100
    
    elif set_type == 'main':
        x_test, y_test = test_sets_main(dataset)
        w = regr_poly([x_test, y_test, gamma, r, M])
        predicted = []
        for i in range(len(x_test)):
            k = np.empty(len(x_test))
            for j in range(len(x_test)):
                k[i] = polynomial(x_test[j], x_test[i], gamma, r, M)
            prediction = np.dot(np.transpose(k), w)
            predicted.append(prediction)
        predicted = np.asarray(predicted)
        mse = mean_squared_error(y_test, predicted)
        acc_score = r2_score(y_test, predicted)
        acc_score = (acc_score/(np.floor(np.log10(acc_score)) + 1)) * 100
    
    print ('Poly :' + str(mse))
    return acc_score, predicted

def pred_gauss(params):
    
    dataset, sigma = params[0], params[2]
    set_type = params[1]
    
    if set_type == 'cv':
        x_train, x_test, y_train, y_test = train_test_split_cv(dataset)
        w = regr_gauss([x_train, y_train, sigma])
        predicted = []
        for i in range(len(x_test)):
            k = np.empty(len(x_train))
            for j in range(len(x_train)):
                k[i] = gauss(x_train[j], x_test[i], sigma)
            prediction = np.dot(k, w)
            predicted.append(prediction)
        predicted = np.asarray(predicted)
        mse = mean_squared_error(y_test, predicted)
        acc_score = r2_score(y_test, predicted)
        acc_score = (acc_score/(np.floor(np.log10(acc_score)) + 1)) * 100
    
    elif set_type == 'main':
        x_test, y_test = test_sets_main(dataset)
        w = regr_gauss([x_test, y_test, sigma])
        predicted = []
        for i in range(len(x_test)):
            k = np.empty(len(x_test))
            for j in range(len(x_test)):
                k[i] = gauss(x_test[j], x_test[i], sigma)
            prediction = np.dot(k, w)
            predicted.append(prediction)
        predicted = np.asarray(predicted)
        mse = mean_squared_error(y_test, predicted)
        acc_score = r2_score(y_test, predicted)
        acc_score = (acc_score/(np.floor(np.log10(acc_score)) + 1)) * 100
    
    print ('Gauss :' + str(mse))
    return acc_score, predicted


if __name__ == "__main__":
    
    filename = "wineQualityReds.csv"
    
    dataset = load_csv(filename)
    percent = 0.80
    dataset_cv = dataset[ : int(percent * dataset.shape[0])]
    dataset_main = dataset[int(percent * dataset.shape[0]) : ]
    
    folds = kfold(dataset, 5)
    
    # Tuning for Hyper Parameters
    tx = time.time()
    # No Tuning for Linear
    acc_lin, _ = pred_lin([dataset_cv, 'cv'])
    # Tuning for Polynomial
    compare_poly = []
    for gamma in [1.0, 2.0, 3.0, 4.0, 5.0]:
        for r in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            for M in [2.0, 3.0, 4.0]:
                acc_poly, _ = pred_poly([dataset_cv, 'cv', (1/(2*(gamma**2))), r, M])
                compare_poly.append([gamma, r, M, acc_poly])
    # Tuning for Gaussian
    compare_gauss = []
    for sigma in [1.0, 2.0, 3.0, 4.0, 5.0]:
        acc_gauss, _ = pred_gauss([dataset_cv, 'cv', sigma])
        compare_gauss.append([sigma, acc_gauss])
    print('\nTime Taken to Tune Hyper Parameter Values = ' + str((time.time()-tx)/60) + ' Mins')
    
    # Saving All Values obtained from Hyper Parameter Tuning
    compare_poly = np.array(compare_poly)
    df = pd.DataFrame(compare_poly)
    df.to_csv("polynomial_results_math.csv")
    compare_gauss = np.array(compare_gauss)
    df = pd.DataFrame(compare_gauss)
    df.to_csv("gaussian_results_math.csv")
    
    # Fetching Hyperparameters based on Max Accuracy Score
    # Polynomial
    poly_max = np.where(compare_poly[:, 3] == np.amax(compare_poly[:, 3]))
    poly_gamma = compare_poly[poly_max[0]][0][0]
    poly_r = compare_poly[poly_max[0]][0][1]
    poly_M = compare_poly[poly_max[0]][0][2]
    # Gaussian
    gauss_max = np.where(compare_gauss[:, 1] == np.amax(compare_gauss[:, 1]))
    gauss_sigma = compare_gauss[gauss_max[0]][0][0]
    
    # Get Test Sets
    x_test, y_test = test_sets_main(dataset_main)
    
    # Generate Prediction for selected Hyperparameters
    # Linear
    print('\nLinear Kernel without Hyperparameters : ')
    t = time.time()
    acc_lin, preds_lin = pred_lin([dataset_main, 'main'])
    lin_t = time.time() - t
    print('Time Taken to Fit the Model and Predict : ' + str(lin_t) + ' Secs')
    print('Accuracy Score of the Model : ' + str(acc_lin))
    plt.figure('Linear Kernel')
    plt.plot([x for x in range(0, len(y_test))], y_test, 'b.')
    plt.plot([x for x in range(0, len(y_test))], preds_lin, 'r-')
    plt.title('Linear Kernel - Y_test vs. Y_pred')
    plt.legend(['Y_test', 'Y_pred'])
    # Polynomial
    print('\nPolynomial Kernel with Hyperparameters : Gamma = ' + str(poly_gamma) + 
          ', Coeff = ' + str(poly_r) + ', Degree = ' + str(poly_M))
    t = time.time()
    acc_poly, preds_poly = pred_poly([dataset_main, 'main', poly_gamma, poly_r, poly_M])
    poly_t = time.time() - t
    print('Time Taken to Fit the Model and Predict : ' + str(poly_t) + ' Secs')
    print('Accuracy Score of the Model : ' + str(acc_poly))
    plt.figure('Polynomial Kernel')
    plt.plot([x for x in range(0, len(y_test))], y_test, 'b.')
    plt.plot([x for x in range(0, len(y_test))], preds_poly, 'r-')
    plt.title('Polynomial Kernel - Y_test vs. Y_pred')
    plt.legend(['Y_test', 'Y_pred'])
    # Gaussian
    print('\nGaussian Kernel with Hyperparameters : Sigma = ' + str(gauss_sigma))
    t = time.time()
    acc_gauss, preds_gauss = pred_gauss([dataset_main, 'main', gauss_sigma])
    gauss_t = time.time() - t
    print('Time Taken to Fit the Model and Predict : ' + str(gauss_t) + ' Secs')
    print('Accuracy Score of the Model : ' + str(acc_gauss))
    plt.figure('Gaussian Kernel')
    plt.plot([x for x in range(0, len(y_test))], y_test, 'b.')
    plt.plot([x for x in range(0, len(y_test))], preds_gauss, 'r-')
    plt.title('Gaussian Kernel - Y_test vs. Y_pred')
    plt.legend(['Y_test', 'Y_pred'])


# End of File