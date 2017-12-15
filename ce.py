from sklearn.datasets import load_svmlight_file
import numpy as np
import matplotlib.pyplot as plt
import random

# Function
## Loss
def loss(x, y, omg):
    # Input - X matrix, Y matrix, Omega List
    # Output - Loss, as a float
    s = sum(np.maximum(0, 1 - y * np.dot(x, omg)))
    return (0.9*s) / y.shape[0]+0.5 * np.dot(omg, omg)

## Gradient
def gra(x, y, omg):
    # Input - X matrix, Y matrix, Omega List
    # Output - Gradient, as a list
    ## Calculate: Gradient for all Omega
    yn = y * np.maximum(0, 1 - y * np.dot(x, omg) / np.abs(1 - y * np.dot(x, omg)))
    return omg+(-np.dot(x.T, y) / y.shape[0])*0.9


# Parameters, overall
## Learning rate
learning_Rate = 0.01

## Iteration count
iter_num = 1000

## SGP Batch size
batch_size = 325

# Parameters, definition of specific method
## NAG
NAG_v = np.zeros(124)
NAG_alpha = 0.9

## RMSProp
RMS_rho = 0.95
RMS_ep = 0.0000001

## AdaDelta
Ada_rho = 0.95
Ada_ep = 0.0000001

## Adam
Adam_rho1 = 0.9
Adam_rho2 = 0.999
Adam_ep = 0.0000001

# Data, Load
## Train
X_t, y_t = load_svmlight_file("a9a")
X_t = X_t.toarray()
X_t = np.column_stack((X_t, np.ones(X_t.shape[0])))  # Add omega[0] column
y_t = y_t + np.ones(y_t.size)
y_t = y_t / 2  # Trans y to 1 and 0 for calculation

## Test
X_v, y_v = load_svmlight_file("a9a.t")
X_v = X_v.toarray()
X_v = np.column_stack((X_v, np.zeros(X_v.shape[0])))  # Stupid data
X_v = np.column_stack((X_v, np.ones(X_v.shape[0])))
y_v = y_v + np.ones(y_v.size)
y_v = y_v / 2

# Data, Record
## Loss data
loss_NAG = []
loss_RMSProp = []
loss_AdaDelta = []
loss_Adam = []

## Omega
omega_NAG = np.zeros(124)
omega_RMSProp = np.zeros(124)
omega_AdaDelta = np.zeros(124)
omega_Adam = np.zeros(124)

## Gradient
gra_NAG = np.zeros(124)
gra_RMSProp = np.zeros(124)
gra_AdaDelta = np.zeros(124)
gra_Adam = np.zeros(124)

# Data, additional definition of specific method
## NAG
NAG_mom = np.zeros(124)

## RMSProp
RMS_t = 0
RMS_g = 0
RMS_d = np.ones(124) / batch_size

## AdaDelta
Ada_t = 0
Ada_g = 0
Ada_d = np.ones(124) / batch_size

## Adam
Adam_v = 0
Adam_d = np.ones(124)
Adam_m = np.zeros(124)

### Main Program ###
# 1 - Do the calculation
for iter in range(iter_num):
    # Picking Random Sample
    X_tr = np.array([])
    y_tr = []
    for n in range(batch_size):
        pick = random.randint(0, y_t.shape[0] - 1)
        X_tr = np.append(X_tr, X_t[pick])
        y_tr = np.append(y_tr, y_t[pick])
    X_tr = X_tr.reshape(-1, 124)

    # Calculate gradient
    gra_NAG = gra(X_tr, y_tr, omega_NAG + NAG_alpha * NAG_mom)
    gra_RMSProp = gra(X_tr, y_tr, omega_RMSProp)
    gra_AdaDelta = gra(X_tr, y_tr, omega_AdaDelta)
    gra_Adam = gra(X_tr, y_tr, omega_Adam)

    # Do the omega calculation for each method
    ## NAG
    NAG_mom = -learning_Rate * gra_NAG + NAG_alpha * NAG_mom

    ## RMSProp
    RMS_t = (1 - RMS_rho) * np.dot(RMS_d, RMS_d) + RMS_rho * RMS_t
    RMS_g = (1 - RMS_rho) * np.dot(gra_RMSProp, gra_RMSProp) + RMS_rho * RMS_g
    RMS_d = -gra_RMSProp * learning_Rate / np.sqrt(RMS_g + RMS_ep)

    ## AdaDelta
    Ada_t = (1 - Ada_rho) * np.dot(Ada_d, Ada_d) + Ada_rho * Ada_t
    Ada_g = (
        1 - Ada_rho) * np.dot(gra_AdaDelta, gra_AdaDelta) + Ada_rho * Ada_g
    Ada_d = -gra_AdaDelta * (np.sqrt(Ada_t + Ada_ep) / np.sqrt(Ada_g + Ada_ep))

    ## Adam
    Adam_m = (1 - Adam_rho1) * gra_Adam + Adam_rho1 * Adam_m
    Adam_v = (1 - Adam_rho2) * np.dot(gra_Adam, gra_Adam) + Adam_rho2 * Adam_v
    Adam_d = -learning_Rate / (
        Adam_ep + np.sqrt(Adam_v / (1 - np.power(Adam_rho2, iter + 1)))) * (
            Adam_m / (1 - np.power(Adam_rho1, iter + 1)))

    # Do the omega updating
    omega_NAG = omega_NAG + NAG_mom
    omega_RMSProp = omega_RMSProp + RMS_d
    omega_AdaDelta = omega_AdaDelta + Ada_d
    omega_Adam = omega_Adam + Adam_d

    # Calculate loss
    loss_NAG.append(loss(X_v, y_v, omega_NAG))
    loss_RMSProp.append(loss(X_v, y_v, omega_RMSProp))
    loss_AdaDelta.append(loss(X_v, y_v, omega_AdaDelta))
    loss_Adam.append(loss(X_v, y_v, omega_Adam))

    # Print the loss
    print("Iter%d: NAG - %.3f; RMSProp - %.3f; AdaDelta - %.3f; Adam - %.3f" %
          (iter, loss_NAG[iter], loss_RMSProp[iter], loss_AdaDelta[iter],
           loss_Adam[iter]))

# 2 - Generating the graph
plt.plot(loss_NAG, label='NAG')
plt.plot(loss_RMSProp, label='RMSProp')
plt.plot(loss_AdaDelta, label='AdaDelta')
plt.plot(loss_Adam, label='Adam')

plt.xlabel("Iter")
plt.ylabel("Loss")

plt.legend()
plt.show()