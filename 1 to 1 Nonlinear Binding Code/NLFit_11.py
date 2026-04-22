# -*- coding: utf-8 -*-
"""
Nonlinear fitting program for 1:1 protonation of Azo dyes
based on equation located in the SI of
     https://pubs.acs.org/doi/full/10.1021/acs.joc.3c00423
This equation, nl_11_bindfit, assumes that one of the interacting species
     is silent (ie doesn't absorb')
@author: goose
"""

import numpy as np
import math
import matplotlib.pyplot as  plt
from   scipy.optimize import curve_fit


# Defining Equations
def read_rawdat(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
#   skiprows=1 means keep header data (this makes it easier for the user
#   to look at input files later!)
    host, h0 = data[0:, 0], data[0, 0]
    guest, g0 = data[0:, 1], data[0, 1]
    absorb, abs0 = data[0:, 2], data[0, 2]
    return host, h0, guest, g0, absorb, abs0


def nl_11_bindfit(r, ehg, Ka, eg, g0, m):
    dA = (ehg-eg) * ((1/(2*Ka))*((Ka*g0*r*m+Ka*g0 + 1)
        - np.sqrt(((Ka*r*m*g0+Ka*g0 +1)**2)-4*r*m*(Ka**2)*(g0**2))))
    return dA


def iterative_fit(molerat, dA, nl_11_bindfit, initial_params, bounds,
                  max_iter=100, tolerance=1e-20):
    params = initial_params
    for i in range(max_iter):
        popt, pcov = curve_fit(nl_11_bindfit, molerat, dA, p0=params, bounds=bounds)
        if np.allclose(params, popt, rtol=tolerance):
            break
        params = popt
    return popt, pcov

# Defining neccessary data from input file
print('Importing Data...')
file = 'OMePAItBu TosOH titr 12 14 2023 T1.csv'  # Name of file
host, h0, guest, g0, absorb, abs0 = read_rawdat(file)

# Doing maths
molerat = host/guest      # molar ratio of [TosOH]/[Azo], aptly named mole rat
dA = absorb-abs0

# Initial parameters
initial_params = [0, 60000, 11500, 3.28042E-05, 1]
# ^ Guess for ehg, Ka, eg, g0, and m
# v bounds for fitting function
bounds = [0, 0, 11500, 3.25e-5, 0.98], [np.inf, np.inf, 11700, 3.3e-5, 1.02]
# Note: some lenience is given to eg and g to acct for experimental error

print('Fitting Curve...')
popt, pcov = iterative_fit(molerat, dA, nl_11_bindfit, initial_params, bounds)

Kafit, ehgfit = popt[1], popt[0]

# Creating Data and fit plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
ax1.scatter(molerat, dA, color='black', label='Data (392.62  nm)')
ax1.plot(molerat, nl_11_bindfit(molerat, dA, *popt[1:]), '--',
         color='orangered', label='Fitted Function')
ax1.set_ylabel("Δ Absorbance")
ax1.tick_params(which='both', direction='in')
ax1.set_xticks([])
ax1.legend(loc="best", frameon=False, facecolor='None')

# Creating residual plot
difference = nl_11_bindfit(molerat, dA, *popt[1:])-dA
ax2.plot(molerat, difference, 'o', color='orangered')
ax2.axhline(y=0, color='lightgray', linestyle="--")
ax2.tick_params(which='both', direction='in')
ax2.set_xlabel('[TosOH]/[OMe-PAI-tBu]')
ax2.set_ylabel("Residual")
fig.subplots_adjust(hspace=0)

print('Saving Figure...')  # Will save to the directory of the input file
plt.savefig("OMePAItBu TosOH titr 12 14 2024 T1 - refit 2.svg", format='svg')

# Printing out fit results and calculating pKa value!
pKa = math.log(Kafit, 10) + 8.5  # <-----change 8.5 to the pKa of your acid

print("The Ka of this data set is:", Kafit)
print("The ehg for this data set is:", ehgfit)
print("The covariance of this data set is:", np.linalg.cond(pcov))
print("The pKa for this data set (assuming you used TosOH) is:", pKa)
print("Done! :)")

plt.show()
