"""
A simple script to generate a simple scatter plot
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.odr as odr
from scipy.stats import linregress
from sklearn.linear_model import QuantileRegressor

def get_orthogonal_point(x, y, a, b):

    b_prime = y + x / a
    x_2 = (b_prime - b) / (a + 1/a)
    y_2 = a * x_2 + b

    return [x_2, y_2]


def test_ortho(func):
    point = (2, 2)
    a = -0.5
    b = 1.5
    x, y = func(point[0], point[1], a, b)
    print(a, b)
    

## Start with a simple scatter plot
np.random.seed(0)
bottom = 0
top = 100
plot_bottom = 0
plot_top = 100

x = np.arange(bottom, top, 1)
sigma = 3
y = np.random.normal(18 * np.log(x + 1), sigma)

plt.style.use('dark_background')
plt.figure(figsize = (5,5))
plt.scatter(x[plot_bottom:plot_top], y[plot_bottom:plot_top], alpha = 0.7, marker = "o")
plt.gca().set_aspect("equal") # Make sure the line actually looks orthogonal
plt.savefig("./content/scatter.png")
plt.close()


points_to_plot = [3, 25, 47, 80]


## Fit an orthogonal distance regression (ODR)
def f(B, x):
    """Linear function y = beta * x"""
    return B[0]*x

linear = odr.Model(f)
data = odr.Data(x, y)
myodr = odr.ODR(data, linear, beta0=[1])
test = myodr.run()
coeff = test.beta[0]

plt.style.use('dark_background')
plt.figure(figsize = (5,5))
# Add the scatter
plt.scatter(x[plot_bottom:plot_top], y[plot_bottom:plot_top], alpha = 0.7, marker = "o")
# Add the fitted line
plt.plot(x[plot_bottom:plot_top], coeff * x[plot_bottom:plot_top], alpha = 0.9, linestyle = "dotted", color = "coral")
# Add some orthogonal projections onto the line of best fit
# Stolen from here: https://www.tutorialspoint.com/how-do-you-create-line-segments-between-two-points-in-matplotlib
for point_id in points_to_plot:
    point_1 = [x[point_id], y[point_id]]
    point_2 = get_orthogonal_point(point_1[0], point_1[1], coeff, 0)
    x_values = [point_1[0], point_2[0]]
    y_values = [point_1[1], point_2[1]]
    plt.plot(x_values, y_values, linestyle = "-", alpha = 0.7, color = 'r')
plt.gca().set_aspect("equal") # Make sure the line actually looks orthogonal
plt.savefig("./content/scatter_orthogonal.png")
plt.close()

## Fit a MAE linear regression, i.e. a median Quantile Regression
quantile = 0.5
qr = QuantileRegressor(quantile = quantile, alpha = 0, solver = "highs")
y_pred = qr.fit(x.reshape(-1, 1), y).predict(x.reshape(-1,1))

plt.style.use('dark_background')
plt.figure(figsize = (5,5))
# Add the scatter
plt.scatter(x[plot_bottom:plot_top], y[plot_bottom:plot_top], alpha = 0.7, marker = "o")
# Add the fitted line
plt.plot(x[plot_bottom:plot_top], y_pred[plot_bottom:plot_top], alpha = 0.9, linestyle = "dashed", color = "palegreen")
# Add some orthogonal projections onto the line of best fit
# Stolen from here: https://www.tutorialspoint.com/how-do-you-create-line-segments-between-two-points-in-matplotlib
for point_id in points_to_plot:
    point_1 = [x[point_id], y[point_id]]
    point_2 = [x[point_id], y_pred[point_id]]
    x_values = [point_1[0], point_2[0]]
    y_values = [point_1[1], point_2[1]]
    plt.plot(x_values, y_values, linestyle = "-", alpha = 0.7, color = 'r')
plt.gca().set_aspect("equal") # Make sure the line actually looks orthogonal
plt.savefig("./content/scatter_parallel.png")
plt.close()


## Fit a regular OLS
slope, intercept, r, p, se = linregress(x, y)

plt.style.use('dark_background')
plt.figure(figsize = (5,5))
# Add the scatter
plt.scatter(x[plot_bottom:plot_top], y[plot_bottom:plot_top], alpha = 0.7, marker = "o")
# Add the fitted line
plt.plot(x[plot_bottom:plot_top], slope * x[plot_bottom:plot_top] + intercept, alpha = 0.9, linestyle = "dashdot", color = "darkcyan")
for point_id in points_to_plot:
    
    point_1 = [x[point_id], y[point_id]]
    point_2 = [x[point_id], x[point_id] * slope + intercept]
    
    distance = np.abs(point_1[1] - point_2[1])

    if point_1[0] * slope + intercept > point_1[1]:
        distance = - distance

    point_3 = [x[point_id] - distance, point_2[1]]
    point_4 = [point_3[0], y[point_id]]
    
    x_values = [point_1[0], point_2[0], point_3[0], point_4[0], point_1[0]]
    y_values = [point_1[1], point_2[1], point_3[1], point_4[1], point_1[1]]
    
    plt.plot(x_values, y_values, linestyle = "-", alpha = 0.7, color = 'r')
    
plt.gca().set_aspect("equal") # Make sure the line actually looks orthogonal
plt.savefig("./content/scatter_squares.png")
plt.close()


## Plot all lines together

plt.style.use('dark_background')
plt.figure(figsize = (5,5))

# Add the scatter
plt.scatter(x[plot_bottom:plot_top], y[plot_bottom:plot_top], alpha = 0.7, marker = "o")
# Add the fitted ODR line
plt.plot(x[plot_bottom:plot_top], coeff * x[plot_bottom:plot_top], alpha = 0.9, linestyle = "dotted", color = "coral")
# Add the fitted MAE line
plt.plot(x[plot_bottom:plot_top], y_pred[plot_bottom:plot_top], alpha = 0.9, linestyle = "dashed", color = "palegreen")
# Add the fitted OLS line
plt.plot(x[plot_bottom:plot_top], slope * x[plot_bottom:plot_top] + intercept, alpha = 0.9, linestyle = "dashdot", color = "darkcyan")

plt.gca().set_aspect("equal") # Make sure the line actually looks orthogonal
plt.savefig("./content/scatter_all_lines.png")
plt.close()


## Plots for covariance stationarity

time = np.arange(0, 1000, 1)
x = np.array([1000])
y = np.array([1000])

for t in time:
    to_append_x = x[t] + np.random.normal(0,1)
    x = np.append(x, to_append_x)

    to_append_y = 0.5 * x[t] + np.random.normal(0,0.3)
    y = np.append(y, to_append_y)

x_plots = []

x_fe = x - np.sum(x)
x_fd = np.diff(x)
x_div = y / x
x_sqrt = np.sqrt(x)
x_cube = np.power(x, 3)
x_log = np.log(x)
x_change = np.diff(x) / x[1:]
x_log_diff = np.diff(x_log)
    
plt.style.use('dark_background')
fig, axs = plt.subplots(3, 3, figsize = (9, 9), dpi = 900) # Somehow even at 900 dpi the text comes out a bit blurry

axs[0, 0].plot(time, x[1:])
axs[0, 0].title.set_text("X over time")
axs[0, 0].get_xaxis().set_visible(False)

axs[0, 1].plot(time, x_fe[1:])
axs[0, 1].title.set_text("X fixed effects over time")
axs[0, 1].get_xaxis().set_visible(False)

axs[0, 2].plot(time, x_fd)
axs[0, 2].title.set_text("X first difference over time")
axs[0, 2].get_xaxis().set_visible(False)

axs[1, 0].plot(time, x_div[1:])
axs[1, 0].title.set_text("X divided by similar variable over time")
axs[1, 0].get_xaxis().set_visible(False)

axs[1, 1].plot(time, x_sqrt[1:])
axs[1, 1].title.set_text("sqrt(X) over time")
axs[1, 1].get_xaxis().set_visible(False)

axs[1, 2].plot(time, x_cube[1:])
axs[1, 2].title.set_text("X^3 over time")
axs[1, 2].get_xaxis().set_visible(False)

axs[2, 0].plot(time, x_log[1:])
axs[2, 0].title.set_text("log(X) over time")
axs[2, 0].get_xaxis().set_visible(False)

axs[2, 1].plot(time, x_change)
axs[2, 1].title.set_text("Change in X over time")
axs[2, 1].get_xaxis().set_visible(False)

axs[2, 2].plot(time, x_log_diff)
axs[2, 2].title.set_text("log(X) difference over time")
axs[2, 2].get_xaxis().set_visible(False)

plt.savefig("./content/transformations.png")
plt.close()
