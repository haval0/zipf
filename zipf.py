# Zipf's law: https://en.wikipedia.org/wiki/Zipf's_law
# log-log diagram: https://en.wikipedia.org/wiki/Log%E2%80%93log_plot
# sklearn LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

# The goal is to evaluate whether the dataset of the most common Chinese words follows Zipf's law.

# numpy is used to import data and manipulate arrays
import numpy as np

# sklearn is used to perform linear regression
from sklearn.linear_model import LinearRegression

# matplotlib is used to plot the diagram
from matplotlib import pyplot as plt

# requests is used to download a file from the Web
import requests

# os allows us to see what files we have
import os.path

# The filename we want to save the file as:
dataset_filename = "global_wordfreq.release_UTF-8.txt"

# If we don't have the file already, download it
if not os.path.exists(dataset_filename):
    # The url to download from:
    url = "https://www.plecoforums.com/download/global_wordfreq-release_utf-8-txt.2593/"
    # The response from the server when we ask for the file (contains the file)
    r = requests.get(url, allow_redirects=True)
    # We extract the file from the response and save it with the filename
    open(dataset_filename, "wb").write(r.content)

# Create an array called data that is basically a simple list
# (e.g. data[4] is how many times the 5th most popular word appears)
data = np.loadtxt(dataset_filename, dtype = int, comments = None, usecols = 1, encoding = "utf-8", max_rows = 100000)

# We want to scale the data logarithmically so that we can get Cn^-a (Zipf's law) => -a*n + C for the linear regression
data_log10 = np.log10(data)

# X-axis is also scaled logarithmically
x = np.log10(range(1, len(data) + 1))
# reshape is not an important function, it makes the array have a format that the later functions are made to work with
x_reshape = x.reshape(-1, 1)

# This line does linear regression (finds the straight line that best fits our data)
reg = LinearRegression().fit(x_reshape, data_log10)
# These 2 lines take the values produced by our regression and uses them to calculate back to what C and alpha would be
# (according to Zipf's law)
print(f"alpha = {-reg.coef_[0]}")
print(f"C = 10^{reg.intercept_}")
# This line tells us how close the regression line is to the raw data/how accurate our results are
print(f"r^2 = {reg.score(x_reshape, data_log10)}")

# This line generates the y values of the linear regression line so we can plot it
y_predict = reg.predict(x_reshape)

# This replicates the linear regression by using the formula of Zipf's law and doing log10 on it
# (to double check if we want to)
#y_zipf = np.log10(10**reg.intercept_ * np.array(range(1, len(data) + 1)) ** reg.coef_)
#plt.plot(x, y_zipf, label = "zipf")

# These lines mean: at the x values given by x, draw 2 graphs with y values given by data_log10 and y_predict
# '.' means to only draw points (look at the output and you will see)
# label =    is used to give each graph a label (look at the output and you will see)
plt.plot(x, data_log10, '.', label = "data")
plt.plot(x, y_predict, label = "linear regression")
# Here we add a title to the whole diagram (text at the top)
plt.title("Word frequencies in Chinese")
# xlabel and ylabel are labels for each axis of the coordinate system
plt.xlabel("Word index (log10)")
plt.ylabel("Occurences (log10)")
# we need this line if we want to add the box where we can see the label of each of the graphs
plt.legend()
# This defines how we save our plotted diagram. Open the file loglog.png after running this python program.
# dpi is how high of a resolution we want the image to have
# png is a lossless (best quality) image format
plt.savefig("loglog.png", dpi=300, format="png")
