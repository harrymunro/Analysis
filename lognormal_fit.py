# Fits a lognormal distribution to data

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Import data
dat = np.loadtxt("x.txt")

n, bins, patches = plt.hist(dat, bins=25, normed=True)

shape, loc, scale = stats.lognorm.fit(dat, floc=0) # Fit a curve to the variates
mu = np.log(scale) # Mean of log(X)
sigma = shape # Standard deviation of log(X)
M = np.exp(mu) # Geometric mean == median
s = np.exp(sigma) # Geometric standard deviation

# Plot figure of results
sns.set_style("whitegrid")
x = np.linspace(dat.min(), dat.max(), num=400)
plt.plot(x, stats.lognorm.pdf(x, shape, loc=0, scale=scale), 'r', linewidth=3) # Plot fitted curve
ax = plt.gca() # Get axis handle for text positioning
txt = plt.text(0.9, 0.9, 'mu = %.2f\nsigma = %.2f' % (mu, sigma), horizontalalignment='right', size='large', verticalalignment='top', transform=ax.transAxes)
plt.xlabel("")
plt.ylabel("")

plt.show()
