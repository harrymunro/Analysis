# Takes in some sample data 
# Uses MLE to estimate parameters for 86 distributions
# Creates a probability plot for each distribution
# Plots the raw data on each probability plot
# Calculates the R-squared coeffecient for the raw data vs the estimated distribution
# Returns the R-squared coeffecients and estimated parameters for each distribution in a .csv file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import math
import pandas as pd

# import data
f_name = "dwell.txt"
#df = pd.read_csv(f_name, sep = '\t')
x = np.loadtxt(f_name)
y = np.log(x)
#mu = np.std(y)

# mean and variance of data
var = np.var(x)
mean = np.mean(x)
# natural logarithms
var_log = np.var(y)
mean_log = np.mean(y)
print "Variance = %r" % var
print "Mean = %r" % mean
print "Variance log = %r" % var_log
print "Mean log = %r" % mean_log



#### lognormal paramaters ####


# mean and variance of distribution
lognormal_mean = math.exp(mean_log + 0.5 * var_log)
lognormal_var = math.exp(2 * mean_log + var_log) * (math.exp(var_log) - 1)
print lognormal_mean
print lognormal_var




# calculate shape parameter (complex way of re-arriving at var_log and mean_log)
mu = math.log(lognormal_mean/math.sqrt(1+(lognormal_var/lognormal_mean**2)))
sigma = math.sqrt(math.log(1+(lognormal_var/lognormal_mean**2)))




# plot
fig = plt.figure()
ax = fig.add_subplot(111)

# lognormal
lognormal_param = stats.lognorm.fit(x) # maximum likelihood fit
#print lognormal_param
#print stats.probplot(x, dist = stats.lognorm, sparams = lognormal_param, fit = True, plot = ax)

# weibull
weibull_param = stats.weibull_min.fit(x)
#print stats.probplot(x, dist = stats.weibull_min, sparams = weibull_param, fit = True, plot = ax)

# beta
beta_param = stats.beta.fit(x)
#print stats.probplot(x, dist = stats.beta, sparams = beta_param, fit = True, plot = ax)

# chi-squared
chi2_param = stats.chi2.fit(x)
#print stats.probplot(x, dist = stats.chi2, sparams = chi2_param, fit = True, plot = ax)

# normal
norm_param = stats.norm.fit(x)
#print stats.probplot(x, dist = stats.norm, sparams = norm_param, fit = True, plot = ax)

# log gamma
pareto_param = stats.pareto.fit(x)
#print stats.probplot(x, dist = stats.pareto, sparams = pareto_param, fit = True, plot = ax)

# to get r-squared value need to do probplot[1][2] then square it

# function to automatically calculate r squared values
def get_r2_dist_fit(distribution, data):
	param = distribution.fit(data)
	probplot = stats.probplot(x, dist = distribution, sparams = param, fit = True, plot = ax)
	print "R-squared value for %s is: %r\n" % (str(distribution), probplot[1][2]**2)
	dist_name = str(distribution)
	dist_name = dist_name[32:len(dist_name)-25]
	return (dist_name, param, probplot[1][2]**2)



# all scipy distributions
DISTRIBUTIONS = [stats.alpha, stats.anglit, stats.arcsine, stats.beta, stats.betaprime, stats.bradford, stats.burr, stats.cauchy, stats.chi, stats.chi2, stats.cosine, stats.dgamma, stats.dweibull, stats.erlang, stats.expon, stats.exponnorm, stats.exponweib, stats.exponpow, stats.f, stats.fatiguelife, stats.fisk, stats.foldcauchy, stats.foldnorm, stats.genlogistic, stats.genpareto, stats.gennorm, stats.genexpon, stats.genextreme, stats.gausshyper, stats.gamma, stats.gengamma, stats.genhalflogistic, stats.gilbrat, stats.gompertz, stats.gumbel_r, stats.gumbel_l, stats.halfcauchy, stats.halflogistic, stats.halfnorm, stats.halfgennorm, stats.hypsecant, stats.invgamma, stats.invgauss, stats.invweibull, stats.johnsonsb, stats.johnsonsu, stats.kstwobign, stats.laplace, stats.levy, stats.levy_l, stats.levy_stable, stats.logistic, stats.loggamma, stats.loglaplace, stats.lognorm, stats.lomax, stats.maxwell, stats.mielke, stats.nakagami, stats.ncx2, stats.ncf, stats.nct, stats.norm, stats.pareto, stats.pearson3, stats.powerlaw, stats.powerlognorm, stats.powernorm, stats.rdist, stats.reciprocal, stats.rayleigh, stats.rice, stats.recipinvgauss, stats.semicircular, stats.t, stats.triang, stats.truncexpon, stats.truncnorm, stats.tukeylambda, stats.uniform, stats.vonmises, stats.vonmises_line, stats.wald, stats.weibull_min, stats.weibull_max, stats.wrapcauchy]

results = []

for distribution in DISTRIBUTIONS:
	try:
		results.append(get_r2_dist_fit(distribution, x))
	except Exception:
		results.append((str(distribution)[32:len(str(distribution))-25], "Failed to fit", "n/a"))
		pass

df = pd.DataFrame(results)
#df = df.transpose()
df.to_csv("results.csv")
print "Successfully saved to csv file."
	

# more advanced plot with statsmodels
#probplot = sm.ProbPlot(x, stats.lognorm, distargs=(mean_log,), fit=True)
#fig = probplot.qqplot(line='45')


# basic histogram
#dist_plot = plt.hist(x)

#plt.show()
