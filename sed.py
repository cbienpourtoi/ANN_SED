#! /usr/local/bin python
# -*- coding: iso-8859-1 -*-
#

__author__ = "Loic Le Tiran"
__copyright__ = "Copyright 2015"
__credits__ = "Loic Le Tiran"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Loic Le Tiran"
__email__ = "loic.letiran@gmail.com"
__status__ = "Development"

""" ML techniques (ANN) for deriving galaxy properties from their colours without using SED fitting"""

#################
### Packages ####
#################
import sys
import pandas as pd
import machine
from sklearn_pandas import DataFrameMapper, cross_val_score
#import sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn import datasets, linear_model
import matplotlib.pylab as plt
import numpy as np
from sklearn import tree

from pybrain.tools.shortcuts import buildNetwork

catalog = "lwr_train_sdss_dr10.dat2"

cat = pd.read_table(catalog, delim_whitespace=True)

# Because of file bad formatting:
cat.drop('#', axis=1, inplace=True)

cols = list(cat.columns.values)

# print a column
#print cat.ix[:,'g']

# Delete "weird" objects:
lcatinit = len(cat)
cat_min = 14
cat_max = 24
for c in cols[:-1]:
    cat = cat[cat.ix[:, c] > cat_min]
    cat = cat[cat.ix[:, c] < cat_max]
lcatclean = len(cat)
print "deleted "+str(lcatinit - lcatclean)+ " objects over " + str(lcatinit) + " ("+str((lcatinit - lcatclean)/float(lcatinit) * 100.) +"% deletion)"


# Keeps only the bright objects to make tests on the good data first.
bright_limit = 20.
cat_bright = cat[cat.ix[:, "r"] < bright_limit]
print "The bright sample contains "+str(len(cat_bright))+" objects."


cat_train = cat_bright.iloc[::2]
cat_target = cat_bright.iloc[1::2]

cat_train_X = cat_train.drop('specz', axis=1)
cat_train_y = cat_train.ix[:,'specz']

cat_target_X = cat_target.drop('specz', axis=1)
cat_target_true = cat_target.ix[:,'specz']


#regr = linear_model.LinearRegression()
#regr = linear_model.BayesianRidge()
regr = tree.DecisionTreeRegressor()
regr.fit(cat_train_X, cat_train_y)

cat_target_prediction = regr.predict(cat_target_X)

diff = np.array(cat_target_true) - cat_target_prediction
np_cat_target_true = np.array(cat_target_true)

print diff.std()

plt.figure()
#plt.hist2d(np.array(cat_target_true), cat_target_prediction, bins=1000)
plt.hist2d(np_cat_target_true, diff, bins=1000)
plt.title(str(regr)[0:15])
plt.xlabel('z')
plt.ylabel('dz')
plt.xlim([0., 0.6])
plt.ylim([-0.08, 0.08])
#plt.show()
plt.savefig("plots/"+str(regr)[0:13]+".png")




"""
model = machine.logistic()
model.train(cat_train_X, cat_train_y)
cat_target_prediction = model.use(cat_target_X)
"""


"""
mapper = DataFrameMapper([('r', sklearn.preprocessing.StandardScaler())])
print mapper.fit_transform(cat_bright.copy())
"""
