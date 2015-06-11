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



catalog = "lwr_train_sdss_dr10.dat2"

cat = pd.read_table(catalog, delim_whitespace=True)

# Because of file bad formatting:
cat.drop('#', axis=1, inplace=True)

cols = list(cat.columns.values)

# print a column
#print cat.ix[:,'g']


