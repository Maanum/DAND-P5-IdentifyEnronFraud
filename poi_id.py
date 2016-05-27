#!/usr/bin/python

import sys
import pickle
import pprint
import csv

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


pp = pprint.PrettyPrinter(indent = 4)

def to_int(val):
    if isinstance(val, int):
        return val
    else:
        return 0

###############################################################################
###############################################################################
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = [
    'poi',
    
#    'total_stock_value',  # These features' data are extraneous/not usable
#    'total_payments',
#    'email_address',

#    'director_fees',   # These features removed due to SelectKBest scores
#    'expenses',
#    'from_messages',
#    'from_poi_to_this_person',
#    'from_this_person_to_poi',
#    'other',
#    'restricted_stock',
#    'shared_receipt_with_poi',
#    'to_messages',
#    'long_term_incentive',    
#    'deferral_payments',
#    'loan_advances',    
#    'restricted_stock_deferred',
    
    'bonus',    
    'deferred_income',
    'exercised_stock_options',
    'salary'    
    ] # You will need to use more features

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

data_dict.pop("TOTAL", None)

feature_names = list(features_list)
feature_names.pop(0)

###############################################################################
# Scan for outlying values

#for feat in feature_names:
#    items = []
#    for k, v in data_dict.iteritems():
#        value = to_int(data_dict[k][feat])
#        to_append = [k, value]
#        items.append(to_append)
#    arr = np.array(items)
#    arr.sort



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
#
###############################################################################
# Create "Pctg. of correspondence with POI" feature
# This will be a ratio of all emails to/from POI to all emails rcvd/sent

for k, v in data_dict.iteritems():
    corr_total_from = to_int(data_dict[k]['from_messages'])
    corr_total_to = to_int(data_dict[k]['to_messages'])
    corr_POI_from = to_int(data_dict[k]['from_poi_to_this_person'])
    corr_POI_to = to_int(data_dict[k]['from_poi_to_this_person'])
    
    corr_total = corr_total_from + corr_total_to
    corr_POI = corr_POI_to + corr_POI_from

    try:
        pct_corr_with_POI = float(corr_POI) / float(corr_total)
    except:
        pct_corr_with_POI = 0
        
    data_dict[k]['pct_corr_with_POI'] = pct_corr_with_POI

features_list.append('pct_corr_with_POI')


###############################################################################
# Create CSV file of data for review

outfile = "Enron_data.csv"

headers = list(features_list)
headers.insert(0, 'Name')
with open(outfile, "w") as f:
    w = csv.writer(f, delimiter=',', lineterminator='\n')
    w.writerow(headers)
    for k, v in data_dict.iteritems():
        values = []
        values.append(k)
        for feat in features_list:
            values.append(data_dict[k][feat])
        w.writerow(values)

# Create dataset for extract
my_dataset = data_dict

#
#
#### Extract features and labels from dataset for local testing
#

data = featureFormat(my_dataset, features_list, remove_any_zeroes=False, sort_keys = True)
labels, features = targetFeatureSplit(data)

###############################################################################
###############################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

###############################################################################
## KNeighborsClassifier Pipeline setup

scaler = StandardScaler()
kBest = SelectKBest()
knn = KNeighborsClassifier()

pipeline = Pipeline([#('fscale', scaler), 
                     ('fselect', kBest),
                     ('knn', knn)])

params_test = dict(fselect__score_func = [f_classif],
                  knn__n_neighbors=[3, 5, 7],
                  fselect__k = [2, 3, 4],
                  knn__weights = ['distance', 'uniform'],
                  knn__algorithm = ['brute', 'kd_tree', 'ball_tree'],
                  knn__leaf_size = [1, 3, 5])


###############################################################################
## SVM Pipeline setup

#scaler = StandardScaler()
#kBest = SelectKBest(f_classif)
#svm = SVC()
#
#pipeline = Pipeline([('fscale', scaler), 
#                     ('fselect', kBest),
#                     ('svm', svm)])
#
#params_test = dict(fselect__k = [2, 3, 4],
#                   svm__C = [1, 2, 10, 50],
#                   svm__kernel = ['rbf', 'linear'],
#                   svm__gamma = ['auto', 10])
          

###############################################################################
## Decision Tree Pipeline setup

#kBest = SelectKBest(f_classif)
#tree = DecisionTreeClassifier()
#
#pipeline = Pipeline([('fselect', kBest),
#                     ('tree', tree)])
#
#params_test = dict(fselect__k = [2, 3, 4],
#                   tree__min_samples_split = [2, 3, 4, 5],
#                   tree__criterion = ['gini', 'entropy'],
#                   tree__max_features = [1, 2])


###############################################################################
## Naive Bayes (Gaussian) Pipeline setup
#
#kBest = SelectKBest(f_classif)
#gnb = GaussianNB()
#
#pipeline = Pipeline([('fselect', kBest),
#                     ('gnb', gnb)])
#
#params_test = dict(fselect__k = [2, 3, 4])


###############################################################################
###############################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


###############################################################################
# Run GridSearchCV with same test/train split as tester
# Set clf to best parameters to be exported.

n_iters = 250
r_state = 42
cv = StratifiedShuffleSplit(labels, n_iter = n_iters, random_state = r_state)

grid_search = GridSearchCV(pipeline, params_test, cv = cv, scoring = "recall") 
grid_search.fit(features, labels)

params_new = grid_search.best_params_
clf = pipeline.set_params(**params_new)

print pipeline.named_steps['fselect'].fit(features, labels).scores_
###############################################################################
# Print final pipeline data/features for review/recording

features_only = list(features_list)
features_only.pop(0)  # Remove 'POI' from feature list (actually the label)

# Check if feature selection was used.  If not, return all feature names
if 'fselect' in pipeline.named_steps:
    features_used = pipeline.named_steps['fselect'].fit(features, labels).get_support()
else:
    features_used = [True] * len(features_only)

# Check if feature scaling was used.  If not, populate with normal values
if 'fscale' in pipeline.named_steps:
    features_scaled = pipeline.named_steps['fscale'].fit_transform(features)
else:
    features_scaled = features

# Cycle through all values and create readable sample dictionary
i = 0
feature_review = {}
for feat in features_only:
    if features_used[i]:
        features_line = []
        for ii in range(10):
            feature_val = features[ii][i]
            feature_val_scaled = features_scaled[ii][i]
            f_tuple = feature_val, feature_val_scaled
            features_line.append(f_tuple)
        feature_review[feat] = features_line
    i += 1

print
print "Number of Observations:\t", len(features)
print "Features used with sample of data including (original value, scaled value):"
pp.pprint(feature_review)

# Print the pipeline steps and best parameters
print
print ("Pipeline Used:")

for step in clf.steps:
    pp.pprint(step)

pred_class = []
actual_class = []

# Calculate precision and recall and report on evaluation metrics
for train_indices, test_indices in cv:
    features_train = [features[ii] for ii in train_indices]
    features_test = [features[ii] for ii in test_indices]
    labels_train = [labels[ii] for ii in train_indices]
    labels_test = [labels[ii] for ii in test_indices]

    clf.fit(features_train, labels_train)
    pred_class.extend(clf.predict(features_test))
    actual_class.extend(labels_test)
print
print 'Results ({} iterations, random state {}):'.format(n_iters, r_state)
print "\tPrecision: ", precision_score(actual_class, pred_class)
print "\tRecall:", recall_score(actual_class, pred_class)


###############################################################################
###############################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)