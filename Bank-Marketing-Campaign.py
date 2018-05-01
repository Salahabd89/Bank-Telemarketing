import pandas as pd
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from sklearn import tree, grid_search, decomposition
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.cross_validation import cross_val_score
from IPython.display import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler


bank_filename = "./Data/bank-additional-full.csv"
bank_data = pd.read_csv(bank_filename, sep = ";", header = 0)

# Function Definitions
def RemoveColumns(column, dataset):
    if(column in dataset.columns):
        del dataset[column]
        print "removing '{0}' variable".format(column)
    else:
        print "Column '{0}' does not exist".format(column)

def AssignIntegerLabels(dataset):
    for column in dataset.columns:
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
    print "Assigning integer label to '{0}' variable".format(column)

def convertSpecificColumnToNumeric(value, dataset):
    for column in dataset.columns:
        if column == value:
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
            print "Converting '{0}' variable to an integer".format(column)

def columnSplitter(df):

    for column in df.columns:
        if df[column].dtype == type(object) and column != 'y':
            i = 1
            category = df[column].unique()

            if (len(category) == 2):
                    mask_0 = df[column] == category[0]
                    df.loc[mask_0, column] = 0
                    mask_1 = df[column] == category[1]
                    df.loc[mask_1, column] = 1
            else:
                for values in df[column].unique():
                    columname = "{0}".format(values)
                    df[columname] = 0

                    mask = df[column] == values
                    df.loc[mask,columname] = i

                print "Splitting categorical variable '{0}'. Converting to integer variable '{1}'".format(column,values )

    for column in df.columns:
        if df[column].dtype == type(object) and column != 'y':
            del df[column]
            print "Removing variable {0}...".format(column)


# *************************** Data Exploration ******************************


print "Data Exploration Begin.."
print
print "Data Frame shape: " , bank_data.shape
print
# rename some variables with "." in name
bank_data = bank_data.rename(columns={'emp.var.rate' : 'emp_var_rate', 'cons.price.idx' : 'cons_price_idx', 'cons.conf.idx' : 'cons_conf_idx', 'nr.employed' : 'nr_employed'})
# remove variable 'duration' - because it is not useful for prediction and is actaully harmful
RemoveColumns('duration',bank_data)
print "Data Frame shape: " , bank_data.shape
print bank_data.dtypes
print
print "Measure of correlation"
print
bank_data.corr(method='pearson', min_periods=1)

bank_data['age'].plot(kind='hist', color= '#37AB65')
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of Age', fontsize = 16)
plt.grid()
plt.show()

print "Minimum of Age : %.0f" % (min(bank_data['age']))
print "Maximum of Age : %.0f" %(max(bank_data['age']))

bank_data['pdays'].plot(kind='hist', color= '#c51b8a')
plt.xlabel('Pdays', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of pdays', fontsize = 16)
plt.grid()
plt.show()

print "Minimum of Pdays : %.0f" % (min(bank_data['pdays']))
print "Maximum of Pdays : %.0f" %(max(bank_data['pdays']))

no_999 = bank_data.loc[lambda df: bank_data.pdays != 999, : ]

no_999['pdays'].plot(kind='hist', color='#c51b8a')
plt.xlabel('Pdays', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title("Histogram of Pdays ('999' values excluded)", fontsize = 16)
plt.grid()
plt.show()

print "Minimum of Pdays ('999' values excluded) : %.0f" % (min(no_999['pdays']))
print "Maximum of Pdays ('999' values excluded) : %.0f" %(max(no_999['pdays']))

bank_data['previous'].plot(kind='hist', color= '#8C0B90')
plt.xlabel('Previous', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of Previous', fontsize = 16)
plt.grid()
plt.show()

print "Minimum of Previous : %.0f" % (min(bank_data['previous']))
print "Maximum of Previous : %.0f" % (max(bank_data['previous']))


bank_data['emp_var_rate'].plot(kind='hist', color= '#C0E4FF')
plt.xlabel('Employment Variation Rate', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of Employment Variation Rate', fontsize = 16)
plt.grid()
plt.show()

print "Minimum of Employment Variation Rate : %.1f" % (min(bank_data['emp_var_rate']))
print "Maximum of Employment Variation Rate : %.1f" %(max(bank_data['emp_var_rate']))

bank_data['cons_price_idx'].plot(kind='hist', color= '#7C60A8')
plt.xlabel('Consumer Price Index ', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of Consumer Price Index ', fontsize = 16)
plt.grid()
plt.show()

print "Minimum of Consumer Price Index  : %.1f" % (min(bank_data['cons_price_idx']))
print "Maximum of Consumer Price Index  : %.1f" %(max(bank_data['cons_price_idx']))

bank_data['cons_conf_idx'].plot(kind='hist', color= '#CF95D7')
plt.xlabel('Consumer Confidence Index ', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of Consumer Confidence Index ', fontsize = 16)
plt.grid()
plt.show()

print "Minimum of Consumer Confidence Index  : %.1f" % (min(bank_data['cons_conf_idx']))
print "Maximum of Consumer Confidence Index  : %.1f" %(max(bank_data['cons_conf_idx']))

bank_data['euribor3m'].plot(kind='hist', color= '#3DF735')
plt.xlabel('Euribor 3 Month Rate', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of Euribor 3 Month Rate', fontsize = 16)
plt.grid()
plt.show()

print "Minimum of Euribor 3 Month Rate : %.1f" % (min(bank_data['euribor3m']))
print "Maximum of Euribor 3 Month Rate : %.1f" %(max(bank_data['euribor3m']))

bank_data['nr_employed'].plot(kind='hist', color= '#C0E4FF')
plt.xlabel('Number Of Employees', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Histogram of Number Of Employees', fontsize = 16)
plt.grid()
plt.show()

print "Minimum of Number Of Employees : %.1f" % (min(bank_data['nr_employed']))
print "Maximum of Number Of Employees : %.1f" %(max(bank_data['nr_employed']))

colours = ['#37AB65', '#3DF735', '#c51b8a', '#AD6D70', '#fec44f', '#2c7fb8', '#d95f0e', '#EC2504', '#8C0B90', '#C0E4FF', '#fff7bc', '#27B502', '#7C60A8', '#CF95D7', '#F6CC1D']


column = 'job'

p3 = bank_data[column].value_counts().sort_values(ascending=True).plot(kind='bar', color = colours[0], figsize=(4,3))
plt.title("Bar chart of Job", fontsize = 16)
plt.ylabel('Count', fontsize=14)
plt.grid()
plt.show()

percent = (bank_data[column].value_counts(normalize=True).round(3) * 100)
print "Percentage of observations in each Category:"
print percent

column = 'contact'

p3 = bank_data[column].value_counts().sort_values(ascending=True).plot(kind='bar', color = colours[1], figsize=(4,3))
plt.title("Bar chart of Contact type", fontsize = 16)
plt.ylabel('Count', fontsize=14)
plt.grid()
plt.show()

percent = (bank_data[column].value_counts(normalize=True).round(3) * 100)
print "Percentage of observations in each Category:"
print percent

column = 'poutcome'

p3 = bank_data[column].value_counts().sort_values(ascending=True).plot(kind='bar', color = colours[2], figsize=(4,3))
plt.title("Bar chart of Previous Outcome", fontsize = 16)
plt.ylabel('Count', fontsize=14)
plt.grid()
plt.show()

percent = (bank_data[column].value_counts(normalize=True).round(3) * 100)
print "Percentage of observations in each Category:"
print percent

column = 'y'

p3 = bank_data[column].value_counts().sort_values(ascending=True).plot(kind='bar', color = colours[3], figsize=(4,3))
plt.title("Bar chart of y", fontsize = 16)
plt.ylabel('Count', fontsize=14)
plt.grid()
plt.show()

percent = (bank_data[column].value_counts(normalize=True).round(3) * 100)
print "Percentage of observations in each Category:"
print percent

df_grouped = bank_data.groupby(['day_of_week']).size().reset_index()

weekdays = ['mon', 'tue', 'wed', 'thu', 'fri']

mapping = {day: i for i, day in enumerate(weekdays)}
key = df_grouped['day_of_week'].map(mapping)
df_grouped = df_grouped.iloc[key.argsort()]

barchart = df_grouped.plot(kind='bar', x='day_of_week', figsize=(8,6),fontsize=12)
barchart.set_title("Bar chart of Day Of Week \n".format('day_of_week'),fontsize=16)
barchart.set_xlabel('\nDay Of Week',fontsize=12)
barchart.set_ylabel("Count",fontsize=12)
plt.gca().legend_.remove()
plt.grid()
plt.show()

percent = (bank_data['day_of_week'].value_counts(normalize=True).round(3) * 100)
print "Percentage of observations in each Category:"
print percent

df_grouped = bank_data.groupby(['month']).size().reset_index()

months = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

mapping = {month: i for i, month in enumerate(months)}
key = df_grouped['month'].map(mapping)
df_grouped = df_grouped.iloc[key.argsort()]

barchart = df_grouped.plot(kind='bar', x='month', figsize=(8,6),fontsize=12)
barchart.set_title("Bar chart of Month \n".format('month'),fontsize=16)
barchart.set_xlabel('\n Month',fontsize=12)
barchart.set_ylabel("Count",fontsize=12)
plt.gca().legend_.remove()
plt.grid()
plt.show()

percent = (bank_data['month'].value_counts(normalize=True).round(3) * 100)
print "Percentage of observations in each Category:"
print percent

df_grouped = bank_data.groupby(['marital']).size().reset_index()

marital_status = ['unknown', 'single', 'married', 'divorced']

mapping = {status: i for i, status in enumerate(marital_status)}
key = df_grouped['marital'].map(mapping)
df_grouped = df_grouped.iloc[key.argsort()]

barchart = df_grouped.plot(kind='bar', x='marital', figsize=(6,4),fontsize=16)
barchart.set_title("Bar chart of Marital Status \n".format('marital'),fontsize=16)
barchart.set_xlabel('\n Marital Status',fontsize=16)
barchart.set_ylabel("Count",fontsize=16)
plt.gca().legend_.remove()
plt.grid()
plt.show()

percent = (bank_data['marital'].value_counts(normalize=True).round(3) * 100)
print "Percentage of observations in each Category:"
print percent

df_grouped = bank_data.groupby(['education']).size().reset_index()

education_status = ['unknown', 'illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'university.degree', 'professional.course']

mapping = {status: i for i, status in enumerate(education_status)}
key = df_grouped['education'].map(mapping)
df_grouped = df_grouped.iloc[key.argsort()]

barchart = df_grouped.plot(kind='bar', x='education', figsize=(7,5),fontsize=12)
barchart.set_title("Bar chart of Education Level \n".format('education'),fontsize=16)
barchart.set_xlabel('\n Education Level',fontsize=12)
barchart.set_ylabel("Count",fontsize=12)
plt.gca().legend_.remove()
plt.show()

percent = (bank_data['education'].value_counts(normalize=True).round(3) * 100)
print "Percentage of observations in each Category:"
print percent

df_grouped = bank_data.groupby(['default']).size().reset_index()

default_status = ['unknown', 'yes', 'no']

mapping = {status: i for i, status in enumerate(default_status)}
key = df_grouped['default'].map(mapping)
df_grouped = df_grouped.iloc[key.argsort()]

barchart = df_grouped.plot(kind='bar', x='default', figsize=(4,4),fontsize=10)
barchart.set_title("Bar chart of Default Status \n".format('default'),fontsize=14)
barchart.set_xlabel('\n Default Status',fontsize=12)
barchart.set_ylabel("Count",fontsize=12)
plt.gca().legend_.remove()
plt.show()

percent = (bank_data['default'].value_counts(normalize=True).round(3) * 100)
print "Percentage of observations in each Category:"
print percent
print
print "Count of observations in each Category:"
print bank_data['default'].value_counts()


df_grouped = bank_data.groupby(['housing']).size().reset_index()

default_status = ['unknown', 'yes', 'no']

mapping = {status: i for i, status in enumerate(default_status)}
key = df_grouped['housing'].map(mapping)
df_grouped = df_grouped.iloc[key.argsort()]

barchart = df_grouped.plot(kind='bar', x='housing', figsize=(8,6),fontsize=10)
barchart.set_title("Bar chart of Housing loan status \n".format('housing'),fontsize=14)
barchart.set_xlabel('\n Housing loan Status',fontsize=12)
barchart.set_ylabel("Count",fontsize=12)
plt.gca().legend_.remove()
plt.show()

percent = (bank_data['housing'].value_counts(normalize=True).round(3) * 100)
print "Percentage of observations in each Category:"
print percent

df_grouped = bank_data.groupby(['loan']).size().reset_index()

default_status = ['unknown', 'yes', 'no']

mapping = {status: i for i, status in enumerate(default_status)}
key = df_grouped['loan'].map(mapping)
df_grouped = df_grouped.iloc[key.argsort()]

barchart = df_grouped.plot(kind='bar', x='loan', figsize=(6,4),fontsize=10)
barchart.set_title("Bar chart of Personal Loan status \n".format('loan'),fontsize=16)
barchart.set_xlabel('\n Personal Loan Status',fontsize=12)
barchart.set_ylabel("Count",fontsize=12)
plt.gca().legend_.remove()
plt.show()

percent = (bank_data['loan'].value_counts(normalize=True).round(3) * 100)
print "Percentage of observations in each Category:"
print percent

scatter_matrix(bank_data, alpha=0.2, figsize=(16,16), diagonal='hist')
plt.show()

bank_data.boxplot(column='age', by='y')
plt.suptitle("")
plt.title('Boxplot of Age by y', fontsize=14)
plt.ylabel('Age', fontsize=10)
plt.xlabel('y', fontsize=10)
plt.show()


bank_data.boxplot(column='age', by='marital')
plt.suptitle("")
plt.title('Boxplot of Age by Marital Status', fontsize=14)
#p3.set_xticklabels(['single-credit', 'more than \n single-credit'], rotation=0, fontsize=14)

plt.ylabel('Age', fontsize=10)
plt.xlabel('\n Marital Status', fontsize=10)
plt.show()

bank_data.boxplot(column='age', by='housing')
plt.suptitle("")
plt.title('Boxplot of Age by Housing Loan Status', fontsize=14)
#p3.set_xticklabels(['single-credit', 'more than \n single-credit'], rotation=0, fontsize=14)

plt.ylabel('Age', fontsize=10)
plt.xlabel('\n Housing Loan Status', fontsize=10)
plt.show()

bank_data.boxplot(column='age', by='loan')
plt.suptitle("")
plt.title('Boxplot of Age by Personal Loan Status', fontsize=14)
#p3.set_xticklabels(['single-credit', 'more than \n single-credit'], rotation=0, fontsize=14)

plt.ylabel('Age', fontsize=10)
plt.xlabel('\n Personal Loan Status', fontsize=10)
plt.show()

bank_data.boxplot(column='euribor3m', by='y')
plt.suptitle("")
plt.title('Boxplot of euribor3m by y', fontsize=14)
plt.ylabel('euribor3m', fontsize=10)
plt.xlabel('y', fontsize=10)
plt.show()

colors_palette = {'yes': 'red', 'no': 'blue'}
colors = [colors_palette[c] for c in bank_data['y']]
bank_data.plot(kind='scatter', x='emp_var_rate', y='nr_employed', c=colors)
red_patch = mpatches.Patch(color='red', label='yes')
blue_patch = mpatches.Patch(color='blue', label='no')
plt.legend(handles=[red_patch, blue_patch])
plt.ylabel('Number of Employees', fontsize=12)
plt.xlabel('Employment Variation Rate', fontsize=12)
plt.title('Scatterplot of Employment Variation Rate \n versus Number of Employees \n versus y', fontsize=16)
plt.show()

colors_palette = {'yes': 'red', 'no': 'blue'}
colors = [colors_palette[c] for c in bank_data['y']]
bank_data.plot(kind='scatter', x='emp_var_rate', y='cons_conf_idx', c=colors)
red_patch = mpatches.Patch(color='red', label='yes')
blue_patch = mpatches.Patch(color='blue', label='no')
plt.legend(handles=[red_patch, blue_patch])
plt.ylabel('Consumer Confidence Index', fontsize=12)
plt.xlabel('Employment Variation Rate', fontsize=12)
plt.title('Scatterplot of Employment Variation Rate \n versus Consumer Confidence Index \n versus y', fontsize=16)
plt.show()

sub_df_1 = bank_data.groupby(['loan'])['y'].size()
sub_df_1.plot(kind="bar", color="red",figsize=(15,15),fontsize=16)

sub_df_2 = bank_data[bank_data['y'] == 'no'].groupby(['loan'])['y'].size()
ax = sub_df_2.plot(kind="bar",figsize=(10,5),fontsize=16)

ax.set_ylabel('Count',fontsize=13)
ax.set_xlabel('Loan Status',fontsize=13)
ax.set_title('Response to campaign based on customer having a loan',fontsize=14)

ax.set_xticklabels(['Has no loan','Unknown','Has loan'])

plt.legend(['Yes', 'No'])
plt.show()

sub_df_1 = bank_data.groupby(['marital'])['y'].size()
sub_df_1.plot(kind="bar", color="red",figsize=(15,15),fontsize=16)

sub_df_2 = bank_data[bank_data['y'] == 'no'].groupby(['marital'])['y'].size()
ax = sub_df_2.plot(kind="bar",figsize=(10,5),fontsize=16)

ax.set_ylabel('Count',fontsize=13)
ax.set_xlabel('Marital Status',fontsize=13)
ax.set_title('Response to campaign based on marital status',fontsize=14)

plt.legend(['Yes', 'No'])
plt.show()

sub_df_1 = bank_data.groupby(['education'])['y'].size()
sub_df_1.plot(kind="bar", color="red",figsize=(15,15),fontsize=16)

sub_df_2 = bank_data[bank_data['y'] == 'no'].groupby(['education'])['y'].size()
ax = sub_df_2.plot(kind="bar",figsize=(10,5),fontsize=16)

ax.set_ylabel('Count',fontsize=13)
ax.set_xlabel('Level of education',fontsize=13)
ax.set_title('Response to campaign based on customer occupation',fontsize=14)

plt.legend(['Yes', 'No'])
plt.show()

sub_df_1 = bank_data.groupby(['job'])['y'].size()
sub_df_1.plot(kind="bar", color="red",figsize=(15,15),fontsize=16)

sub_df_2 = bank_data[bank_data['y'] == 'no'].groupby(['job'])['y'].size()
ax = sub_df_2.plot(kind="bar",figsize=(10,5),fontsize=16)

ax.set_ylabel('Count',fontsize=13)
ax.set_xlabel('Occupation',fontsize=13)
ax.set_title('Response to campaign based on customer education',fontsize=14)

plt.legend(['Yes', 'No'])
plt.show()

bank_data.loc[bank_data['pdays'] == 999, 'pdays'] = 50
# Check for null values
null_data = bank_data[bank_data.isnull().any(axis=1)]
len(null_data)

print "Data Exploration Complete"

# *************************** Data Modelling ******************************

# Convert Y variable to numeric
convertSpecificColumnToNumeric('y',bank_data)

# Create seperate data sets
knn_Data = bank_data.copy(deep=True)
DT_Data  = bank_data.copy(deep=True)
LR_Data  = bank_data.copy(deep=True)

AssignIntegerLabels(DT_Data)
columnSplitter(LR_Data)
columnSplitter(knn_Data)

y = bank_data.y
x_DT = DT_Data.ix[:, DT_Data.columns != 'y']
x_KNN = knn_Data.ix[:, knn_Data.columns != 'y']
x_LR = LR_Data.ix[:, LR_Data.columns != 'y']

# Test Train Split for KNN
x_KNN_train, x_KNN_test, y_KNN_train, y_KNN_test = train_test_split(x_KNN, y, test_size=0.2, random_state=0)

# Test Train Split for Decision Tree
x_DT_train, x_DT_test, y_DT_train, y_DT_test = train_test_split(x_DT, y, test_size=0.2, random_state=0)

# Test Train Split for Logistic Regresion
x_LR_train, x_LR_test, y_LR_train, y_LR_test = train_test_split(x_LR, y, test_size=0.2, random_state=0)

# DT GRID SEARCH
print "Decision Tree Grid Search Started.."
print
parameters = {'criterion':('entropy','gini'),'max_depth': range(1, 10),'min_samples_split': range(2,10), 'min_samples_leaf': range(2,10)}
DT = grid_search.GridSearchCV(DecisionTreeClassifier(), parameters, cv=10, scoring='accuracy')
DT.fit(X=x_DT_train, y=y_DT_train)
print
print"DT Best mean score is: {0}".format(DT.best_score_)
print"DT Best set of paramters are:{0}".format(DT.best_params_)

grid_mean_scores = [result.mean_validation_score for result in DT.grid_scores_]
index = grid_mean_scores.index(min(grid_mean_scores))
print
print"DT Lowest mean score is: {0}".format(DT.grid_scores_[0].mean_validation_score)
print"DT Lowest scoring set of paramters are:{0}".format(DT.grid_scores_[0].parameters)
print
print "Decision Tree Grid Search Done."
print
# KNN CROSS FOLD VALIDATION - condition 1

# empty list that will hold cv scores
cv_scores_1 = []

# PIPELINE VERSION - each K-fold is normalized then run through KNN
# range([start], stop[, step])  ==> it doesn't include the 'stop' value.
print "KNN CROSS FOLD VALIDATION - condition 1"
print

for i in range(1, 20, 2):
    clf = make_pipeline(MinMaxScaler(feature_range=(0, 1)),KNeighborsClassifier(n_neighbors=i))
    scores = cross_val_score(clf, x_KNN_train, y_KNN_train, cv=10, scoring='accuracy')  # cv=10 means that there are 10 groups for cross-val
    print (scores)
    # The mean score and the 95% confidence interval of the score estimate are hence given by:
    print "n_neighbours = ",i,".   Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    cv_scores_1.append(scores.mean())

# KNN CROSS FOLD VALIDATION - condition 2
cv_scores_2 = []

print "KNN CROSS FOLD VALIDATION - condition 2"
print

for i in range(1, 20, 2):
    clf = make_pipeline(MinMaxScaler(feature_range=(0, 1)), KNeighborsClassifier(n_neighbors=i, weights='distance'))
    scores = cross_val_score(clf, x_KNN_train, y_KNN_train, cv=10, scoring='accuracy')
    print(scores)
    print "n_neighbours = ",i, ". weights = distance .", " Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    cv_scores_2.append(scores.mean())

# KNN CROSS FOLD VALIDATION - condition 3
cv_scores_3 = []

print "KNN CROSS FOLD VALIDATION - condition 3"
print

for i in range(1, 20, 2):
    clf = make_pipeline(MinMaxScaler(feature_range=(0, 1)), KNeighborsClassifier(n_neighbors=i, p=2))
    scores = cross_val_score(clf, x_KNN_train, y_KNN_train, cv=10, scoring='accuracy')
    print(scores)
    print "n_neighbours = ",i, ". p = 2 .", " Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    cv_scores_3.append(scores.mean())

    # KNN CROSS FOLD VALIDATION - condition 4
cv_scores_4 = []

print "KNN CROSS FOLD VALIDATION - condition 4"
print

for i in range(1, 20, 2):
    clf = make_pipeline(MinMaxScaler(feature_range=(0, 1)), KNeighborsClassifier(n_neighbors=i, weights='distance', p=2))
    scores = cross_val_score(clf, x_KNN_train, y_KNN_train, cv=10, scoring='accuracy')
    print(scores)
    print "n_neighbours = ",i, ". weights = distance .", " p = 2 .", " Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
    cv_scores_4.append(scores.mean())
# changing to misclassification error
MSE_1 = [1 - x for x in cv_scores_1]
MSE_2 = [1 - x for x in cv_scores_2]
MSE_3 = [1 - x for x in cv_scores_3]
MSE_4 = [1 - x for x in cv_scores_4]

# determining best k
neighbors = range(1, 20, 2)
optimal_k_1 = neighbors[MSE_1.index(min(MSE_1))]
optimal_k_2 = neighbors[MSE_2.index(min(MSE_2))]
optimal_k_3 = neighbors[MSE_3.index(min(MSE_3))]
optimal_k_4 = neighbors[MSE_4.index(min(MSE_4))]

print "condition 1: The optimal number of neighbors is %d" % optimal_k_1
print "condition 2: The optimal number of neighbors is %d" % optimal_k_2
print "condition 3: The optimal number of neighbors is %d" % optimal_k_3
print "condition 4: The optimal number of neighbors is %d" % optimal_k_4


# plot misclassification error vs k
plt.plot(neighbors, MSE_1, color = "red", label = "condition 1")
plt.plot(neighbors, MSE_2, color = "blue", label = "condition 2")
plt.plot(neighbors, MSE_3, color = "Yellow", label = "condition 3")
plt.plot(neighbors, MSE_3, color = "#00ff7f", label = "condition 4")
plt.xlabel('Number of Neighbors')
plt.ylabel('Misclassification Error')
plt.legend()
plt.show()

# plot misclassification error vs k
plt.plot(neighbors, MSE_1, color = "red", label = "condition 1")
#plt.plot(neighbors, MSE_2, color = "blue", label = "condition 2")
plt.plot(neighbors, MSE_3, color = "Yellow", label = "condition 3")
#plt.plot(neighbors, MSE_3, color = "#00ff7f", label = "condition 4")
plt.xlabel('Number of Neighbors')
plt.ylabel('Misclassification Error')
plt.legend()
plt.show()


# TESTING THE MODEL FOR DECISION TREE
clfDT = DecisionTreeClassifier(min_samples_split =6, criterion='entropy', max_depth=3, min_samples_leaf = 9) # use optimized parameters

fitDT = clfDT.fit(x_DT_train, y_DT_train)
predicted = fitDT.predict(x_DT_test)

print "***[Decision Tree Results]***"
print
print confusion_matrix(y_DT_test, predicted)
print
print "accuracy_score: {0}".format(accuracy_score(y_DT_test, predicted))
print
print classification_report(y_DT_test, predicted)

dotfile = open("bank_tree.dot", 'w')
dotfile = tree.export_graphviz(clfDT, out_file = dotfile, feature_names = x_DT.columns)


# TESTING THE MODEL FOR KNN
min_max_scaler = preprocessing.MinMaxScaler()

x_KNN_train_scaled = min_max_scaler.fit_transform(x_KNN_train)  # fits the scaler and then transforms the tng data
x_KNN_test_scaled = min_max_scaler.transform(x_KNN_test) # transforms the test  data (using the fitted scaler calculated in the previous step)

# final test of model
clfKNN = KNeighborsClassifier(n_neighbors= 19, p=1)  # use optimized parameters
fit = clfKNN.fit(x_KNN_train_scaled, y_KNN_train)
predicted = fit.predict(x_KNN_test_scaled)

print "***[K-nearest neighbors Results]***"
print
print confusion_matrix(y_KNN_test, predicted)
print
print "accuracy_score: {0}".format(accuracy_score(y_KNN_test, predicted))
print
print classification_report(y_KNN_test, predicted)


# LR CROSS FOLD VALIDATION - condition 1

# empty list that will hold cv scores
lr_cv_scores_1 = []

collection = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

print "LOGISTIC REGRESSION CROSS FOLD VALIDATION - condition 1"
print

for i in collection:
    lr = make_pipeline(MinMaxScaler(feature_range=(0, 1)),LogisticRegression(C=i, penalty='l1'))
    scores = cross_val_score(lr, x_LR_train, y_LR_train, cv=10, scoring='accuracy')
    print(scores)
    print(i, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    lr_cv_scores_1.append(scores.mean())

# LR CROSS FOLD VALIDATION - condition 2

lr_cv_scores_2 = []

collection = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

print "LOGISTIC REGRESSION CROSS FOLD VALIDATION - condition 2"
print

for i in collection:
    lr = make_pipeline(MinMaxScaler(feature_range=(0, 1)),LogisticRegression(C=i, penalty='l1', class_weight='balanced'))
    scores = cross_val_score(lr, x_LR_train, y_LR_train, cv=10, scoring='accuracy')
    print(scores)
    print(i, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    lr_cv_scores_2.append(scores.mean())

# LR CROSS FOLD VALIDATION - condition 3

lr_cv_scores_3 = []

collection = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

print "LOGISTIC REGRESSION CROSS FOLD VALIDATION - condition 3"
print

for i in collection:
    lr = make_pipeline(MinMaxScaler(feature_range=(0, 1)),LogisticRegression(C=i, penalty='l2'))
    scores = cross_val_score(lr, x_LR_train, y_LR_train, cv=10, scoring='accuracy')
    print(scores)
    print(i, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    lr_cv_scores_3.append(scores.mean())

# LR CROSS FOLD VALIDATION - condition 4

lr_cv_scores_4 = []

collection = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

print "LOGISTIC REGRESSION CROSS FOLD VALIDATION - condition 4"
print

for i in collection:
    lr = make_pipeline(MinMaxScaler(feature_range=(0, 1)),LogisticRegression(C=i, penalty='l2', class_weight='balanced'))
    scores = cross_val_score(lr, x_LR_train, y_LR_train, cv=10, scoring='accuracy')
    print(scores)
    print(i, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    lr_cv_scores_4.append(scores.mean())

    # changing to misclassification error
MSE_1 = [1 - x for x in lr_cv_scores_1]
MSE_2 = [1 - x for x in lr_cv_scores_2]
MSE_3 = [1 - x for x in lr_cv_scores_3]
MSE_4 = [1 - x for x in lr_cv_scores_4]

# determining best C
collection = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
optimal_C_1 = collection[MSE_1.index(min(MSE_1))]
optimal_C_2 = collection[MSE_2.index(min(MSE_2))]
optimal_C_3 = collection[MSE_3.index(min(MSE_3))]
optimal_C_4 = collection[MSE_4.index(min(MSE_4))]

print "condition 1: The optimal value of C is %.4f" % optimal_C_1
print "condition 2: The optimal value of C is %.4f" % optimal_C_2
print "condition 3: The optimal value of C is %.4f" % optimal_C_3
print "condition 4: The optimal value of C is %.4f" % optimal_C_4


# plot misclassification error vs k
plt.plot(collection, MSE_1, color = "red", label = "condition 1")
plt.plot(collection, MSE_2, color = "blue", label = "condition 2")
plt.plot(collection, MSE_3, color = "Yellow", label = "condition 3")
plt.plot(collection, MSE_4, color = "#00ff7f", label = "condition 4")
plt.xlabel('Value of C')
plt.ylabel('Misclassification Error')
plt.legend()
plt.show()

# plot misclassification error vs k
plt.plot(collection, MSE_1, color = "red", label = "condition 1")
#plt.plot(collection, MSE_2, color = "blue", label = "condition 2")
plt.plot(collection, MSE_3, color = "Yellow", label = "condition 3")
#plt.plot(collection, MSE_4, color = "#00ff7f", label = "condition 4")
plt.xlabel('Value of C')
plt.ylabel('Misclassification Error')
plt.legend()
plt.show()

# plot misclassification error vs k
plt.plot(collection, MSE_1, color = "red", label = "condition 1")
#plt.plot(collection, MSE_2, color = "blue", label = "condition 2")
plt.plot(collection, MSE_3, color = "Yellow", label = "condition 3")
#plt.plot(collection, MSE_4, color = "#00ff7f", label = "condition 4")
plt.xlabel('Value of C')
plt.ylabel('Misclassification Error')
plt.legend()
plt.xlim(-0.001, 10)
plt.ylim(0.1, 0.105)
plt.show()



# TESTING THE LOGISTIC REGRESSION MODEL
min_max_scaler = preprocessing.MinMaxScaler()
x_LR_train_scaled = min_max_scaler.fit_transform(x_LR_train)  # fits the scaler and then transforms the tng data
x_LR_test_scaled = min_max_scaler.transform(x_LR_test) # transforms the test  data (using the fitted scaler calculated in the previous step)

clfLR = LogisticRegression(penalty='l1', C=0.10)

fitLR = clfLR.fit(x_LR_train_scaled, y_LR_train)
predicted = fitLR.predict(x_LR_test_scaled)

print "***[Logisitic Regression Results]***"
print
print confusion_matrix(y_LR_test, predicted)
print
print "accuracy_score: {0}".format(accuracy_score(y_LR_test, predicted))
print
print classification_report(y_LR_test, predicted)

