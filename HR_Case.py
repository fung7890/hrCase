import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import ShuffleSplit


df = pd.read_csv(r'C:\Users\owner\Dropbox\MBA Final Term\Analytics\HR_DATA.csv') # need to specify your file path

del df['Candidate.Ref'] #deleted column
del df['X'] #deleted column
del df['Unnamed: 0'] #deleted column
del df['Pecent.hike.expected.in.CTC'] #deleted column
del df['Percent.hike.offered.in.CTC'] #deleted column


df['Duration.to.accept.offer'].fillna(df['Duration.to.accept.offer'].mean(), inplace = True) #impute with means of that column
df['Percent.difference.CTC'].fillna(df['Percent.difference.CTC'].mean(), inplace = True) #impute with means of that column

df = pd.get_dummies(df, columns= ['DOJ.Extended', 'Joining.Bonus', 'Candidate.relocate.actual', 'Gender',
'Offered.band','Candidate.Source','LOB', 'Location','Status'], drop_first=True) #create dummies, drop one category in each so no colinear issues


X = df.drop(['Status_Not Joined'], 1) #X variables, dropped y
y = df['Status_Not Joined'] #y variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #cross_validation shuffles data and outputs training and testing data, random_state the same start


logit_model = sm.Logit(y, X).fit() #print summary statistics of variables
print (logit_model.summary() )


alg = sklearn.linear_model.LogisticRegression() #LogisticRegression selection
x = alg.fit(X_train, y_train) #running LogisticRegression on training set
test_score = alg.score(X_test, y_test) #finding out predictability score on testing set

print (test_score)
