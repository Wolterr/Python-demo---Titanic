get_ipython().magic('matplotlib inline')
# Scientific computing
import numpy as np

# Data frames (R-style)
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning packages
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Read data
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')
test_y = pd.read_csv('./data/gender_submission.csv').Survived

train_data.head(10)

train_data.info()
print("--------------")
test_data.info()


# As shown we have 891 entries in the training data and 418 entries in the test data. However, the info also shows there are missing values in the age column. For test there is also a missing Fare value.

grouped = train_data.groupby('Survived')
grouped.count()


#From the breakdown above we see the our target class (Survived) is relatively balanced over the training set. We also again see missing values.
#
#There are some columns (Name, Ticket, Embarked) that probably are not as strong for our prediction.
#They are also fairly difficult to transform into something more useful. Lets discard these.
#
#After we get rid of the columns specified above, we still have to deal with the NaN values we found.

train_data.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1, inplace=True)
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
train_data.Age = train_data.Age.round(decimals=0)
train_data.head(10)

test_data.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1, inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
test_data.Age = test_data.Age.round(decimals=0)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
test_data.info()


#In our final preparation step we turn the 'Sex' column into a dummy column with 0 being male and 1 being female.

#train_data.Pclass = train_data.Pclass.astype('category')
#train_data.Sex = train_data.Sex.astype('category')
train_data['Sex'] = np.where(train_data['Sex'] == 'female', 1, 0)

#test_data.Pclass = test_data.Pclass.astype('category')
#test_data.Sex = test_data.Sex.astype('category')
test_data['Sex'] = np.where(test_data['Sex'] == 'female', 1, 0)
test_data.head(10)


#Lets start looking at our data now we have cleaned it. 
#Do we need any other steps to handle correlations before we train a model?
correlations = train_data.corr()
print(correlations['Survived'])

plt.figure(figsize=(10,10))
sns.heatmap(correlations,linewidths=0.25, square=True, cbar_kws={'shrink' : .6}, annot=True, vmin=0, vmax=1)
plt.title("Heatmap over de correlaties tussen de variabelen")


#It appears are variables are not strongly correlated to each other.
#From the list of how the variables are correlated to the Survived label we can get the feeling a model might work quite well.
#
#As a last step before we start training a model we will plot the pairwise distribution between the variables.
#Note: This only works because we only have a few columns. With more columns the figures look terrible.
sns.pairplot(train_data)


# Train the model 
train_x = train_data.drop('Survived', axis=1, inplace=False)
train_y = train_data.Survived

classifier = tree.DecisionTreeClassifier()
classifier.fit(train_x, train_y)


# Check score on training set
cm_train = confusion_matrix(train_y, classifier.predict(train_x))
auc_train = roc_auc_score(train_y, classifier.predict(train_x))
f1_train = f1_score(train_y, classifier.predict(train_x))
print(cm_train)
print('------------------------------')
print('AUC      : {}'.format(auc_train))
print('F1 Score : {}'.format(f1_train))


# Create predictions on the test set
predictions = classifier.predict(test_data)


# Score on the test outcomes
cm_test = confusion_matrix(test_y, predictions)
f1_test = f1_score(test_y, predictions)
auc_test = roc_auc_score(test_y, predictions)
print(cm_test)
print('------------------------------')
print('AUC      : {}'.format(auc_test))
print('F1 Score : {}'.format(f1_test))


# Possibility to save the tree to file, load it, turn it into PNG and show it here. Didn't work on work laptop.
# Save decision tree to file, reload it and show in a plot
#with open("dt.dot", 'w') as f:
#    export_graphviz(classifier, out_file=f,
#                    feature_names=list(train_x))
#
#    pydot.graph_from_dot_data(dotfile.getvalue()).write_png(file_path)
#    i = misc.imread(file_path)
#    plt.imshow(i)


# There seems to be a lot of overfitting. Perhaps we can tweak the model to have some more constraints.
c2 = tree.DecisionTreeClassifier(max_depth = 15, max_leaf_nodes=10)
c2.fit(train_x, train_y)

f1_score(train_y, c2.predict(train_x))


# We now have a highly constrained decision tree. A much lower score on the train set means we're not fitting to all our training examples anymore. 
# 
# Lets see if this helps our test predictions any
pred = c2.predict(test_data)
print(confusion_matrix(test_y, pred))
print("------------------------------")
print("AUC Score = {}".format(roc_auc_score(test_y, pred)))
print("F1 Score  = {}".format(f1_score(test_y, pred)))


# We get a much better score with our constrained tree. Overfitting indeed was the problem.
# 
# Finally lets try a Random Forest to see if a whole set of (constrained) decision trees does better then a single tree.
forest = RandomForestClassifier(n_estimators=100, max_leaf_nodes=10)
forest.fit(train_x, train_y)
y_pred = forest.predict(test_data)
print("Accuracy score = {}".format(forest.score(test_data, test_y)))
print("F1 score       = {}".format(f1_score(test_y, y_pred)))
print("AUC score      = {}".format(roc_auc_score(test_y,y_pred)))
print("---------")
print(confusion_matrix(test_y, y_pred))

