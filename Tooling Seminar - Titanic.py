
# coding: utf-8

# In[1]:

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
from sklearn.model_selection import train_test_split


# In[3]:

# Read data
train_data = pd.read_csv('./train.csv')

train_data, test_data = train_test_split(train_data, test_size = 0.2)
test_y = test_data.Survived
test_data.drop('Survived', axis=1, inplace=True)

train_data.head(10)


# In[4]:

train_data.info()
print("--------------")
test_data.info()


# We hebben dus 891 entries. De meeste data lijkt compleet, maar bij leeftijd missen er een aantal waardes. Voor test is dit hetzelfde, maar mist er ook een fare waarde.

# In[5]:

grouped = train_data.groupby('Survived')
grouped.count()


# Als we de data groeperen op onze doel-variabele, zien we dat de klassen redelijk gebalanceerd zijn.

# We hebben dus redelijk gebalanceerde aantallen over de doel-variabele en de data is ook vrij goed (weinig missende getallen).
# 
# Wel hebben we een aantal categoriën die minder zeggen (naam, id, kaartnummer, waar ze ingestapt zijn). Deze variabelen kunnen we uit de set gooien.
# 
# Daarna moeten we de NAN's in de age kolom nog aanpakken. 
# 
# We gaan er hier even van uit dat dit hetzelfde is voor de test data.

# In[6]:

train_data.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1, 
                inplace=True)
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
train_data.Age = train_data.Age.round(decimals=0)
train_data.head(10)


# In[7]:

test_data.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1, 
               inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
test_data.Age = test_data.Age.round(decimals=0)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
test_data.info()


# Als laatste voorbereiding maken we van een aantal variablen categorische variabelen.

# In[8]:

#train_data.Pclass = train_data.Pclass.astype('category')
#train_data.Sex = train_data.Sex.astype('category')
train_data['Sex'] = np.where(train_data['Sex'] == 'female', 1, 0)

#test_data.Pclass = test_data.Pclass.astype('category')
#test_data.Sex = test_data.Sex.astype('category')
test_data['Sex'] = np.where(test_data['Sex'] == 'female', 1, 0)
test_data.head(10)


# Nu kunnen we gaan kijken naar de data

# In[9]:

correlations = train_data.corr()
print(correlations['Survived'])

plt.figure(figsize=(10,10))
sns.heatmap(correlations,linewidths=0.25, square=True, 
            cbar_kws={'shrink' : .6}, annot=True, vmin=0, vmax=1)
plt.title("Heatmap over de correlaties tussen de variabelen")


# Het lijkt er op dat er niet veel variabelen zijn die sterk met elkaar gecorreleerd zijn. Hier hoeven we dus niet veel voorwerk te doen.
# 
# Laten we als laatste eens kijken naar hoe de variabelen onderling verdeeld zijn. 

# In[10]:

sns.pairplot(train_data)


# In[11]:

# Train the model 
train_x = train_data.drop('Survived', axis=1, inplace=False)
train_y = train_data.Survived

classifier = tree.DecisionTreeClassifier()
classifier.fit(train_x, train_y)


# In[12]:

# Check score on training set
cm_train = confusion_matrix(train_y, classifier.predict(train_x))
auc_train = roc_auc_score(train_y, classifier.predict(train_x))
f1_train = f1_score(train_y, classifier.predict(train_x))
print(cm_train)
print('------------------------------')
print('AUC      : {}'.format(auc_train))
print('F1 Score : {}'.format(f1_train))


# In[13]:

# Create predictions on the test set
predictions = classifier.predict(test_data)


# In[14]:

# Score on the test outcomes
cm_test = confusion_matrix(test_y, predictions)
f1_test = f1_score(test_y, predictions)
auc_test = roc_auc_score(test_y, predictions)
print(cm_test)
print('------------------------------')
print('AUC      : {}'.format(auc_test))
print('F1 Score : {}'.format(f1_test))


# In[ ]:

# Save decision tree to file, reload it and show in a plot
#with open("dt.dot", 'w') as f:
#    export_graphviz(classifier, out_file=f,
#                    feature_names=list(train_x))
#
#    pydot.graph_from_dot_data(dotfile.getvalue()).write_png(file_path)
#    i = misc.imread(file_path)
#    plt.imshow(i)


# There seems to be a lot of overfitting. Perhaps we can tweak the model to have some more constraints.

# In[15]:

# Look at our current tree
classifier.get_params


# In[16]:

c2 = tree.DecisionTreeClassifier(max_depth = 15, max_leaf_nodes=10)
c2.fit(train_x, train_y)

f1_score(train_y, c2.predict(train_x))


# We now have a highly constrained decision tree. A much lower score on the train set means we're not fitting to all our training examples anymore. 
# 
# Lets see if this helps our test predictions any

# In[17]:

pred = c2.predict(test_data)
print(confusion_matrix(test_y, pred))
print("------------------------------")
print("AUC Score = {}".format(roc_auc_score(test_y, pred)))
print("F1 Score  = {}".format(f1_score(test_y, pred)))


# We get a much better score with our constrained tree. Overfitting indeed was the problem.
# 
# Finally lets try a Random Forest to see if a whole set of (constrained) decision trees does better then a single tree.

# In[74]:

forest = RandomForestClassifier(n_estimators=100, max_features = 'log2', 
                                max_leaf_nodes = 15)
forest.fit(train_x, train_y)
y_pred = forest.predict(test_data)
print("Accuracy score = {}".format(forest.score(test_data, test_y)))
print("F1 score       = {}".format(f1_score(test_y, y_pred)))
print("AUC score      = {}".format(roc_auc_score(test_y,y_pred)))
print("---------")
print(confusion_matrix(test_y, y_pred))


# In[ ]:



