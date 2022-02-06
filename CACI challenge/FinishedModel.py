import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition, datasets
#from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.dummy import DummyClassifier



def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, cmap="YlGnBu", xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show()

rcParams['figure.figsize'] = 14, 7
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

pd.set_option("display.max_columns", 200)
pd.options.display.width = 200

target = "Interested in sustainable range"
unImportant = ['Gender', 'Age','Customer ID']

df = pd.read_csv("CACI Dataset 1.csv")
genders = pd.get_dummies(df.Gender, prefix = "Gender")
ages = pd.get_dummies(df.Age, prefix = "Age")
df = df.drop(columns = unImportant)
df = pd.concat([df,ages,genders], axis = 1)

X = df.drop(columns = target).values
y = (df[target]).astype(int)

dt = DecisionTreeClassifier(random_state=15, criterion = 'entropy', max_depth = 10)
dt.fit(X,y)


fi_col = []
fi = []

for i,column in enumerate(df.drop(target, axis = 1)):
    print('The feature importance for {} is : {}'.format(column, dt.feature_importances_[i]))
    
    fi_col.append(column)
    fi.append(dt.feature_importances_[i])




fi_df = zip(fi_col, fi)
fi_df = pd.DataFrame(fi_df, columns = ['Feature','Feature Importance'])



# Ordering the data
fi_df = fi_df.sort_values('Feature Importance', ascending = False).reset_index()

# Creating columns to keep
columns_to_keep = fi_df['Feature'][0:42]
print("\n Columns to keep")
print(columns_to_keep)

x = df[columns_to_keep].values
y = (df[target]).astype(int)


X_train, X_test, y_train, y_test = train_test_split(x, y,train_size = 0.7, random_state=92)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,train_size = 0.9, random_state=92)



ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)


#LOGISTIC REGRESSION
model = LogisticRegression(random_state = 92, max_iter =5000, solver = 'lbfgs')
model.fit(X_train_scaled, y_train)

ypretr = model.predict(X_train)
ypre = model.predict(X_test)

ypreTePob =  model.predict_proba(X_train)

print(classification_report(y_train,ypretr))
print('training score: ')
print(model.score(X_train, y_train))
print('testing score: ')
print(model.score(X_test, y_test))

#confusion matrix
cm = confusion_matrix(y_train, ypretr)
cm_norm = cm / cm.sum(axis=1).reshape(-1,1)

plot_confusion_matrix(cm_norm, classes = model.classes_, title='Confusion matrix')

# Calculating False Positives (FP), False Negatives (FN), True Positives (TP) & True Negatives (TN)

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)
print("The True Positive Rate is:", TPR)

# Precision or positive predictive value
PPV = TP / (TP + FP)
print("The Precision is:", PPV)

# False positive rate or False alarm rate
FPR = FP / (FP + TN)
print("The False positive rate is:", FPR)


# False negative rate or Miss Rate
FNR = FN / (FN + TP)
print("The False Negative Rate is: ", FNR)



##Total averages :
print("")
print("The average TPR is:", TPR.sum()/2)
print("The average Precision is:", PPV.sum()/2)
print("The average False positive rate is:", FPR.sum()/2)
print("The average False Negative Rate is:", FNR.sum()/2)

# Running Log loss on training
print("The Log Loss on Training is: ", log_loss(y_train, ypreTePob))

# Running Log loss on testing
pred_proba_t = model.predict_proba(X_test)
print("The Log Loss on Testing Dataset is: ", log_loss(y_test, pred_proba_t))


C_List = np.geomspace(1e-5, 1e5, num=20)
CA = []
Logarithmic_Loss = []

for c in C_List:
    log_reg2 = LogisticRegression(random_state=10, solver = 'lbfgs', C=c, max_iter = 5000)
    log_reg2.fit(X_train, y_train)
    score = log_reg2.score(X_test, y_test)
    CA.append(score)
    print("The CA of C parameter {} is {}:".format(c, score))
    pred_proba_t = log_reg2.predict_proba(X_test)
    log_loss2 = log_loss(y_test, pred_proba_t)
    Logarithmic_Loss.append(log_loss2)
    print("The Logg Loss of C parameter {} is {}:".format(c, log_loss2))
    print("")
    ypretr = log_reg2.predict(X_train)
    cm = confusion_matrix(y_train, ypretr)
    cm_norm = cm / cm.sum(axis=1).reshape(-1,1)

    plot_confusion_matrix(cm_norm, classes = model.classes_, title='Confusion matrix')

CA2 = np.array(CA).reshape(20,)
Logarithmic_Loss2 = np.array(Logarithmic_Loss).reshape(20,)

# zip
outcomes = zip(C_List, CA2, Logarithmic_Loss2)

#df
df_outcomes = pd.DataFrame(outcomes, columns = ["C_List", 'CA2','Logarithmic_Loss2'])



# Ordering the data (sort_values)
df_outcomes = df_outcomes.sort_values("Logarithmic_Loss2", ascending = True).reset_index()

#Dummy Classifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
score = dummy_clf.score(X_test, y_test)

pred_proba_t = dummy_clf.predict_proba(X_test)
log_loss2 = log_loss(y_test, pred_proba_t)

print('Dummy Classifier results:')
print("Testing Acc:", score)
print("Log Loss:", log_loss2)

log_reg3 = LogisticRegression(random_state=10, solver = 'lbfgs', C=6.158482, max_iter=5000)
log_reg3.fit(X_train, y_train)

score = log_reg3.score(X_valid, y_valid)

pred_proba_t = log_reg3.predict_proba(X_valid)
log_loss2 = log_loss(y_valid, pred_proba_t)

print('my results:')
print("M Testing Acc:", score)
print("M Log Loss:", log_loss2)
print(log_reg3.score(X_train, y_train))
print(log_reg3.score(X_test, y_test))

#Final prediction - use model to predict and order it.
raw_data = pd.read_csv("CACI Dataset 2.csv")
genders2 = pd.get_dummies(raw_data.Gender, prefix = "Gender")
ages2 = pd.get_dummies(raw_data.Age, prefix = "Age")
df2 = raw_data.drop(columns = unImportant)
df2 = pd.concat([df2,ages2,genders2], axis = 1)

a = df2[columns_to_keep].values
b = df2[columns_to_keep].values
suitInterest = pd.DataFrame(log_reg3.predict(a),columns = [target])

confidenceScore = pd.DataFrame(log_reg3.predict_proba(b),columns = ["ConScore0", "conScore1"])

results = pd.concat([raw_data,suitInterest,confidenceScore],axis=1)

results.drop(results[results[target].values == 0].index,inplace = True)
invList = results.sort_values(["ConScore0"], ascending = True).reset_index()

invList = invList.head(250)
print('customer IDs of who to invite')
print(invList['Customer ID'].values)


