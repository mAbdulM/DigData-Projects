#DigData - prediction Model
#___________________________________IMPORT MODULES______________________________
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from treeinterpreter import treeinterpreter as ti
import warnings
warnings.filterwarnings("ignore")

#___________________________________SETUP_____________________________________
#---------------pandas
pd.set_option("display.max_columns", 200)
pd.options.display.width = 200
#---------------matplotlib
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True


#___________________________________LOAD IN DATABASE___________________________

df = pd.read_csv('Capital One - Data Set (STEP UP).csv').drop("CUSTOMER_ID", axis = 1)

#---------------find null values

print('is there a null value in the database: ' + str(df.isnull().values.any()))
print('\n show each column and whether thay have a null value or not\n')
print(df.isna().any())
print('total number of null values in the dataBase:' + str(df.isnull().sum().sum()))

#---------------get number of rows with null values

nullRecords = df[df.isna().any(axis = 1)]
print('\n number of rows with null values: ' + str(len(nullRecords))) #- not too much rows relative to dataset so remove rows

#---------------drop rows

df_no_null = df.dropna()
print('\ndoes the database have anymore null values: ' + str(df_no_null.isnull().values.any()))# no missing values

#__________________________________Train,Test and one-hot encoding____________

#---------------train test
X = df_no_null.drop('SPEND_M3_TOTAL', axis=1).copy()
y = df_no_null['SPEND_M3_TOTAL'].copy()


#---------------One-Hot Encoding
X_encoded = pd.get_dummies(X, columns = ['REGION',
                                          'OCCUPATION',
                                          'CARD_COLOUR'])

#---------------split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y,train_size = 0.7, random_state=45)

##_________________________________Dummy Model To Beat_______________________
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)
dy_pred = dummy_regr.predict(X_test)
dummy_perf = (str(mean_absolute_error(y_test,dy_pred)), str(r2_score(y_test,dy_pred)),str(np.sqrt(mean_squared_error(y_test, dy_pred))))
print('\n DummyRegressor Baseline ')
print('MAE: ', mean_absolute_error(y_test,dy_pred))
print('R2: ', r2_score(y_test,dy_pred))
print('RMSE: ',np.sqrt(mean_squared_error(y_test, dy_pred)))



##_________________________________My Model__________________________________

dt = RandomForestRegressor(random_state=15, n_estimators = 200)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)

predAccuracy = (str(mean_absolute_error(y_test,y_pred)), str(r2_score(y_test,y_pred)),str(np.sqrt(mean_squared_error(y_test, y_pred))))
print('\n RandomForestRegressor')
print('MAE: ', predAccuracy[0] )
print('R2: ', predAccuracy[1])
print('RMSE: ', predAccuracy[2])

yt_pred = dt.predict(X_train)

print('\n RandomForestRegressor')
print('MAE: ', mean_absolute_error(y_train,yt_pred))
print('R2: ', r2_score(y_train,yt_pred))
print('RMSE: ',np.sqrt(mean_squared_error(y_train,yt_pred)))

objects = ('MAE', 'R2', 'RMSE')
y_pos = np.arange(len(objects))
RF_performance = predAccuracy
##
##X = ['MAE','RMSE']
##RF_gr = [float(RF_performance[0]),float(RF_performance[2])]
##dummy_gr = [float(dummy_perf[0]),float(dummy_perf[2])]
##  
##X_axis = np.arange(len(X))
##  
##plt.bar(X_axis - 0.2, RF_gr, 0.4, label = 'RandomForestRegressor')
##plt.bar(X_axis + 0.2, dummy_gr, 0.4, label = 'DummyRegessor')
##  
##plt.xticks(X_axis, X)
##plt.xlabel("accuracy Methods")
##plt.ylabel("Error Value")
##plt.legend()
##plt.show()
####--
##plt.figure()
##X = ['R2']
##RF_gr = [float(RF_performance[1])]
##dummy_gr = [float(dummy_perf[1])]
##  
##X_axis = np.arange(len(X))
##  
##plt.bar(X_axis - 0.2, RF_gr, 0.4, label = 'RandomForestRegressor')
##plt.bar(X_axis + 0.2, dummy_gr, 0.4, label = 'DummyRegessor')
##plt.ylim([-1,1])  
##plt.xticks(X_axis, X)
##plt.xlabel("accuracy Methods")
##plt.ylabel("score")
##plt.legend()
##plt.show()



##_________________________________interperation__________________________________
pred, bias, contributions = ti.predict(dt, X_train)

####print(pred)
##print(contributions)
res1 = pd.DataFrame(pred, columns = ['prediction'])
res2 = pd.DataFrame(contributions, columns = X_test.columns)
results = pd.concat([res1,res2],axis = 1)


eh = pd.DataFrame(yt_pred)
x_t = X_train.reset_index()
he = pd.concat([eh,x_t], axis = 1)

print(len(eh), len(x_t), len(he))

print(he.head(50))
print(he.tail(50))
print(results.head(50))
print(results.tail(50))

for i in x_t:
    if i != 'index':
        
        plt.figure()
        plt.xlabel(i)
        plt.ylabel('contribution')
        plt.title('how ' + i+ ' contributes to prediction')
        plt.scatter(x_t[i],results[i],alpha = 0.1, s= 5, color="red")
    print(i)


plt.show()

results.sort_values(by = 'prediction', ascending = False, inplace = True, ignore_index = True)
Pfindings = results.head(50).mean(axis = 0)
nfindings = results.tail(50).mean(axis = 0)
print('\n average prediction and impact of all attributes for top 1000, who spent the MOST on month 3:\n')
print(Pfindings)
print('\n average prediction and impact of all attributes for bottom 1000, who spent the LEAST on month 3:\n')
print(nfindings)
##print(y_pred)
##print(results)

fig = plt.figure()
b = X_train
X = X_train.columns
print(X)
print(X_train.columns)
RF_gr = []
for v in range(1, len(Pfindings)):
    RF_gr.append(Pfindings.values[v])
    print(v, Pfindings.values[v])
print(RF_gr)
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, RF_gr, 0.4)

  
plt.xticks(X_axis, X)
plt.xlabel("Attributes")
plt.ylabel("impact")
#plt.legend()
fig.autofmt_xdate()


fig = plt.figure()
b = X_train
X = X_train.columns
print(X)
print(X_train.columns)
RF_gr = []
for v in range(1, len(nfindings)):
    RF_gr.append(nfindings.values[v])
    print(v, nfindings.values[v])
print(RF_gr)
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, RF_gr, 0.4)

  
plt.xticks(X_axis, X)
plt.xlabel("Attributes")
plt.ylabel("impact")
#plt.legend()
fig.autofmt_xdate()


#_________________________________Plot Feature Importance____________________
 
##
for x in X_encoded:
    print(x)
    plt.figure()
    plt.xlabel(x)
    plt.ylabel('SPEND_M3_TOTAL')
    plt.scatter(X_encoded[x],y, alpha = 0.01, s= 5, color="blue")
##
##feature_names = [X_encoded.columns[i] for i in range(X_encoded.shape[1])]
##print([X_encoded.columns[i] for i in range(X_encoded.shape[1])])
##
##importances = dt.feature_importances_
##std = np.std([tree.feature_importances_ for tree in dt.estimators_], axis=0)
##forest_importances = pd.Series(importances, index=feature_names)
##
##fig, ax = plt.subplots()
##forest_importances.plot.bar(yerr=std, ax=ax)
##ax.set_title("Feature importances using MDI")
##ax.set_ylabel("Mean decrease in impurity")
##
##
##
##fig.tight_layout()   
##
plt.show()
##print('done')
