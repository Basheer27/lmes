import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''Loading dataset and File Reading'''

df= pd.read_csv('bar_pass_prediction.csv')
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
print(df.head())
print(df.tail())
print(df.shape)

describe = df.describe()
print(describe)

information = df.info()
print(information)

finding_null_row = df.isnull().sum()
print(finding_null_row)

'''Data Visualization(Gender Distribution)'''

plt.figure(figsize=(10,8))
ax = sns.countplot(data=df, x= 'gender',label='Gender Count')
ax.bar_label(ax.containers[0])
plt.show()

# The number of males is higher than the number of females from the above chart

'''Grouping the required data for Visualization'''

gb = df.groupby('race').agg({'ugpa':'mean','zgpa':'mean','gpa':'mean'})
print(gb)

sns.heatmap(gb)
plt.title('Students marks comparison based on Race')
plt.show()

#from the above chart zgpa marks scored was too low compared with other remaining marks obtained

gb1 = df.groupby('ID').agg({'lsat':'mean','ugpa':'mean'})
print(gb1)

sns.heatmap(gb1)
plt.title('Students Marks Comparison')
plt.grid(True)
plt.show()

#From the above chart the Isat marks of students are better than the ugpa marks

'''Checking out for Outliers in Important Datas'''

sns.boxplot(data=df,x='lsat')
plt.show()
sns.boxplot(data=df,x='ugpa')
plt.show()
sns.boxplot(data=df,x='bar_passed')
plt.show()
sns.boxplot(data=df,x='dnn_bar_pass_prediction')
plt.show()


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score

data_columns = df.columns
print(data_columns)

x = df[['lsat','ugpa']]
y = df['gpa']

'''Model training and testing'''

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

'''Model fitting and prediction '''

model = RandomForestRegressor(random_state=42)

hyper_parameter_input = {
    "n_estimators" :[50,100,200],
    "max_depth" :[None, 10,20,30],
    "min_samples_split" : [2,5,10],
    "min_samples_leaf":[1,2,4],
    "max_features" : ["auto","sqrt","log2"]
}

grid_model = GridSearchCV(model,param_grid=hyper_parameter_input,cv=5)
grid_model.fit(X_train,y_train)
print(X_test)

# print(X_train)
# print(y_test)
# print(y_train)

best_paramater = grid_model.best_params_
best_estimater = grid_model.best_estimator_

'''Predicting whether or not a student will pass the bar, 
based on their Law School Admission Test (LSAT) score and undergraduate GPA.'''

y_prediction = best_estimater.predict(X_test)
print(y_prediction)

'''Model metric check'''

accuracy_score = r2_score(y_test,y_prediction)
print(accuracy_score)

#Final Result:

'''From the above prediction and Model Accuracy Score it shows that the Law students will 
pass the Law School'''