import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("Most Runs - 2022.csv")

# pd.set_option("display.max_columns",None)
# pd.set_option("display.max_rows",None)
# print(data)
#
#
# data = pd.read_csv("Most Runs - 2022.csv")
# information = data.info()
# print(information)

# data = pd.read_csv("Most Runs - 2022.csv")
# description = data.describe()
# print(description)
# #
finding_null_row = data.isnull().sum()
print(finding_null_row)

finding_null_row = data.fillna(10)
print(finding_null_row)


# label_obj = LabelEncoder()
# data['Player'] = label_obj.fit_transform(data['Player'])
# print(data['Player'])

label_obj = OneHotEncoder(sparse_output=False)
encoded_column = label_obj.fit_transform(data[['Player']])
print(encoded_column)

column_names_encoded_data = label_obj.get_feature_names_out(["Player"])
print(column_names_encoded_data)

new_df = pd.DataFrame(encoded_column,columns=column_names_encoded_data)
print(new_df)

data.drop(columns=['Player'],inplace=True)

final_df = pd.concat([data,new_df],axis=1)
print(final_df)
print(final_df.head())



# df = pd.read_csv("Most Runs - 2022.csv")
# x= df['Player']
# y = df['Runs']
# sns.barplot(x=x,y=y, label= 'Comparison', color= 'violet')
# sns.scatterplot(x=x,y=y)
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.xticks(rotation=45,fontsize=3)
# plt.legend()
# plt.grid()
# plt.show()
#
# plt.boxplot(y)
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV

# df= pd.read_csv("Most Runs - 2022.csv")
# print(df.head())

data_columns = final_df.columns
print(data_columns)


# x= final_df[['POS','Mat','Inns', 'NO', 'HS', 'Avg', 'BF']]
# y =final_df['Runs']
#
x= final_df[['Player_Vaibhav Arora', 'Player_Varun Chakaravarthy',
      'Player_Venkatesh Iyer', 'Player_Vijay Shankar', 'Player_Virat Kohli',
      'Player_Wanindu Hasaranga', 'Player_Washington Sundar']]

y =final_df['Runs']


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42)
model = RandomForestClassifier(random_state=42)

# model = RandomForestClassifier()
# model.fit(x_train,y_train)
# y_prediction = model.predict(x_test)
# print(y_prediction)
#
# mean_accuracy = accuracy_score(y_test,y_prediction)
# print(mean_accuracy)

hyper_parameter_input = {
    "n_estimators" :[50,100,20],
    "max_depth" :[None, 10,20,30],
    "min_samples_split" : [2,5,10],
    "min_samples_leaf":[1,2,4],
    "max_features" : ["auto","sqrt","log2"]
}

grid_model = GridSearchCV(model,param_grid=hyper_parameter_input,cv=5)
grid_model.fit(x_train,y_train)

best_paramater = grid_model.best_params_
best_estimater = grid_model.best_estimator_

y_prediction = best_estimater.predict(x_test)
print(y_prediction)

mean_accuracy = accuracy_score(y_test,y_prediction)
print(mean_accuracy)



