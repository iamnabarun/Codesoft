#!/usr/bin/env python
# coding: utf-8

# Importing Modules/Dependencies for Data Handling

# In[56]:


import numpy as np
import pandas as pd


# Reading Training and Testing Data Sets (Downloaded from https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTrain.csv)

# In[57]:


credit_data = pd.read_csv('fraudTrain.csv')
test_data = pd.read_csv('fraudTest.csv')


# In[58]:


# first 5 rows of the dataset
credit_data.head()

# last 5 rows of the dataset
credit_data.tail()


# In[59]:


# Getting info on the data n the data set
credit_data.info()


# In[60]:


def preprocessing(data) :
  # deleting useless columns
  del_col = ['merchant','first','last','street','zip','unix_time','Unnamed: 0','trans_num','cc_num']
  data.drop(columns=del_col,inplace=True)

  # converting data-time features from object type to Numerical value
  data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
  data['trans_date'] = data['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
  data['trans_date'] = pd.to_datetime(data['trans_date'])
  data['dob'] = pd.to_datetime(data['dob'])

  data["age"] = (data["trans_date"] - data["dob"]).dt.days

  data['trans_month'] = data['trans_date'].dt.month
  data['trans_year'] = data['trans_date'].dt.year

  # using one-hot encoding for categorical data features
  data['gender'] = data['gender'].apply(lambda x : 1 if x=='M' else 0)
  data['gender'] = data['gender'].astype(int)
  data['lat_dis'] = abs(data['lat']-data['merch_lat'])
  data['long_dis'] = abs(data['long']-data['merch_long'])
  data = pd.get_dummies(data,columns=['category'])
  data = data.drop(columns=['city','trans_date_trans_time','state','job','merch_lat','merch_long','lat','long','dob','trans_date'])

  # returning the preprocessed dataset
  return data

# performing data p


# In[61]:


# performing data preprocessing on credit_data (Training Data) and test_data
credit_data = preprocessing(credit_data.copy())
test_data = preprocessing(test_data.copy())


# In[62]:


# Checking data after preprocessing
credit_data.head()


# In[63]:


# Checking the data features, every feature is in numeric data type
credit_data.info()


# In[64]:


# creating correlation matrix
correlation_matrix = credit_data.corr()


# In[65]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[66]:


# plotting the correation matrix
plt.figure(figsize=(14, 14))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Observation : All the data features have low correlation making every fature less and equally important for the prediction,
# except the amount of transaction (amt)


# In[67]:


# Counting the number of legit and fraud transaction data in dataset
credit_data['is_fraud'].value_counts()


# In[68]:


# separating the data for analysis
legit = credit_data[credit_data.is_fraud == 0]
fraud = credit_data[credit_data.is_fraud == 1]


# In[69]:


# priting their shape
print(legit.shape)
print(fraud.shape)


# In[70]:


# statistical measures of the legit transaction data
legit.amt.describe()


# In[71]:


# statistical measures of the frad transaction data
fraud.amt.describe()


# In[72]:


# compare the values for both transactions
credit_data.groupby('is_fraud').mean()


# In[73]:


# creating a sample legit data set of same size as that of fraud dataset
legit_sample = legit.sample(n=7506)


# In[74]:


# joining the legit and fraud data set
new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[75]:


# checking the new dataset
new_dataset.head()


# In[76]:


new_dataset.tail()


# In[77]:


# checking the count of legit and fraud transaction datasets in new dataset
new_dataset['is_fraud'].value_counts()


# In[78]:


X_train = new_dataset.drop(columns='is_fraud', axis=1)
Y_train = new_dataset['is_fraud']

X_test = test_data.drop(columns='is_fraud', axis=1)
Y_test = test_data['is_fraud']


feature_columns = X_train.columns


# In[79]:


# printing the features
print(X_train)


# In[80]:


# printing the target
print(Y_train)


# In[81]:


print(X_test)


# In[82]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[121]:


from sklearn.preprocessing import StandardScaler


X_train = pd.DataFrame(X_train, columns=feature_columns)
X_test = pd.DataFrame(X_test, columns=feature_columns)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[107]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Train model
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train_scaled, Y_train)

# Prediction (USE SCALED DATA)
y_pred_logistic = logistic_regression.predict(X_test_scaled)

# Accuracy
accuracy_logistic = accuracy_score(Y_test, y_pred_logistic)
print("Accuracy:", accuracy_logistic)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred_logistic))

# Classification Report
print("\nClassification Report:")
print(classification_report(Y_test, y_pred_logistic))

# ROC-AUC (USE SCALED DATA)
y_prob = logistic_regression.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, thresholds = roc_curve(Y_test, y_prob)
auc_score = roc_auc_score(Y_test, y_prob)

plt.plot(fpr, tpr)
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("AUC Score:", auc_score)


# In[106]:


from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42
)

DecisionTree.fit(X_train_scaled, Y_train)

y_pred_dt = DecisionTree.predict(X_test_scaled)

print("Accuracy:", accuracy_score(Y_test, y_pred_dt))
print(confusion_matrix(Y_test, y_pred_dt))
print(classification_report(Y_test, y_pred_dt))


# In[99]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train_scaled, Y_train)

y_pred_rf = random_forest.predict(X_test_scaled)

print("Accuracy:", accuracy_score(Y_test, y_pred_rf))
print(confusion_matrix(Y_test, y_pred_rf))
print(classification_report(Y_test, y_pred_rf))


# In[108]:


print("\nClassification Report for Logistic Regression:\n", classification_report(Y_test, y_pred_logistic))
print("\nClassification Report for Decision Tree:\n", classification_report(Y_test, y_pred_dt))
print("\nClassification Report for Random Forest:\n", classification_report(Y_test, y_pred_rf))


# In[109]:


from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

# Convert classification reports to dictionary
log_report = classification_report(Y_test, y_pred_logistic, output_dict=True)
dt_report = classification_report(Y_test, y_pred_dt, output_dict=True)
rf_report = classification_report(Y_test, y_pred_rf, output_dict=True)

# Calculate ROC-AUC using SCALED test data
log_auc = roc_auc_score(Y_test, logistic_regression.predict_proba(X_test_scaled)[:, 1])
dt_auc = roc_auc_score(Y_test, DecisionTree.predict_proba(X_test_scaled)[:, 1])
rf_auc = roc_auc_score(Y_test, random_forest.predict_proba(X_test_scaled)[:, 1])

# Create comparison table
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Precision (Fraud)": [
        log_report['1']['precision'],
        dt_report['1']['precision'],
        rf_report['1']['precision']
    ],
    "Recall (Fraud)": [
        log_report['1']['recall'],
        dt_report['1']['recall'],
        rf_report['1']['recall']
    ],
    "F1-Score (Fraud)": [
        log_report['1']['f1-score'],
        dt_report['1']['f1-score'],
        rf_report['1']['f1-score']
    ],
    "ROC-AUC": [log_auc, dt_auc, rf_auc]
})

comparison


# In[120]:


# Create Input Data
input_data = pd.DataFrame({
    'Unnamed: 0': [1],
    'trans_date_trans_time': ['2022-01-01 12:00:00'],
    'cc_num': [1234567890123456],
    'merchant': ['ExampleMerchant'],
    'category': ['ExampleCategory'],
    'amt': [150.0],
    'first': ['John'],
    'last': ['Doe'],
    'gender': ['M'],
    'street': ['123 Example St'],
    'city': ['ExampleCity'],
    'state': ['EX'],
    'zip': [12345],
    'lat': [40.7128],
    'long': [-74.0060],
    'city_pop': [100000],
    'job': ['ExampleJob'],
    'dob': ['1990-01-01'],
    'trans_num': ['ExampleTransNum'],
    'unix_time': [1609459200],
    'merch_lat': [40.730610],
    'merch_long': [-73.935242],
})

# Preprocess
input_data = preprocessing(input_data.copy())

# Add Missing Columns
for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Ensure Correct Order
input_data = input_data[feature_columns]

# Scale using same scaler
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = random_forest.predict(input_data_scaled)

if prediction[0] == 0:
    print("Legitimate Transaction")
else:
    print("Fraudulent Transaction")


# In[ ]:




