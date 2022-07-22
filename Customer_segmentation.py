# -*- coding: utf-8 -*-
"""



"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping

from Customer_segmentation_module import EDA
from Customer_segmentation_module import ModelDevelopment
from Customer_segmentation_module import ModelEvaluation

#%% Constant
CSV_PATH = os.path.join(os.getcwd(),'Datasets','Train.csv')
MMS_PATH_X = os.path.join(os.getcwd(),'Models','mms.pkl')
OHE_PATH = os.path.join(os.getcwd(),'Models','ohe.pkl')
LOGS_PATH = os.path.join(os.getcwd(),'Models','logs',datetime.datetime.now()
                          .strftime('%Y%m%d-%H%M%S'))
BEST_MODEL_PATH = os.path.join(os.getcwd(),'Models','best_model.h5')

#%% Step 1) Data Loading
df = pd.read_csv(CSV_PATH,na_values='')

#%% STEP 2) Data visualization
df.info()
ds = df.describe().T

# Drop id, days_since_prev_campaign_contact columns
df = df.drop(labels =['id','days_since_prev_campaign_contact'],axis=1)

# Categorical columns 
cat_col = list (df.columns[df.dtypes=='object'])
cat_col.append('num_contacts_prev_campaign')
cat_col.append('term_deposit_subscribed')

# Categorical vizualization
eda = EDA()
eda.countplot_graph(cat_col, df)

# Continous Features
con_col = list(df.columns[(df.dtypes=='int64') | (df.dtypes=='float64')])
con_col.remove('num_contacts_prev_campaign')
con_col.remove('term_deposit_subscribed')

# Continous Visualization
eda.displot_graph(con_col, df)

# Vizualize Relationships
df.groupby(['term_deposit_subscribed','education','marital']).agg(
    {'term_deposit_subscribed':'count'}).plot(kind='bar')
# Married with secondary education individual are unsubscibed to term deposit
df.groupby(['term_deposit_subscribed','month']).agg(
    {'term_deposit_subscribed':'count'}).plot(kind='bar')
# May are the month where the highest unsubsribed to term deposit
# Term deposit subscribed are teh same troughout the year
df.groupby(['term_deposit_subscribed','personal_loan'
            ]).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')
df.groupby(['term_deposit_subscribed','personal_loan','housing_loan'
            ]).agg({'term_deposit_subscribed':'count'}).plot(kind='bar')
df.groupby(['term_deposit_subscribed','personal_loan','housing_loan',
            'communication_type']).agg({'term_deposit_subscribed':'count'}
                                       ).plot(kind='bar')

#%% Step 3) Data Cleaning
# Change all the string into int by using label encoder
le = LabelEncoder()
for i in cat_col:
    df[i] = le.fit_transform(df[i])
    ENCODER_PATH = os.path.join(os.getcwd(),'Models', i + '_encoder.pkl')
    pickle.dump(le,open(ENCODER_PATH,'wb'))

# Checking Nans - There are a lot of NaNs
df.isna().sum()

# Checking Missing Value/ NaNs
msno.matrix(df)
msno.bar(df)

# Removing NaNs --->KNN Imputation
columns_names = df.columns
knn_i = KNNImputer()
df = knn_i.fit_transform(df) # return numpy array
df = pd.DataFrame(df) # to convert back into dataframe
df.columns = columns_names

#%% Step 4) Features Selection
# Continous vs Categorical--> LogisticRegression (must be in integers)
# To check correlation between continous data vs categorical data

y = df['term_deposit_subscribed']

selected_features = []

for i in con_col:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[i], axis=-1),y)
    print(i)
    print(lr.score(np.expand_dims(df[i], axis=-1),y))
    if lr.score(np.expand_dims(df[i],axis=-1),y) >= 0.6:
        selected_features.append(i)

print (selected_features)
# From con_col result, the correlation that > 0.6 are customer_age,balance,
# day_of_month,last_contact_duration,num_contacts_in_campaign

# To check correlation between categorical vs categorical data
for i in cat_col:
    print(i)
    cmx = pd.crosstab(df[i],y).to_numpy()
    print(eda.cramers_corrected_stat(cmx))
    if eda.cramers_corrected_stat(cmx) > 0.6:
        selected_features.append(i)

print(selected_features)
# From cat_col result, the correlation that > 0.6 
# are only term_deposit_subscribed which will be our y.

# As conclusion, the selected features are customer_age,balance,
# day_of_month,last_contact_duration,num_contacts_in_campaign

df = df.loc[:,selected_features]
X = df.drop(labels='term_deposit_subscribed',axis=1)
y = df['term_deposit_subscribed']

#%% Step 5) Data Preprocessing
# MinMaxScaler (MMS)
mms = MinMaxScaler()
X = mms.fit_transform(X)

# Save MMS file
with open(MMS_PATH_X,'wb') as file:
    pickle.dump(mms,file)

# OneHotEncoder(OHE)- Deep Learning
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y,axis=-1))

# Save OHE file
with open(OHE_PATH,'wb') as file:
    pickle.dump(OHE_PATH,file)

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                  test_size=0.3,
                                                  random_state=123)

#%% Model Development (Deep Learning Approach)
md = ModelDevelopment()

nb_class = len(np.unique(y,axis=0)) # to check the length of the class of y
X_shape = np.shape(X_train)[1:] # shape of X

model = md.simple_dl_model(X_shape, nb_class)

# To plot model
plot_model(model,show_shapes=True,show_layer_names=True)

# To wrap the container
model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=['acc'])

#%% Tensorboard
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)

# ModelCheckpoint
mdc = ModelCheckpoint(BEST_MODEL_PATH,monitor='val_acc',
                      save_best_only=True,
                      modes='max',verbose=1)
# EarlyStopping
early_callback = EarlyStopping(monitor='val_loss',patience=5)

hist = model.fit(X_train,y_train,
                  epochs=50,
                  verbose=1,
                  validation_data=(X_test,y_test),
                  callbacks=[mdc,tensorboard_callback,early_callback])

# to check the hist keys
print(hist.history.keys())
#%%Model Evaluation
me = ModelEvaluation()

# Plot Training and Validation graph for loss
me.plot_loss_grapy(hist)

# Plot Training and Validation graph for accuracy
me.plot_acc_graph(hist)

print(model.evaluate(X_test,y_test))
#%%Model Analysis
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
y_test = np.argmax(y_test,axis=1)


cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)

# Plotting heatmap 
labels = ['0','1']
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

print(y_pred)
print(cr)