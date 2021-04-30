#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data_info = pd.read_csv('../DATA/lending_club_info.csv',index_col='LoanStatNew')


# In[3]:


data_info


# In[4]:


print(data_info.loc['revol_util']['Description'])


# In[5]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[6]:


feat_info('mort_acc')


# In[7]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df = pd.read_csv('../DATA/lending_club_loan_two.csv')


# In[9]:


df.head()


# In[10]:


df.info()


# In[11]:


sns.countplot(x='loan_status',data=df)


# In[12]:


plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],kde=False)


# In[13]:


df.columns


# In[14]:


df.corr()


# In[15]:


plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True,cmap='viridis')


# In[16]:


feat_info('loan_amnt')


# In[17]:


feat_info('installment')


# In[18]:


sns.scatterplot(x='installment',y='loan_amnt',data=df)


# In[19]:


sns.boxplot(x='loan_status',y='loan_amnt',data=df)


# In[20]:


df.groupby('loan_status').describe()['loan_amnt']


# In[21]:


df['grade'].sort_values().unique()


# In[22]:


df['sub_grade'].sort_values().unique()


# In[23]:


sns.countplot(x='grade',data=df,hue='loan_status')


# In[24]:


plt.figure(figsize=(12,4))
sns.countplot(x=df['sub_grade'].sort_values(),palette='coolwarm')


# In[25]:


plt.figure(figsize=(12,4))
sns.countplot(x=df['sub_grade'].sort_values(),palette='coolwarm',hue=df['loan_status'])


# In[26]:


plt.figure(figsize=(12,4))
sns.countplot(x=df[df['sub_grade']>'F']['sub_grade'].sort_values(),hue=df['loan_status'])


# In[27]:


df['loan_repaid'] = pd.get_dummies(df['loan_status'],drop_first=True)


# In[28]:


df[['loan_status','loan_repaid']]


# In[29]:


df.corr()['loan_repaid'].sort_values()[:-1].plot(kind='bar')


# In[30]:


df.isnull().sum()/3960.30


# In[31]:


feat_info('emp_title')


# In[32]:


feat_info('emp_length')


# In[33]:


df['emp_title'].nunique()


# In[34]:


df['emp_title'].value_counts()


# In[35]:


df = df.drop('emp_title',axis=1)


# In[36]:


list(df['emp_length'].dropna().sort_values().unique())


# In[37]:


emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']


# In[38]:


plt.figure(figsize=(12,4))
sns.countplot(df['emp_length'].sort_values(),order=emp_length_order)


# In[39]:


plt.figure(figsize=(12,4))
sns.countplot(df['emp_length'].sort_values(),hue=df['loan_status'])


# In[40]:


df.groupby('loan_status').count()


# In[41]:


emp_co = df[df['loan_status']=='Charged Off'].groupby('emp_length').count()['loan_status']


# In[42]:


emp_fp = df[df['loan_status']=='Fully Paid'].groupby('emp_length').count()['loan_status']


# In[43]:


emp_len = emp_co/emp_fp


# In[44]:


emp_len


# In[45]:


emp_len.plot(kind='bar')


# In[46]:


df = df.drop('emp_length',axis=1)


# In[47]:


df.isnull().sum()


# In[48]:


df['purpose'].head(10)


# In[49]:


df['title'].head(10)


# In[50]:


feat_info('purpose')


# In[51]:


feat_info('title')


# In[52]:


df = df.drop('title',axis=1)


# In[53]:


feat_info('mort_acc')


# In[54]:


df['mort_acc'].value_counts()


# In[55]:


df.corr()['mort_acc'].sort_values()


# In[56]:


total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


# In[57]:


def fill_mort_acc(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


# In[58]:


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'],x['mort_acc']),axis=1)


# In[59]:


df.isnull().sum()


# In[60]:


df = df.dropna()


# In[61]:


df.isnull().sum()


# In[62]:


len(df)


# In[63]:


df.select_dtypes(include='object').columns


# In[64]:


df['term'] = df['term'].apply(lambda x:int(x.split()[0]))


# In[65]:


df['term'].value_counts()


# In[66]:


df = df.drop('grade',axis=1)


# In[67]:


subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)


# In[68]:


df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)


# In[69]:


df.head()


# In[70]:


df.columns


# In[71]:


df.select_dtypes(include='object').columns


# In[72]:


feat_dummies = pd.get_dummies(df[['verification_status','application_type','initial_list_status','purpose']],drop_first=True)


# In[73]:


df = df.drop(['verification_status','application_type','initial_list_status','purpose'],axis=1)


# In[74]:


df = pd.concat([df,feat_dummies],axis=1)


# In[75]:


df.head()


# In[76]:


df['home_ownership'].value_counts()


# In[77]:


df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER')


# In[78]:


df['home_ownership'].value_counts()


# In[79]:


dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)


# In[80]:


df['zip_code'] = df['address'].apply(lambda x:x.split(' ')[-1])


# In[81]:


df['zip_code']


# In[82]:


zip_dummies = pd.get_dummies(df['zip_code'],drop_first=True)


# In[83]:


df = df.drop(['zip_code','address'],axis=1)


# In[84]:


df = pd.concat([df,zip_dummies],axis=1)


# In[85]:


df = df.drop('issue_d',axis=1)


# In[86]:


df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda x:int(x[-4:]))


# In[87]:


df = df.drop('earliest_cr_line',axis=1)


# In[88]:


df = df.drop('loan_status',axis=1)


# In[89]:


X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values


# In[90]:


from sklearn.model_selection import train_test_split


# In[91]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[92]:


from sklearn.preprocessing import MinMaxScaler


# In[93]:


scaler = MinMaxScaler()


# In[94]:


X_train = scaler.fit_transform(X_train)


# In[95]:


X_test = scaler.transform(X_test)


# In[96]:


X_train.shape


# In[97]:


X_test.shape


# In[98]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


# In[99]:


model = Sequential()

model.add(Dense(78,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(39,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(19,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')


# In[100]:


model.fit(x=X_train,y=y_train,
         validation_data=(X_test,y_test),epochs=25,
         batch_size=256,verbose=1)


# In[101]:


from tensorflow.keras.models import load_model


# In[102]:


model.save('ANN_model.h5')


# In[103]:


losses = pd.DataFrame(model.history.history)


# In[104]:


losses.plot()


# In[105]:


predictions = model.predict_classes(X_test)


# In[106]:


from sklearn.metrics import classification_report,confusion_matrix


# In[107]:


print(confusion_matrix(y_test,predictions))


# In[108]:


print(classification_report(y_test,predictions))


# In[109]:


import random


# In[110]:


random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


# In[111]:


model.predict_classes(new_customer.values.reshape(1,78))


# In[112]:


df.iloc[random_ind]['loan_repaid']


# In[ ]:




