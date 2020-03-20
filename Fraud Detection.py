#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[249]:


import numpy as np
import pandas as pd


# # Reading Datasets

# In[250]:


train=pd.read_csv("train-1561627878332.csv")
train_order=pd.read_csv("train_order_data-1561627847149.csv")
train_merchant=pd.read_csv("train_merchant_data-1561627820784.csv")


# In[251]:


test=pd.read_csv("test-1561627952093.csv")
test_order=pd.read_csv("test_order_data-1561627931868.csv")
test_merchant=pd.read_csv("test_merchant_data-1561627903902.csv")


# In[252]:


ip_bound=pd.read_csv("ip_boundaries_countries-1561628631121.csv")


# # Data Exploration

# In[253]:


train.Merchant_ID.value_counts()


# # Merging given datasets on a common attribute

# In[254]:


train1=pd.merge(train,train_merchant,how='outer',on='Merchant_ID')


# In[255]:


train2=pd.merge(train1,train_order,how='outer',on='Merchant_ID')


# In[256]:


test1 =pd.merge(test,test_merchant,how='outer',on='Merchant_ID')


# In[257]:


test2=pd.merge(test1,test_order,how='outer',on='Merchant_ID')


# In[217]:


train2


# # Merging ip_bound and given train data sets to add country column to training data

# In[258]:


ip_bound.lower_bound_ip_address


# In[259]:


l1=ip_bound['lower_bound_ip_address'].apply(lambda x: str(x).split('.',4))


# In[220]:


l1


# In[260]:


l=[j[0]+j[1]+j[2]+j[3] for j in l1]


# In[222]:


l


# In[261]:


u=ip_bound['upper_bound_ip_address'].apply(lambda x: str(x).split('.',4))


# In[224]:


u


# In[262]:


low1=[j[0] for j in l1]


# In[263]:


low2=[j[1] for j in l1 ]


# In[265]:


low3 =[j[2] for j in l1]


# In[266]:


low4=[j[3] for j in l1 ]


# In[267]:


ip_bound['low1']=low1


# In[268]:


ip_bound['low2']=low2
ip_bound['low3']=low3
ip_bound['low4']=low4


# In[269]:


u1=[j[0] for j in u]
u2=[j[1] for j in u]
u3=[j[2] for j in u]
u4=[j[3] for j in u]


# In[270]:


ip_bound['up1']=u1
ip_bound['up2']=u2
ip_bound['up3']=u3
ip_bound['up4']=u4


# In[271]:


ip_bound


# In[273]:


ip_split=[]


# In[274]:


ip=train2.IP_Address.apply(lambda x: x.split(".",4))


# In[275]:


ips


# In[276]:


train2.shape


# In[277]:


ip_split=np.array_split(ip,20)


# In[278]:


ip_split[0].shape


# In[279]:


ip_bound


# # Changing data types appropriately for columns

# In[281]:


train2.info()


# In[282]:


train2.Fraudster=train2.Fraudster.astype('category')
train2.Gender=train2.Gender.astype('category')
train2.Order_Source=train2.Order_Source.astype('category')
train2.Order_Payment_Method=train2.Order_Payment_Method.astype('category')


# In[285]:


#test2.Fraudster=test2.Fraudster.astype('category')
test2.Gender=test2.Gender.astype('category')
test2.Order_Source=test2.Order_Source.astype('category')
test2.Order_Payment_Method=test2.Order_Payment_Method.astype('category')


# # Dropping irrelevant columns

# In[286]:


train2.info()


# In[287]:


test2.info()


# In[288]:


train2.drop('Merchant_ID',axis=1,inplace=True)


# In[289]:


train2.drop('Order_ID',axis=1,inplace=True)


# In[290]:


train2.drop('Ecommerce_Provider_ID',axis=1,inplace=True)


# In[291]:


test2.drop('Merchant_ID',axis=1,inplace=True)
test2.drop('Order_ID',axis=1,inplace=True)
test2.drop('Ecommerce_Provider_ID',axis=1,inplace=True)


# In[292]:


train2.describe(include='object')


# In[293]:


train2.describe(include='category')


# In[294]:


train2.drop('Registered_Device_ID',axis=1,inplace=True)


# In[295]:


test2.drop('Registered_Device_ID',axis=1,inplace=True)


# In[296]:


train2.info()


# # Extracting features out of the given data

# In[297]:


train2.describe(include=['object','category','int64'])


# In[298]:


str(train2['Merchant_Registration_Date'][0])[17:19]


# In[299]:


train2['MRD_year']=train2['Merchant_Registration_Date'].apply(lambda x: str(x)[0:4])


# In[300]:


train2['MRD_month']=train2['Merchant_Registration_Date'].apply(lambda x: str(x)[5:7])


# In[301]:


train2['MRD_day']=train2['Merchant_Registration_Date'].apply(lambda x: str(x)[8:10])


# In[302]:


train2['MRD_hour']=train2['Merchant_Registration_Date'].apply(lambda x: str(x)[11:13])


# In[303]:


train2['MRD_min']=train2['Merchant_Registration_Date'].apply(lambda x: str(x)[14:16])


# In[304]:


train2['MRD_sec']=train2['Merchant_Registration_Date'].apply(lambda x: str(x)[17:19])


# In[305]:


test2['MRD_year']=test2['Merchant_Registration_Date'].apply(lambda x: str(x)[0:4])


# In[306]:


test2['MRD_month']=test2['Merchant_Registration_Date'].apply(lambda x: str(x)[5:7])


# In[307]:


test2['MRD_day']=test2['Merchant_Registration_Date'].apply(lambda x: str(x)[8:10])


# In[308]:


test2['MRD_hour']=test2['Merchant_Registration_Date'].apply(lambda x: str(x)[11:13])


# In[309]:


test2['MRD_min']=test2['Merchant_Registration_Date'].apply(lambda x: str(x)[14:16])


# In[310]:


test2['MRD_sec']=test2['Merchant_Registration_Date'].apply(lambda x: str(x)[17:19])


# In[311]:


train2.describe(include=['object','category','int64'])


# In[312]:


test2.describe(include=['object','category','int64'])


# In[313]:


train2.drop('MRD_year',axis=1,inplace=True)


# In[314]:


test2.drop('MRD_year',axis=1,inplace=True)


# In[315]:


(train2.Date_of_Order[0])[5:7]


# In[316]:


train2['DO_year']=train2.Date_of_Order.apply(lambda x: str(x)[0:4])


# In[317]:


train2['DO_month']=train2.Date_of_Order.apply(lambda x: str(x)[5:7])


# In[318]:


train2['DO_day']=train2.Date_of_Order.apply(lambda x: str(x)[8:10])


# In[319]:


train2['DO_hour']=train2.Date_of_Order.apply(lambda x: str(x)[11:13])


# In[320]:


train2['DO_min']=train2.Date_of_Order.apply(lambda x: str(x)[14:16])


# In[321]:


train2['DO_sec']=train2.Date_of_Order.apply(lambda x: str(x)[17:19])


# In[322]:


#####################


# In[323]:


test2['DO_year']=test2.Date_of_Order.apply(lambda x: str(x)[0:4])


# In[324]:


test2['DO_month']=test2.Date_of_Order.apply(lambda x: str(x)[5:7])


# In[325]:


test2['DO_day']=test2.Date_of_Order.apply(lambda x: str(x)[8:10])


# In[326]:


test2['DO_hour']=test2.Date_of_Order.apply(lambda x: str(x)[11:13])


# In[327]:


test2['DO_min']=test2.Date_of_Order.apply(lambda x: str(x)[14:16])


# In[328]:


test2['DO_sec']=test2.Date_of_Order.apply(lambda x: str(x)[17:19])


# In[329]:


test2


# In[330]:


train2.info()


# In[331]:


test2.info()


# # Dropping irrelevant columns

# In[332]:


train2.describe(include=['object','category','int64'])


# In[333]:


train2.drop('DO_year',axis=1,inplace=True)


# In[334]:


test2.drop('DO_year',axis=1,inplace=True)


# In[335]:


train2.describe(include)


# # Extracting Features out of given Data

# In[336]:


train2['ip1']=[j[0] for j in ip]


# In[337]:


train2['ip2']=[j[1] for j in ip]


# In[338]:


train2['ip3']=[j[2] for j in ip]


# In[339]:


train2['ip4']=[j[3] for j in ip]


# In[340]:


ip_test=test2['IP_Address'].apply(lambda x: str(x).split('.',4))


# In[341]:


test2['ip1']=[j[0] for j in ip_test]


# In[342]:


test2['ip2']=[j[1] for j in ip_test]


# In[343]:


test2['ip3']=[j[2] for j in ip_test]


# In[344]:


test2['ip4']=[j[3] for j in ip_test]


# In[345]:


test2


# In[346]:


train2.info()


# # Dropping irrelevant columns

# In[347]:


train2.drop('Merchant_Registration_Date',axis=1,inplace=True)


# In[348]:


train2.drop('Date_of_Order',axis=1,inplace=True)


# In[349]:


train2.drop('IP_Address',axis=1,inplace=True)


# In[350]:


test2.drop('Merchant_Registration_Date',axis=1,inplace=True)


# In[351]:


test2.drop('Date_of_Order',axis=1,inplace=True)


# In[352]:


test2.drop('IP_Address',axis=1,inplace=True)


# In[353]:


train2.describe(include=['object','category','int64'])


# In[354]:


train2.info()


# # converting columns to appropriate data types

# In[355]:


train2.ip1=train2.ip1.astype('int64')
train2.ip2=train2.ip2.astype('int64')
train2.ip3=train2.ip3.astype('int64')
train2.ip4=train2.ip4.astype('int64')


# In[356]:


test2.ip1=test2.ip1.astype('int64')
test2.ip2=test2.ip2.astype('int64')
test2.ip3=test2.ip3.astype('int64')
test2.ip4=test2.ip4.astype('int64')


# In[357]:


train2.MRD_day=train2.MRD_day.astype('int64')
train2.MRD_hour=train2.MRD_hour.astype('int64')
train2.MRD_min=train2.MRD_min.astype('int64')
train2.MRD_month=train2.MRD_month.astype('int64')
train2.MRD_sec=train2.MRD_sec.astype('int64')


# In[358]:


test2.MRD_day=test2.MRD_day.astype('int64')
test2.MRD_hour=test2.MRD_hour.astype('int64')
test2.MRD_min=test2.MRD_min.astype('int64')
test2.MRD_month=test2.MRD_month.astype('int64')
test2.MRD_sec=test2.MRD_sec.astype('int64')


# In[359]:


train2.DO_day=train2.DO_day.astype('int64')
train2.DO_hour=train2.DO_hour.astype('int64')
train2.DO_min=train2.DO_min.astype('int64')
train2.DO_month=train2.DO_month.astype('int64')
train2.DO_sec=train2.DO_sec.astype('int64')


# In[360]:


test2.DO_day=test2.DO_day.astype('int64')
test2.DO_hour=test2.DO_hour.astype('int64')
test2.DO_min=test2.DO_min.astype('int64')
test2.DO_month=test2.DO_month.astype('int64')
test2.DO_sec=test2.DO_sec.astype('int64')


# In[361]:


train2.DO_day.value_counts()


# In[362]:


train2.info()


# # Dropping Customer ID column

# In[363]:


train2.drop('Customer_ID',axis=1,inplace=True)


# In[364]:


test2.drop('Customer_ID',axis=1,inplace=True)


# In[365]:


train2.info()


# In[366]:


train2.describe(include=['object','category','int64'])


# In[367]:


test2.isnull().sum().sum()


# In[368]:


train2.isnull().sum().sum()


# In[369]:


train2


# # Plotting Data distribution in Target Variable

# In[370]:


# Plot Distribution
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Fraudster',data=train2)
plt.show()

# What are the counts?
print(train2.Fraudster.value_counts())

# What is the percentage?
count_yes = len(train2[train2.Fraudster == '1'])
count_no = len(train2[train2.Fraudster != '1'])

percent_success = (count_yes/(count_yes + count_no))*100

print('Percentage of fraudster Merchants:', percent_success, "%")


# In[371]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Gender',data=train2)
plt.show()


# In[372]:


# Check distribution of age
get_ipython().run_line_magic('matplotlib', 'inline')
sns.distplot(train2["Age"] )


# In[373]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Order_Source',data=train2)
plt.show()


# In[374]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Order_Payment_Method',data=train2)
plt.show()


# In[375]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='Gender',data=train2)
plt.show()


# In[376]:


cat_cols=['Order_Payment_Method','Order_Source','Gender']


# In[377]:


d1=pd.get_dummies(columns=cat_cols,data=train2[cat_cols],prefix=cat_cols, prefix_sep="_",drop_first=True)


# In[378]:


d1


# In[379]:


d2=pd.get_dummies(columns=cat_cols,data=test2[cat_cols],prefix=cat_cols, prefix_sep="_",drop_first=True)


# In[380]:


d2


# In[381]:


test3=pd.concat([test2,d2],axis=1)


# In[382]:


train3=pd.concat([train2,d1],axis=1)


# In[383]:


train3


# # Dropping irrelevant columns

# In[384]:


train3.drop('Order_Source',axis=1,inplace=True)
train3.drop('Order_Payment_Method',axis=1,inplace=True)
train3.drop('Gender',axis=1,inplace=True)


# In[385]:


test3.drop('Order_Source',axis=1,inplace=True)
test3.drop('Order_Payment_Method',axis=1,inplace=True)
test3.drop('Gender',axis=1,inplace=True)


# In[386]:


train3.info()


# In[387]:


y=train3.Fraudster
x=train3
x.drop('Fraudster',axis=1,inplace=True)


# # handling Data Imbalance

# In[388]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=646)


# Fit on Data
os_data_x,os_data_y = smote.fit_sample(x,y)


# In[389]:


os_data_x=pd.DataFrame(os_data_x,columns=x.columns)


# In[390]:


os_data_x


# In[391]:


os_data_y=pd.DataFrame(os_data_y,columns=['Fraudster'])


# In[392]:


os_data_x.shape


# In[393]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(os_data_x, os_data_y, test_size=0.20,random_state=646) 


# # Logistic Regression

# In[394]:


from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()


# In[395]:


model1.fit(x_train,y_train)


# In[403]:


from sklearn.metrics import confusion_matrix, f1_score
train_pred = model1.predict(x_train)
test_pred = model1.predict(x_test)

print(model1.score(x_train, y_train))
print(model1.score(x_test, y_test))

print(confusion_matrix(y_true=y_train, y_pred = train_pred))

confusion_matrix_test = confusion_matrix(y_true=y_test, y_pred =  test_pred)


# In[404]:


f1_score(y_train,train_pred)


# In[397]:


test3


# In[398]:


Test_pred=model1.predict(test3)


# In[399]:


Test_pred=pd.DataFrame(Test_pred)


# In[400]:


# Create Dataframe
#os_data_X = pd.DataFrame(data=os_data_X)
#os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])


# In[401]:


Test_pred['Merchant_ID']=test1.Merchant_ID


# In[402]:


Test_pred1=pd.DataFrame(Test_pred1)


# In[ ]:


Test_pred1


# In[ ]:


Test_pred.to_csv('pred_1.csv',index=True)


# # Standardization

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[ ]:


sc.fit(x_train,y_train)


# In[ ]:


x_train


# In[ ]:


std_x_train=sc.transform(x_train)


# In[ ]:


std_x_test=sc.transform(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
train_pred = model1.predict(std_x_train)
test_pred = model1.predict(std_x_test)

print(model1.score(std_x_train, y_train))
print(model1.score(std_x_test, y_test))

print(confusion_matrix(y_true=y_train, y_pred = train_pred))

confusion_matrix_test = confusion_matrix(y_true=y_test, y_pred =  test_pred)


# # Decision Trees

# In[405]:


from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=4,random_state=646)


# In[406]:


clf.fit(x_train,y_train)


# In[407]:


importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
cols=pd.DataFrame([x_train.columns[indices],np.sort(importances)[::-1]])


# In[413]:


feat_importances = pd.Series(clf.feature_importances_, index = x_train.columns)
feat_importances.plot(kind='bar')


# In[414]:


feat_importances_ordered = feat_importances.nlargest(11)
feat_importances_ordered.plot(kind='bar')


# In[422]:


cols


# In[423]:


cols=cols.iloc[0,0:6]


# In[ ]:


cols


# In[408]:


tree.plot_tree(clf.fit(x_train,y_train))


# In[409]:


train_pred=clf.predict(x_train)


# In[410]:


confusion_matrix(y_train,train_pred)


# In[411]:


f1_score(y_train,train_pred)


# In[ ]:


clf.predict(x_test)


# In[ ]:


confusion_matrix(y_train,train_pred)


# In[ ]:


test_pred=clf.predict(test3)


# In[ ]:


test_pred=pd.DataFrame(test_pred)


# In[ ]:


test_pred.to_csv('pred2.csv')


# In[415]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()


# In[416]:


rfc.fit(x_train, y_train)


# In[417]:


train_predictions = rfc.predict(x_train)
test_predictions = rfc.predict(x_test)


# In[418]:


confusion_matrix(y_train,train_predictions)


# In[419]:


f1_score(y_train,train_predictions)


# In[ ]:


test_pred3=rfc.predict(test3)


# In[ ]:


test_pred3=pd.DataFrame(test_pred3)


# In[ ]:


test_pred3.to_csv('pred3.csv')


# In[ ]:


test3


# In[ ]:


test4=test3[cols]


# In[ ]:


test4


# # Random Forest

# In[420]:


#from sklearn.ensemble import RandomForestClassifier

rfc1 = RandomForestClassifier()


# In[424]:


x_train1=x_train[cols]


# In[ ]:


x_train1=pd.DataFrame(x_train1)


# In[425]:


rfc1.fit(x_train1,y_train)


# In[427]:


y_pred=rfc1.predict(x_train1)


# In[428]:


confusion_matrix(y_train,y_pred)


# In[429]:


f1_score(y_train,y_pred)


# In[ ]:


pred4=rfc1.predict(test4)


# In[ ]:


pred4=pd.DataFrame(pred4)


# In[ ]:


pred4.to_csv('pred4.csv')


# In[ ]:


rfc.feature_importances_


# # Support Vector Machine

# In[430]:


from sklearn.svm import SVC

## Create an SVC object and print it to see the default arguments
svc = SVC()


# In[ ]:


svc.fit(x_train,y_train)


# In[431]:


x_pred=svc.predict(x_train)


# In[ ]:


test_pred5=svc.predict(test3)

