#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Let import all the required librairies

#for mathematical operations
import numpy as np
#for dataframe operations
import pandas as pd

#for data visualizations
import seaborn as sns
import matplotlib.pyplot as plt

#for machine learning
import sklearn
import imblearn


# In[2]:


#reading dataset
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# In[3]:


# lets check the shape of the train test datasets
print('The shape of train set is:',train.shape)
print('The shape of test set is:', test.shape)


# In[4]:


#Columns in traing data
train.columns


# In[5]:


#columns in test dataset
test.columns


# In[6]:


train.head()


# In[7]:


train.tail()


# In[8]:


#Descristive statistics
train.describe()


# In[9]:


train.describe(include='object')


# In[10]:


#missing value
train.isnull().sum().sum()


# In[11]:


train.isnull().sum().sort_values(ascending=False)


# In[12]:


percetenge_of_missing_value=round(100*(train.isnull().sum())/train.shape[0],2).sort_values(ascending=False)
percetenge_of_missing_value


# In[13]:


#impute the value 
train['education']=train['education'].fillna(train['education'].mode()[0])
train['previous_year_rating']=train['previous_year_rating'].fillna(train['previous_year_rating'].mean())
train.isnull().sum().sum()


# # Outliers detection

# In[14]:


# lets first analyze the numerical columns
#train.select_dtypes('number').columns
train.select_dtypes('number').head()


# In[15]:


plt.rcParams['figure.figsize']=(12,3)
plt.style.use('fivethirtyeight')
#for j in range(1,3):
for i in train.select_dtypes('number').columns:
       # plt.subplot(1,3,1)
        sns.boxplot(train[i])
        plt.show()


# In[16]:


# age and services
train=train[~(train['length_of_service']>13)]
train=train[~(train['age']>50)]
train=train[~(train['previous_year_rating']<2)]


# In[17]:


#univariante analysis
#Correlation matrix
corrmat=train.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=.8,square=True)
plt.show()


# # univariante

# In[18]:


plt.subplot(1,3,1)
labels=['0','1']
sizes=train['KPIs_met >80%'].value_counts()
color=plt.cm.Wistia(np.linspace(0,1,5))
explode=[0,0]

plt.pie(sizes,labels=labels,colors=color,explode=explode,shadow=True,startangle=90)
plt.title('KPIs_met >80%',fontsize=20)

# plotting a pie chart to represent share of Previous_year_rating of employee

plt.subplot(1,3,2)
labels=['0','1','2','3','4']
sizes=train['previous_year_rating'].value_counts()
colors=plt.cm.Wistia(np.linspace(0,1,5))
explode=[0,0,0,0,0.1]
plt.pie(sizes,colors=colors,labels=labels,explode=explode,shadow=True,startangle=90)
plt.title('previous_year_rating')
#plt.legend()
# plotting a pie chart to represent of previous year award

plt.subplot(1,3,3)
labels=['0','1']
sizes=train['awards_won?'].value_counts()
colors=plt.cm.Wistia(np.linspace(0,1,5))
explode=[0,0.1]

plt.pie(sizes,colors=colors,explode=explode,labels=labels,shadow=True,startangle=90)
plt.title('award_won?')

plt.legend()
plt.show()


# In[19]:


#Lets checks the distibution of trainings undertaken by the employees

plt.rcParams['figure.figsize']=(8,4)
sns.countplot(train['no_of_trainings'],palette='spring')
plt.xlabel(' ',fontsize=14)
plt.title('Distribution of trainings undertaken by the Employees')
plt.show()


# In[20]:


#Distrubition of age amoung the Employees
plt.rcParams['figure.figsize']=(8,4)
plt.hist(train['age'],color='black')
plt.title('Distribution of age amoung the Employess', fontsize=10)
plt.xlabel('Age of the Employees')
plt.grid()
plt.show()


# In[21]:


# Lets check different departments
plt.rcParams['figure.figsize']=(18,12)
sns.countplot(y=train['department'],palette='cividis',orient='v')
plt.xlabel('')
plt.ylabel('Department name')
plt.title('Distribution of Employees in Different departments', fontsize=15)
plt.grid()
plt.show()


# In[22]:


#lets check the regions
plt.rcParams['figure.figsize']=(18,12)
sns.countplot(y=train['region'])
plt.xlabel('')
plt.title('Regions',fontsize=15)


# In[23]:


plt.subplot(1,3,1)
labels= train['education'].value_counts().index
sizes=train['education'].value_counts()
colors=plt.cm.copper(np.linspace(0,1,5))
explode=[0,0,0.1]

plt.pie(sizes,labels=labels,colors=colors,explode=explode,shadow=True,startangle=90)
plt.title('Education',fontsize=20)

# plotting a pie chart to represent share of gender
plt.subplot(1,3,2)
labels=train['gender'].value_counts().index
sizes=train['gender'].value_counts()
explode=[0,0.1]
plt.pie(sizes,labels=labels,colors=colors,explode=explode,shadow=True,startangle=90)
plt.title('Gender',fontsize=20)

plt.subplot(1,3,3)
sizes=train['recruitment_channel'].value_counts()
labels=train['recruitment_channel'].value_counts().index
explode=[0,0,0.1]
plt.pie(sizes,colors=colors,labels=labels,explode=explode,shadow=True,startangle=90)
plt.title('Recruitment channel',fontsize=20)


# # Bivariate analysis

# In[24]:


#lets compare the gender gap in prmotion
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize']=(12,4)
x=pd.crosstab(train['gender'],train['is_promoted'])
color=plt.cm.copper(np.linspace(0,1,5))
x.div(x.sum(1).astype(float),axis=0).plot(kind='bar',stacked=False,color=colors)
plt.title('Effect of gender on promotion',fontsize=20)

plt.show()


# In[25]:


#lets  compare the off different Departments and promotion

plt.rcParams['figure.figsize']=(14,8)
x=pd.crosstab(train['department'],train['is_promoted'])
colors=plt.cm.copper(np.linspace(0,1,3))
x.div(x.sum(1).astype(float),axis=0).plot(kind='area',color=colors,stacked=False)
plt.title('Effect of Department on promotions',fontsize=20)
plt.xticks(rotation=20)


# In[26]:


#Effect of age on the promotion
plt.rcParams['figure.figsize']=(12,4)
sns.boxenplot(train['is_promoted'],train['age'],palette='PuRd')


# In[27]:


#Department vs average training score
plt.rcParams['figure.figsize']=(12,4)
sns.boxplot(train['department'],train['avg_training_score'],palette='autumn')
plt.ylabel('Average training')


# # Multivariate analysis

# In[28]:


#Correlation matrix
plt.rcParams['figure.figsize']=(12,8)
corrmat=train.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=.8,square=True)
plt.show()


# In[29]:


plt.rcParams['figure.figsize']=(12,8)
corrmat=train.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat,annot=True,linewidth=0.5,square=True)
plt.show()


# In[30]:


#Department vs average training score
plt.rcParams['figure.figsize']=(12,4)
sns.barplot(train['department'],train['avg_training_score'],hue=train['gender'],palette='autumn')
plt.ylabel('Average training')


# # Feature engineering

# In[31]:


# lets create some extra features from existing to impove our model
# creating a metric of sum
train['sum_metric']=train['awards_won?']+train['KPIs_met >80%']+train['previous_year_rating']
test['sum_metric']=test['awards_won?']+test['KPIs_met >80%']+test['previous_year_rating']
#creating a total score column
train['total_score']=train['avg_training_score']*train['no_of_trainings']
test['total_score']=test['avg_training_score']*test['no_of_trainings']

#lets remove unecessary columns
train=train.drop(['recruitment_channel','region','employee_id'],axis=1)
test=test.drop(['recruitment_channel','region','employee_id'],axis=1)


# In[32]:


train.columns


# # Dealing with categorical columns

# In[33]:


# Lets check all th categorical columns present  in dataset
train.select_dtypes('object').head()


# In[34]:


train['department'].value_counts()


# In[35]:


#lets use label Encoding for department to convert them into numerical
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
train['department']=le.fit_transform(train['department'])
test['department']=le.fit_transform(test['department'])


# In[36]:


#lets use label Encoding for department to convert them into numerical
train['gender']=le.fit_transform(train['gender'])
test['gender']=le.fit_transform(test['gender'])


# In[37]:


train['education'].value_counts()


# In[38]:


#lets start encoding these categorical columns to convert them into numerical columns
#lets encode the education in their degree of imortance

train['education']=train['education'].replace(("Master's & above","Bachelor's","Below Secondary"),(3,2,1))
test['education']=test['education'].replace(("Master's & above","Bachelor's","Below Secondary"),(3,2,1))


# # Data processing

# - Target column splitting
# - validation set splitting
# - Statistical sampling to make the data balanced

# ### Splitting data

# In[39]:


#lets split the target data from the train data
y=train['is_promoted']
x=train.drop(['is_promoted'],axis=1)
x_test=test
#lets print the shape of thes newly formed data sets

print("shape of the x:",x.shape)
print("shape of the y:",y.shape)
print("shape of the x test :",x_test.shape)


# ### Resampling

# In[40]:


from imblearn.over_sampling import SMOTE
x_resample,y_resample=SMOTE().fit_resample(x,y.values.ravel())

print(x_resample.shape)
print(y_resample.shape)


# In[41]:


#lets check the value counts of our target value
print("before resampling:" ,y.value_counts())
y_resample=pd.DataFrame(y_resample)
print("after resampling: ",y_resample[0].value_counts())


# # Feature Scaling

# In[42]:


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25, random_state=1)


# In[43]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_valid=sc.fit_transform(x_valid)
x_test=sc.fit_transform(x_test)


# # Predicting modelling
# 

# In[44]:


# lets use decision trees to classify the data
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_valid)


# # Performance analysis

# In[45]:


from sklearn.metrics import confusion_matrix,classification_report
print("Training Accuracy: ", model.score(x_train,y_train))
print("validation Accuracy: ", model.score(x_valid,y_valid))


# In[46]:


cm = confusion_matrix(y_valid,y_pred)
plt.rcParams['figure.figsize']=(3,3)
sns.heatmap(cm,annot=True,fmt='.8g')
plt.xlabel('predict value')
plt.ylabel('actual values')
plt.show()


# In[47]:


# classification report
cr=classification_report(y_valid,y_pred)
print(cr)


# In[61]:


#model.predict(x.loc[10,:])
#x=np.array([pclass,sex,age]).reshape(1,3)
A=np.array(x.loc[0:10,:]).reshape(10,12)
model.predict(A)


# In[63]:


model.feature_importances_


# In[67]:


tmp=pd.DataFrame({'Features':x.columns,'Features importance':model.feature_importances_})
tmp=tmp.sort_values(by='Features importance',ascending=False)
plt.figure(figsize=(18,8))
s=sns.barplot(x='Features',y='Features importance',data=tmp)
plt.title('Feature importance',fontsize=14)


# In[73]:


from sklearn.metrics import roc_auc_score
pred1=model.predict(x_train)
roc_auc_score(pred1,y_train)


# In[ ]:




