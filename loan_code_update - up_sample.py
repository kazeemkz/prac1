import t
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm

from sklearn.utils import resample


#read csv file

file  = pd.read_csv('loan_train.csv') # read the pdf file

print(file.head(20)) # spit the first 20 rows

print(file.isna().sum())


the_mode  = file['Gender'].mode()  # fill with the data that appears the most
file['Gender'].fillna(value=the_mode[0],inplace=True)


the_gender_mode = file['Married'].mode() # fill with the data that appears the most
file['Married'].fillna(value=the_gender_mode[0],inplace=True)

self_employed_mode = file['Self_Employed'].mode()  # fill with the data that appears the most
file['Self_Employed'].fillna(value=self_employed_mode[0],inplace=True)

loan_amount_mean = file['LoanAmount'].mean() # fill with the mean of the column
file['LoanAmount'].fillna(value=loan_amount_mean,inplace=True)

credit_history_mode = file['Credit_History'].mode()# fill with the data that appears the most

file['Credit_History'].fillna(value=credit_history_mode[0],inplace=True)
#print(file.isna().sum())  # print sumary of empty spaces, per column

file.drop("Dependents",axis=1,inplace=True)
file.drop("ApplicantIncome",axis=1,inplace=True)
file.drop("CoapplicantIncome",axis=1,inplace=True)
file.drop("Loan_Amount_Term",axis=1,inplace=True)
file.drop("Loan_ID",axis=1,inplace=True)

print(file.isna().sum())

le = LabelEncoder() # get the encodeer

education_encode = le.fit_transform(file['Education'])
file.drop('Education',axis=1,inplace=True) # drop this column and add the encoded version
file['Education'] = education_encode

gender_encode = le.fit_transform(file['Gender'])
file.drop('Gender',axis=1,inplace=True)
file['Gender'] = gender_encode

married_encode = le.fit_transform(file['Married'])
file.drop('Married',axis=1,inplace=True)
file['Married'] = married_encode

Self_Employed_encode = le.fit_transform(file['Self_Employed'])
file.drop('Self_Employed',axis=1,inplace=True)
file['Self_Employed'] = Self_Employed_encode

Property_Area_encode = le.fit_transform(file['Property_Area'])
file.drop('Property_Area',axis=1,inplace=True)
file['Property_Area'] = Property_Area_encode

Loan_Status_encode = le.fit_transform(file['Loan_Status'])
file.drop('Loan_Status',axis=1,inplace=True)
file['Loan_Status'] = Loan_Status_encode


print(file.head(10))

print(file.groupby('Loan_Status').sum()) # group data based on loan status
#print(file.Loan_Status.value_counts())

#df_majority_approved = file[file.Loan_Status == 1]
df_majority_approved = file[file["Loan_Status"]== 1]
df_minority_declined = file[file["Loan_Status"]== 0]



df_manority_downsampled = resample(df_minority_declined,
                                   replace=True,
                                   n_samples=445,  # to match majority class
                                   random_state=123)  # reproducible results



#replace=True: If you set replace to True, it means that during resampling,
# the same data point can be selected multiple times. This is called "sampling with replacement.
# " It's commonly used when you want to generate a dataset of the same size as the original but
# with some data points duplicated.

#file = pd.concat([df_majority_downsampled, df_minority_declined])

file = pd.concat([df_majority_approved, df_manority_downsampled],axis=0)

# Display new class counts
#file.Loan_Status.value_counts()
print(file.groupby("Loan_Status").sum())

#print(file.head().values)  # check header names
#Gender            0
#Married           0
#Education         0
#Self_Employed     0
#LoanAmount        0
#Credit_History    0
#Property_Area     0
#Loan_Status       0
file_np = file.values   # take it to numpy array
x  = file_np[:,0:7]
y= file_np[:,7]

x_train,x_val, y_train, y_val = train_test_split(x,y,test_size=.30,random_state=908)

model = svm.SVC(kernel='rbf',gamma=1.0)
model.fit(x_train,y_train)
answer = model.predict(x_val)
checker = accuracy_score(answer,y_val)
print(checker)

model_name = 'loan_upsample.sav'
#joblib.dump(model,model_name)

#0.8084291187739464
