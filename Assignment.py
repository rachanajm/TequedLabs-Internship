import pandas as pd
data=pd.read_csv('diabetes.csv')
df=pd.DataFrame(data,columns=['Age','BloodPressure','Glucose'])
print(df)

