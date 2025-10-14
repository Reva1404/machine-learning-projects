import numpy as np
import pandas as pd
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],'Age': [25, 30, 35, 40, 45],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']}
df= pd.DataFrame(data)
print(df) #prints dataframe
print(df[df['Age']>30]) #Prints data where Age>30
df['AgeGroup']= np.where(df['Age']>=40, 'Senior','Adult') #Groups into senior and Adult
print(df)
Arranged= df.groupby("AgeGroup") #Arranges according to AgeGroup
print(Arranged['Age'].mean()) #Gives the mean 
