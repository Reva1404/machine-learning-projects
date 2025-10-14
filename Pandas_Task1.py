import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],'Age': [25, 30, 35, 40, 45],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']}

df= pd.DataFrame(data)
print(df) #prints dataframe
print(df.head(2)) #shows the first 2 rows 
print("The mean of Age is",df['Age'].mean()) #calculates the mean of age coloumn