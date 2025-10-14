import numpy as np
import random 
arr=np.random.randint(1,50,size=12) #Gives an array of 12 Random integers
print(arr)
matrix=arr.reshape(3,4) #Makes it a matrix
print(matrix)
print(arr.mean()) #gives mean
print(arr.max()) #gives maximum
print(arr.min()) #gives minimum