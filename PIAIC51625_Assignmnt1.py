#!/usr/bin/env python
# coding: utf-8

# In[352]:


import numpy as np
x=np.array([[1,2,3], [4,5,6],[7,8,9]])
print(x)


# In[353]:


import numpy as np
x=np.zeros((10))
print(x)


# In[354]:


import numpy as np
x=10 + np.arange(40)
print(x)


# In[51]:


import numpy as np
x=10 + np.arange(40)
print(x.shape)


# In[52]:


import numpy as np
x=10+ np.arange(40)
type(x)


# In[23]:


import numpy as np
print(np._version_)
print(np.show_config())


# In[31]:


import numpy as np
x=np.zeros((10))
x.ndim


# In[38]:


import numpy as np
x=np.array([[1,2,3], [4,5,6],[7,8,9]])
bool_array=(x>0)
print(bool_array)


# In[41]:


# Two Dimentional Array
import numpy as np
x=np.array([[1,2,3], [4,5,6]])
print(x)
print(x.shape)
print(x.dtype)


# In[42]:


#Three Dimentional Array
import numpy as np
x=np.array([[1,2,3], [4,5,6],[7,8,9]])
print(x)
print(x.shape)
print(x.dtype)


# In[46]:


#Reverse a vector (first element becomes last)
import numpy as np
x=np.arange(1,10)
print("Original Array")
print(x)
print("Reverse Array")
x=x[::-1]
print(x)


# In[60]:


#Create a null vector of size 10 but the fifth value which is 1
import numpy as np
x=np.arange(10)
print("Without Modification")
print(x)
print("Print 1 at 5th index")
x[5]=1
print(x)


# In[61]:


#Create a 3x3 identity matrix
import numpy as np
x=np.identity(3)
print(x)


# In[66]:


#Convert the data type of the given array from int to float
import numpy as np
arr=np.array([1,2,3,4,5])
print("Integer Type Array")
print(arr)
print("Float Type Array")
x=dtype=np.float64(arr)
print(x)


# In[76]:


#Multiply arr1 with arr2
import numpy as np
arr1=np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
arr2=np.array([[0.0,4.0,1.0],[7.0,2.0,12.0]])
print("1st way of Multiplication")
print(arr1*arr2)
print("2nd way of Multiplication")
arr= np.multiply(arr1,arr2)
print(arr)


# In[90]:


#Make an array by comparing both the arrays provided above
import numpy as np
arr1=np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
arr2=np.array([[0.0,4.0,1.0],[7.0,2.0,12.0]])
print("arr1 > arr2")
print(np.greater(arr1, arr2))
print("arr1 >= arr2")
print(np.greater_equal(arr1, arr2))
print("arr1 < arr2")
print(np.less(arr1, arr2))
print("arr1 <= arr2")
print(np.less_equal(arr1, arr2))


# In[129]:


#Extract all odd numbers from arr with values(0-9)
import numpy as np 
 
data=np.arange(1,10,dtype=int) 
for i in range(len(data)): 
    if(i%2==1): 
        data[i]=i
print(data) 


# In[127]:


#Replace all odd numbers to -1 from previous array
import numpy as np 
 
data=np.arange(1,10,dtype=int) 
for i in range(len(data)): 
    if(i%2==0): 
        data[i]=-1
print(data) 


# In[135]:


#Replace the values of indexes 5,6,7 and 8 to 12
import numpy as np
arr = np.arange(10)
arr[5]=12
arr[6]=12
arr[7]=12
arr[8]=12
print(arr)


# In[139]:


#Create a 2d array with 1 on the border and 0 inside
import numpy as np
x = np.ones((5,5))
print("Original")
print(x)
print("1 on Border and 0 Inside the array")
x[1:-1,1:-1] = 0
print(x)


# In[143]:


#Replace the value 5 to 12
import numpy as np
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d)
arr2d[1,1]=12
print(arr2d)


# In[147]:


#Convert all the values of 1st array to 64
import numpy as np
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print("Original Array")
print(arr3d)
print("After Correction")
arr3d[0,0]=[64]
print(arr3d)


# In[151]:


# Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it
import numpy as np
arr2d = np.array([[11, 22, 33],
                [44, 55, 66],
                [77, 88, 99]])

a, b = data[:, :-1], data[:, -1]
print("2D Array")
print(a)
print("1D Array")
print(b)


# In[157]:


#Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it
import numpy as np
arr2d = np.array([[11, 22, 33],
                [44, 55, 66],
                [77, 88, 99]])

a = data[:1:],
print("1D Array")
print(a)


# In[209]:


#Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it
import numpy as np
arr2d= np.array([[1,2,3], [4,5,6],[7,8,9]])
print(arr2d)
print("After Slicing")
a=arr2d[1,1]
print(a)


# In[234]:


#Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows
import numpy as np
arr2d= np.array([[1,2,3], [4,5,6],[7,8,9]])
print(arr2d)
c=arr2d[:2,:-1]
print(c)


# In[240]:


#Create a 10x10 array with random values and find the minimum and maximum values
import numpy as np
arr10d=np.random.random((10,10))
print(arr10d)


# In[248]:


#Find the common items between a and b
import numpy as np
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.intersect1d(a,b))


# In[252]:


#Find the positions where elements of a and b match
import numpy as np
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.where((np.in1d(a,b))))


# In[256]:


#Find all the values from array data where the values from array names are not equal to Will
import numpy as np
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
x=np.where(names!='Will')
print(x)


# In[272]:


#Find all the values from array data where the values from array names are not equal to Will and Joe

import numpy as np
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
x=np.where(names!='Will')
y=np.where(names!='Joe')
print(x,y)


# In[279]:


#Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.
import numpy as np
x=np.random.rand(5, 3) * 5 + 10
print(x)


# In[276]:


#Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.
import numpy as np
x=np.random.rand(2,2,4) * 5 + 11
print(x)


# In[282]:


#Swap axes of the array you created in Question 32
import numpy as np
x=np.random.rand(2,2,4) * 5 + 11
y=np.swapaxes(x,0,2)
print(x)
print("After Swaping Axes")
print(y)


# In[293]:


#Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0
import numpy as np
x=np.arange(10)
print("Original Array")
print(x)
y=np.sqrt(x)
print("After Square Root")
print(y)
print("After Applying Condition")
for i in range(len(data)): 
    if(i<0.5): 
        data[i]=0
print(data)


# In[303]:


#Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays
import numpy as np
arr1=np.random.random(12)
arr2=np.random.random(12)
print(arr1)
print(arr2)

z=np.maximum(arr1,arr2)
print("Maximum Array Will be")
print(z)


# In[310]:


#Find the unique names and sort them out
import numpy as np
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
print("Before Unique & Sort")
print(names)
y=np.unique(names)
z=np.sort(y)
print("After Unique Command")
print(y)
print("After Sort Command")
print(z)


# In[313]:


#From array a remove all items present in array b
import numpy as np
a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
c=np.setdiff1d(a,b)
print(c)


# In[338]:


#Following is the input NumPy array delete column two and insert following new column in its place.
import numpy as np
sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
newColumn = np.array([[10,10,10]])
print("Original Array")
print(sampleArray)
print("After Removing Column 2")
a_del = np.delete(sampleArray, 1, 1)
print(a_del)
print("After Adding Column 2")
a_add = np.insert(sampleArray,1,1)
print(a_add)
newArray = np.append(sampleArray, [[50, 60, 70]], axis = 0)
print(newArray)


# In[344]:


import numpy as np
a = np.array([[34,43,73],[82,22,12],[53,94,66]])
b = np.array([[10,10,10]])
print(np.append(a, b, axis=0))


# In[347]:


#Find the dot product of the above two matrix
import numpy as np
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
z=np.dot(x,y)
print(z)


# In[350]:


#Generate a matrix of 20 random values and find its cumulative sum
import numpy as np
arr=np.random.random(20)
print(arr)
print("After Sum")
arrresult=np.cumsum(arr)
print(arrresult)


# In[ ]:




