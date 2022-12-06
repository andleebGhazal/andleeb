import numpy as np
data1=np.array([6,7.5,8,0])        # array of dimension 1 x 4
arr= np.array([1,5,9,15])         # array of dimension 1 x 4
v=data1+arr
print("1d array of 1 x 4 ",v)
data2=np.array([[6,7.5,8,0],       # matrix of dimension 5 x 4
                [1,4,8,4],
                [3.8,7,50,21],
                [2,9,6,4],
                [4,9,0.5,34]])
x=data2+arr
print("2d matrix of 5 x 4 ",x)
data3=np.zeros((15,3,4))          # matrix of dimension 15x 3 x 4
y=data3+arr    
print("3d matrix of 15 x 3x 4 ",y)

  