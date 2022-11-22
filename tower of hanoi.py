# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 23:01:49 2022

@author: Students Computers
"""
n=3
#s="starting point"
#m="middle"
#d="destination"
s='Sourse'
m='middle'
d='Destination'
k=(2**n)-1
if n%2==0:
  te=d
  d=m
  m=te   
for i in range(1,k+1):
 if i==5:
     print("block move from ",m,"to ",s)
 if i==6:
     print("block move from ",m,"to ",d)
 if i%3==1:
     print("block move from ",s,"to ",d)
 elif i%3==2 and i!=5:
     print("block move from ",s,"to ",m)
 elif i%3==0 and i!=6:
     print("block move from ",d,"to ",m)