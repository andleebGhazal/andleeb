# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 22:59:55 2022

@author: Students Computers
"""

import turtle 
x=turtle.Turtle() 
def square(angle):
    x.forward(5) 
    x.right(angle) 
    x.forward(5) 
    x.right(angle) 
    x.forward(5) 
    x.right(angle)
    x.forward(5) 
    x.right(angle+5)
for i in range(40): 
    square(1)

