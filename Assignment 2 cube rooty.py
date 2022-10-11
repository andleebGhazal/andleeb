# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:24:52 2022

@author: Students Computers
"""
cube = 0.5
epsilon = 0.01;
num_guesses = 0;
low = 0;
high = cube;
guess = (high + low)/2.0;
if cube == 1:
 print('The cube root of',cube,'is 1') 
if cube == 0:
 print('The cube root of',cube,'is 0')
if abs(cube) > 1:
  while abs(guess**3 - cube) >= epsilon:
    if guess**3 < cube:
        low = guess
    else:
     high = guess
     guess = (high + low)/2.0
     num_guesses += 1    
if abs(cube) < 1:
    low = cube
    high = 1
    guess = (high + low) / 2.0
    while abs(guess**3 - cube) >= epsilon:
      if guess**3 < cube:
          low = guess
      else:
          high = guess
      guess = (high + low)/2.0
      num_guesses += 1
print(num_guesses)
print('The cube root of',cube,'is closest to',guess)



