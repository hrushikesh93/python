# imports
from sympy import symbols, Eq
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.simplefilter('ignore')


def gradient_descent(data, initial_guess, multiplier , precesion , max_iter):
    
    x = data
    fx = input("enter f(x) equation:")
    dfx = input("enter df(x) equation:")
    
    def f(x):
        return eval(f"{fx}")
    
    def df(x):
        return eval(f"{dfx}")
    
 #def gradient_descent2(derivative_func , initial_guess, multiplier , precesion , max_iter):
    
    new_x = initial_guess
    gamma = multiplier
    precision = precesion
    x_list = [new_x]
    slope_list = [df(new_x)]

    for i in range(max_iter):
        previous_x = new_x
        
        slope = df(previous_x)
        
        new_x = previous_x - gamma*slope

        x_list.append(new_x)
        
        slope_list.append(df(new_x))

        step_size = abs(new_x - previous_x)

        if step_size < precision:
            print("the loop ran these many times:....",i,'\n')
            break

    plt.figure(figsize=(18,8))
    plt.subplot(2,2,3)
    plt.plot(x, f(x))
    values = np.array(x_list)
    plt.scatter(x_list,f(values), color = "black" , alpha=0.6)
    plt.xlim(-1,2)
    plt.ylim(-4,4)
    plt.grid()

    plt.subplot(2,2,4)
    plt.plot(x, df(x) , color="pink")
    plt.scatter(x_list ,slope_list,s=100, color = "black" , alpha=0.6)
    plt.xlim(-1,2)
    plt.ylim(-4,4)
    plt.grid()

    plt.show()



def simple_linear_regression(x,y):
    myregr = LinearRegression()
    myregr.fit(x,y)
    y_predict = myregr.predict(x)
    plt.scatter(x,y,color='green')
    plt.plot(x,y_predict,color='black')
    
