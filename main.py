import timeit
from functools import partial
import numpy as np
import csv
import math

# 1. Differentialligning: dy/dx = y
def dif1(x, y):
    return y

def sol1(x):
    # Løsning: y(x) = exp(x-1)  (da y(1)=exp(0)=1)
    return math.exp(x - 1)

# 2. Differentialligning: dy/dx = x
def dif2(x, y):
    return x

def sol2(x):
    # Løsning: y(x) = x**2/2 + 1/2  (y(1)=0.5+0.5=1)
    return (x**2) / 2.0 + 1/2.0

# 3. Differentialligning: dy/dx = x + y
def dif3(x, y):
    return x + y

def sol3(x):
    # Løsning: y(x) = 3*exp(x-1) - x - 1  
    # Tjek: y(1)=3*1 -1 -1 =1
    return 3 * math.exp(x - 1) - x - 1

# 4. Differentialligning: dy/dx = y**2
def dif4(x, y):
    return y ** 2

def sol4(x):
    if x == 2:
        return math.nan
    # Løsning: y(x) = 1/(2-x)  (y(1)=1/(2-1)=1)
    return 1 / (2 - x)

# 5. Differentialligning: dy/dx = sin(x)*y
def dif5(x, y):
    return math.sin(x) * y

def sol5(x):
    # Løsning: y(x) = exp(cos(1)-cos(x))  (y(1)=exp(cos(1)-cos(1))=1)
    return math.exp(math.cos(1) - math.cos(x))

# 6. Differentialligning: dy/dx = y/x
def dif6(x, y):
    if x==0:
        return math.nan
    return y / x

def sol6(x):
    # Løsning: y(x) = x  (y(1)=1)
    return x

# 7. Differentialligning: dy/dx = x*y
def dif7(x, y):
    return x * y

def sol7(x):
    # Løsning: y(x) = exp((x**2 - 1)/2)  (y(1)=exp(0)=1)
    return math.exp((x**2 - 1) / 2.0)

# 8. Differentialligning: dy/dx = y*log(y)
def dif8(x, y):
    if y<=0:
        return 0
    return y * math.log(y)

def sol8(x):
    # Løsning: y(x) = 1  
    # y(x)=1 er en konstant løsning, da log(1)=0.
    return 1

# 9. Differentialligning: dy/dx = x**2*y
def dif9(x, y):
    return x**2 * y

def sol9(x):
    # Løsning: y(x) = exp((x**3 - 1)/3)  (y(1)=exp(0)=1)
    return math.exp((x**3 - 1) / 3.0)

# 10. Differentialligning: dy/dx = x - y
def dif10(x, y):
    return x - y

def sol10(x):
    # Løsning: y(x) = (x-1) + exp(1-x) 
    # Tjek: y(1) = (1-1) + exp(0) = 1.
    return (x - 1) + math.exp(1 - x)

de_pairs = [
#    [dif1, sol1],
    [dif2, sol2],
    [dif3, sol3],
    [dif4, sol4],
    [dif5, sol5],
#    [dif6, sol6],
    [dif7, sol7],
    [dif8, sol8],
#    [dif9, sol9],
    [dif10, sol10]
]

def eulerMethod(func, iterations: int, step: float, x0: float, y0: float):
    XValues = x0 + step * np.arange(iterations + 1)
    YValues = np.zeros(iterations + 1)
    YValues[0] = y0

    for i in range(iterations):
        YValues[i + 1] = YValues[i] + func(XValues[i], YValues[i]) * step

    return XValues, YValues


def eulerImprovedMethod(func, iterations, step, x0, y0):
    XValues = x0 + step * np.arange(iterations + 1)
    YValues = np.zeros(iterations + 1)
    YValues[0] = y0

    for i in range(iterations):
        YNextEstimate = YValues[i]+func(XValues[i],YValues[i])*step
        aMid = (func(XValues[i],YValues[i])+func(XValues[i+1],YNextEstimate))/2
        YValues[i+1]=YValues[i]+aMid*step
    
    return XValues, YValues


def RK4(func, iterations, step,x0,y0):
    XValues = x0 + step * np.arange(iterations + 1)
    YValues = np.zeros(iterations + 1)
    YValues[0] = y0

    for i in range(iterations):
        k1 = func(XValues[i],YValues[i])
        k2 = func(XValues[i] + 1/2 * step, YValues[i] + 1/2 * step * k1)
        k3 = func(XValues[i] + 1/2 * step, YValues[i] + 1/2 * step * k2)
        k4 = func(XValues[i] + step, YValues[i] + k3 * step)

        k = (k1+2*k2+2*k3+k4)/6

        YValues[i+1] = YValues[i] + k*step
    return XValues,YValues

def timing(iterations:int, func,functionIters:int,step:float,x0:float,y0:float)->dict[str,]:
    eulerTimer = timeit.Timer(partial(eulerMethod, func, functionIters, step,x0,y0))
    eulerImprovedTimer = timeit.Timer(partial(eulerImprovedMethod, func, functionIters, step,x0,y0))
    RK4Timer = timeit.Timer(partial(RK4,func, functionIters, step,x0,y0))
    eulerTime = eulerTimer.timeit(number=iterations)
    eulerImprovedTime = eulerImprovedTimer.timeit(number=iterations)
    RK4Time = RK4Timer.timeit(number=iterations)

    return {
        "Euler Method": eulerTime,
        "Euler Improved Method": eulerImprovedTime,
        "RK4 Method": RK4Time
    }

def accuracy(func, iterations: int, step: float, x0: float, y0: float, solution):
    eulerResults = eulerMethod(func, iterations, step, x0, y0)
    eulerImprovedResults = eulerImprovedMethod(func, iterations, step, x0, y0)
    RK4Results = RK4(func, iterations, step, x0, y0)

    XActualResults = x0 + step * np.arange(iterations + 1)
    YActualResults = np.array([solution(x) for x in XActualResults])

    def compute_percentage_diff(approx_values):
        return np.abs((approx_values - YActualResults) / YActualResults) * 100

    eulerError = compute_percentage_diff(eulerResults[1])
    eulerImprovedError = compute_percentage_diff(eulerImprovedResults[1])
    RK4Error = compute_percentage_diff(RK4Results[1])

    return {
        "Euler Method": eulerError,
        "Euler Improved Method": eulerImprovedError,
        "RK4 Method": RK4Error
    }


def main():

    rows = []
    header = [
        "Equation", 
        "Iteration", 
        "Euler Time", 
        "Euler Improved Time", 
        "RK4 Time",
        "Euler Error", 
        "Euler Improved Error", 
        "RK4 Error"
    ]
    rows.append(header)

    for eq_idx, de in enumerate(de_pairs, start=1):
        time_dict = timing(10000, de[0], 100, 0.2, 1, 1)
        diff_dict = accuracy(de[0], 100, 0.2, 1, 1, de[1])
        num_iterations = len(diff_dict["Euler Method"])

        for i in range(num_iterations):
            row = [
                f"Equation {eq_idx}",
                i,
                time_dict["Euler Method"],
                time_dict["Euler Improved Method"],
                time_dict["RK4 Method"],
                diff_dict["Euler Method"][i],
                diff_dict["Euler Improved Method"][i],  
                diff_dict["RK4 Method"][i]
            ]
            rows.append(row)

    with open("results.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)

if __name__ == "__main__":
    main()
