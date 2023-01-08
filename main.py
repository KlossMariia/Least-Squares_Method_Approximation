import matplotlib.pyplot as plt
import numpy.linalg as linalg
import numpy as np
import math

x = {}
xy = {}

# here is the function that needs to be approximated
def fun(x):
    f = math.log(math.cos(x/math.exp(2)))
    return f


def split(a, b, n):  # splitting the segment based on user's input
    # segment [a, b] with step h
    global x
    h = (b-a)/n
    while a <= b:
        xy[0].append(fun(a))
        x[1].append(a)
        a = round(a+h, 2)
    # drawing plot of the origin function
    print(f"x values: {x[1]}")
    print(f"Original function y values: {xy[0]}")
    plt.scatter(x[1], xy[0], s=10, c="red", label="Original function")


# this function solves equation system, which is expressed as matrix and vector
def solve_equations_system(matrix, vector):
    matrix = linalg.inv(matrix)
    matrix = np.array(matrix)
    vector = np.array(vector)
    result = matrix.dot(vector)
    result = [round(item, 8) for item in result]

    return result

# this fun gets approximated polynom as string
def get_function(coeffs:list):
    fun = f"{coeffs[len(coeffs)-1]}"
    for i in range(len(coeffs)-2, -1, -1):
        fun += f" + {coeffs[i]}*x ** {len(coeffs) - 1 - i}"
    return fun

# this function builds system of equations, based on least square method
# it also finds solution for this system and draws plot of approximated function
def approximation(stepen):
    for i in range(2, stepen*2 + 1):  #создаем массивы х, х^2, ... и находим сумму элементов массива
        for item in x[1]:
            x[i].append(item**i)
        x[i] = sum(x[i])
    for i in range(1, stepen + 1):    #создаём массивы xy, x^2y ... и находим сумму элементов в них
        for m in range(0, len(x[1])):
            xy[i].append(xy[0][m]*(x[1][m]**i))
        xy[i] = sum(xy[i])
    xy[0] = sum(xy[0])
    vector = []
    for i in range(stepen, -1, -1):
        vector.append(xy[i])
    print(x[1])
    vectorx = x[1]
    x[1] = sum(x[1])
    matrix = []
    for i in range(0, stepen+1):
        matrix.append([])
    for i in range(0, stepen+1):
        for j in range(0, stepen+1):
            matrix[i].append(x[stepen*2-j-i])
    result = solve_equations_system(matrix, vector)
    get_approximate_plot(result, vectorx)
    print(f"Vector with polynom coeffs: {result}")
    print(f"Result: {get_function(result)}")


# builds plot pf approximated function
def get_approximate_plot(coeffs: list, vectorx: list):
    vectory = []
    for x in vectorx:
        y = 0
        power = 0
        for m in range(len(coeffs)-1, -1, -1):
            y += coeffs[m] * x ** power
            power += 1
        vectory.append(y)
    print(f"Approximated function y values: {vectory}")
    plt.scatter(vectorx,vectory, c="green", s=6, label="approximated function")


print("Least squares method approximation")
print("Pleace input segment for approximation")
a = int(input("From а="))
b = int(input("To b="))
n = int(input("With number of splits n="))

#  there is a possibility by input of 2 arrays (x and y arrays)
# x[1] = input("Input x values, split by space = ").split()
# xy[0] = input("Input y values, split by space = ").split()
# n = len(x[1])
power = int(input("Polynom's power = "))

for i in range(1, power * 2 + 1):
    x[i] = []
for i in range(0, power + 1):
    xy[i] = []
x[0] = n
split(a, b, n)
approximation(power)

plt.grid(True)
plt.legend()
plt.show()


