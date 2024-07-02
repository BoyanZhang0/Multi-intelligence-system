import numpy as np

if __name__ == '__main__':
    x = np.zeros(100)
    y = np.zeros(100)
    x[0] = -2
    y[0] = -2
    for i in range(99):
        x[i + 1] = x[i]
        y[i + 1] = y[i]


    for i in range(100):
        print(x[i])
        print(y[i])
