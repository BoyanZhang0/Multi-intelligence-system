import numpy as np
import math
import matplotlib.pyplot as plt

T = 20
p = 0.5
origin_c = 1000
origin_d = 1
last_c = np.zeros(T)
last_d = np.zeros(T)
last_c[0] = origin_c
last_d[0] = origin_d
def payoff(a, b):
    # 用0表示C策略，1表示D策略
    if a == 0 and b == 0:
        return 2
    elif a == 0 and b == 1:
        return -1
    elif a == 1 and b == 0:
        return 3
    else:
        return 0

def evolution(c, d, i, payoff):
    payoff_c = c[i] * payoff(0, 0) + d[i] * payoff(0, 1)
    payoff_d = c[i] * payoff(1, 0) + d[i] * payoff(1, 1)
    if payoff_c >= payoff_d:
        d[i + 1] = math.floor(d[i] * (1 - p))
        c[i + 1] = c[i] + d[i] - d[i + 1]
    else:
        c[i + 1] = math.floor(c[i] * (1 - p))
        d[i + 1] = d[i] + c[i] - c[i + 1]
if __name__ == '__main__':
    time = 0
    while last_c[time] != 0 and last_d[time] != 0:
        evolution(last_c, last_d, time, payoff)
        time += 1

    while time < T:
        last_c[time] = last_c[time - 1]
        last_d[time] = last_d[time - 1]
        time += 1

    plt.figure()
    plt.plot(last_c, label='C')
    plt.plot(last_d, label='D')
    plt.xlabel('day')
    plt.ylabel('number')
    plt.legend()
    plt.show()
