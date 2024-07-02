import numpy as np
import matplotlib.pyplot as plt

N = 1e7  # 人口

def SI():
    T = 30  # 时间
    beta = 0.9
    s = np.zeros([T])  # 易感染比例
    i = np.zeros([T])  # 感染比例

    # 初始化
    i[0] = 45 / N
    s[0] = 1 - i[0]

    for t in range(T - 1):
        s[t + 1] = s[t] - beta * i[t] * s[t]
        i[t + 1] = 1 - s[t + 1]
    return s, i

def SIS():
    T = 80  # 时间
    s = np.zeros([T])
    i = np.zeros([T])
    beta = 0.9
    gamma = 0.5

    # 初始化
    i[0] = 45.0 / N
    s[0] = 1 - i[0]

    for t in range(T - 1):
        s[t + 1] = s[t] - beta * s[t] * i[t] + i[t] * gamma
        i[t + 1] = i[t] + beta * s[t] * i[t] - i[t] * gamma
    return s, i

def SIR():
    T = 80  # 时间
    s = np.zeros([T])
    i = np.zeros([T])
    r = np.zeros([T])
    beta = 0.9
    gamma = 0.5
    # 初始化
    i[0] = 10.0 / N
    s[0] = (1e7 - 10) / N

    for t in range(T - 1):
        s[t + 1] = s[t] - beta * s[t] * i[t]
        i[t + 1] = i[t] + beta * s[t] * i[t] - gamma * i[t]
        r[t + 1] = r[t] + gamma * i[t]
    return s, i, r

def SEIR():
    T = 170
    s = np.zeros([T])
    e = np.zeros([T])
    i = np.zeros([T])
    r = np.zeros([T])
    beta = 0.9
    gamma = 0.5
    sigma = 0.25

    # 初始化
    i[0] = 10.0 / N
    s[0] = (1e7 - 50) / N
    e[0] = 40.0 / N

    for t in range(T - 1):
        s[t + 1] = s[t] - beta * s[t] * i[t]
        e[t + 1] = e[t] + beta * s[t] * i[t] - sigma * e[t]
        i[t + 1] = i[t] + sigma * e[t] - gamma * i[t]
        r[t + 1] = r[t] + gamma * i[t]
    return s, e, i, r

if __name__ == "__main__":
    s1, i1 = SI()
    s2, i2 = SIS()
    s3, i3, r3 = SIR()
    s4, e4, i4, r4 = SEIR()
    # SI
    plt.figure()
    plt.subplot(221)
    plt.plot(s1, label='s')
    plt.plot(i1, label='i')
    plt.xlabel('Day')
    plt.ylabel('Infective Ratio')
    plt.legend()
    # SIS
    plt.subplot(222)
    plt.plot(s2, label='s')
    plt.plot(i2, label='i')
    plt.xlabel('Day')
    plt.ylabel('Infective Ratio')
    plt.legend()
    # SIR
    plt.subplot(223)
    plt.plot(s3, label='s')
    plt.plot(i3, label='i')
    plt.plot(r3, label='r')
    plt.xlabel('Day')
    plt.ylabel('Infective Ratio')
    plt.legend()
    # SEIR
    plt.subplot(224)
    plt.plot(s4, label='s')
    plt.plot(e4, label='e')
    plt.plot(i4, label='i')
    plt.plot(r4, label='r')
    plt.xlabel('Day')
    plt.ylabel('Infective Ratio')
    plt.legend()

    plt.show()
