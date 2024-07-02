import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
uPos = [
    [1, 0],
    [1, 1],
    [1, -1],
    [0, 0],
    [0, 1],
    [0, -1],
    [-1, 0],
    [-1, 1],
    [-1, -1]
]

uPredatorPos = [
    [0, 0.1]
]

sensPeriod = 10

consSpeed = 0.8

left = -1.5
front = -1.5
width, depth = 3, 3
right, back = left + width, front + depth
height = 0.5

N = 9
M = 1

boidsList = []
predatorsList = []

inputVector = [0.0, 0.0, 0.0]

v_Max = 1.0
nearbyDis = 2
avoidDist = 0.6
alignDist = 1
cohesionDist = 2
mustAvoidDist = 0.35
avoidPredatorDis = 0.6

minDist = 1000
limPos = [2, 2, 2, 2, 1, 1]

boidF = []
predatorF = []

uBoid = []
uPredator = []
def mySqrt(n):
    res = n
    last = (res + n/res)/2
    while abs(res-last)>0.001:
        last = res
        res = (res + n/res)/2
    return 1/res

def rqrt(n):  # 开平方根
    number = np.array([n])
    y = number.astype(np.float32)  # 改变np.array中所有元素的数据类型
    x2 = y * 0.5
    i = y.view(np.int32)  # 改变数据类型
    i[:] = 0x5f3759df - (i >> 1)
    y = y * (1.5 - x2 * y * y)
    return y[0]

def vector_sub(x, y):
    z = [0, 0, 0]
    z[0] = x[0] - y[0]
    z[1] = x[1] - y[1]
    z[2] = x[2] - y[2]
    return z

def vector_add(x, y):
    z = [0, 0, 0]
    z[0] = x[0] + y[0]
    z[1] = x[1] + y[1]
    z[2] = x[2] + y[2]
    return z

def vector_div(x, y):
    z = [0, 0, 0]
    z[0] = x[0] / y
    z[1] = x[1] / y
    z[2] = x[2] / y
    return z

def vector_mult(x, y):
    z = [0, 0, 0]
    z[0] = x[0] * y
    z[1] = x[1] * y
    z[2] = x[2] * y
    return z

def vector_null(x):
    if(x[0] == 0 and x[1] == 0 and x[2] == 0):
        return True
    else:
        return False

def vector_mag(x):#速度大小
    return np.sqrt(math.pow(x[0], 2) + math.pow(x[1], 2) + math.pow(x[2], 2))

def vector_norm(x):#速度方向单位向量
    num = x[0] * x[0] + x[1] * x[1] + x[2] * x[2]#速度大小平方
    #invSqrt = InvSqrt.InvSqrt(ctypes.c_float(num))#平方根倒数
    invSqrt=mySqrt(num)
    #mag = vector_mag(x)
    z = [0, 0, 0]
    z = vector_mult(x, invSqrt)
    return z

def vector_set_mag(x, mag):
    z = [0, 0, 0]
    z = vector_norm(x)#x单位向量
    z = vector_mult(z, mag)#单位向量的mag倍
    return z

def vector_degree(x, y):#速度x,y夹角
    if vector_null(x) == True or vector_null(y) == True:
        return 0
    z = (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]) / (vector_mag(x) * vector_mag(y))
    if z > 1.0:
        z = 1.0
    if z < -1.0:
        z = -1.0
    z = math.acos(z) * 180 / math.pi
    return z

def vector_cross_product(x, y):#叉乘
    z = [0, 0, 0]
    z[0] = x[1] * y[2] - x[2] * y[1]
    z[1] = x[2] * y[0] - x[0] * y[2]
    z[2] = x[0] * y[1] - x[1] * y[0]
    if vector_null(z) == True:#叉乘等于0，其中一向量为0，或两向量平行或反向
        if x[2] == 0:
            z = [0, 0, 1]
        else:
            z = [random.uniform(0, 1), random.uniform(0, 1), 0]#random.uniform(0, 1)返回0-1的随机浮点数
            z[2] = -(z[0] * x[0] + z[1] * x[1]) / x[2]
    #z = [0, 0, 1]
    return z

def matrix_mult(x, y):
    z = [0, 0, 0]
    z[0] = x[0][0] * y[0] + x[0][1] * y[1] + x[0][2] * y[2]
    z[1] = x[1][0] * y[0] + x[1][1] * y[1] + x[1][2] * y[2]
    z[2] = x[2][0] * y[0] + x[2][1] * y[1] + x[2][2] * y[2]
    return z

class Predator(object):  # 捕食者类
    def __init__(self):
        self.baseRotationSpeed = 170    #旋转角速度
        self.currRotationSpeed = 170
        self.baseSpeed = 5
        self.currSpeed = 5
        self.pos = [random.uniform(left, right), random.uniform(front, back), height]
        self.currDirection = [1, 0, 0]
        self.desiredDirection = [1, 0, 0]
        self.minDist = cohesionDist
        self.landing = False
        self.readyLanding = False
        self.time = 0.0

    def setData(self, position):
        print(self.pos)
        self.pos = [
            position[0],
            position[1],
            height
        ]
    def getData(self):
        pos = [
            self.pos[0],
            self.pos[1],
            self.pos[2]
        ]
        vel = [
            self.currDirection[0] * self.currSpeed,
            self.currDirection[1] * self.currSpeed,
            self.currDirection[2] * self.currSpeed
        ]
        return pos, vel

    def getLandingFlag(self):
        flag = False
        flag = self.readyLanding
        return flag

    def getNearbyBoids(self):
        nearbyBoids = []
        nearbyPredator = []
        global minDist
        for boid in boidsList:
            pos = [boid.pos[0], boid.pos[1], boid.pos[2]]
            d = vector_sub(pos, self.pos)
            dist = vector_mag(d)
            if dist > 0 and dist < nearbyDis:
                self.minDist = min(self.minDist, dist)
                nearbyBoids.append([boid.pos, dist, vector_mult(boid.currDirection, boid.currSpeed)])

        for predator in predatorsList:
            pos = [predator.pos[0], predator.pos[1], predator.pos[2]]
            d = vector_sub(pos, self.pos)
            dist = vector_mag(d)
            if dist > 0 and dist < avoidPredatorDis:
                self.minDist = min(self.minDist, dist)
                nearbyPredator.append([pos, dist])
        minDist = min(minDist, self.minDist)
        return nearbyBoids, nearbyPredator

    def boundCheckBorder(self):  #边界检测
        if self.minDist < mustAvoidDist:
            return [0, 0, 0]
        radiusX = width / 2 + left
        radiusY = depth / 2 + front
        checker = 0.85
        center = [radiusX, radiusY, height]

        centerOffset = vector_sub(center, self.pos)
        checkX = abs(centerOffset[0]) / (width / 2)
        checkY = abs(centerOffset[1]) / (depth / 2)

        xOrYorZ = max(checkX, checkY)

        if xOrYorZ < checker:
            return [0, 0, 0]

        if xOrYorZ <= 0.82:
            self.currRotationSpeed = 175
        elif xOrYorZ <= 0.85:
            self.currRotationSpeed = 200
        elif xOrYorZ <= 0.9:
            self.currRotationSpeed = 250
        elif xOrYorZ <= 0.92:
            self.currRotationSpeed = 300
        elif xOrYorZ <= 0.96:
            self.currRotationSpeed = 500
        elif xOrYorZ <= 0.98:
            self.currRotationSpeed = 1000
        else:
            self.currRotationSpeed = 1500

        return centerOffset

    def case(self, nearbyBoids):#聚集
        centerOfMass = [0, 0, 0]
        cnt = 0
        for boid in nearbyBoids:
            centerOfMass = vector_add(centerOfMass, boid[0])  #boid::[boid.pos, dist, vector_mult(boid.currDirection, boid.currSpeed)]
            cnt += 1
        if cnt != 0:
            centerOfMass = vector_div(centerOfMass, cnt)  #nearbyBoids的x,y,z方向质心
            centerOfMass = vector_sub(centerOfMass, self.pos)  #nearbyBoids的x,y,z方向质心相对self偏移量
            centerOfMass = vector_set_mag(centerOfMass, 0.5)#self-nearbyBoids质心方向单位向量的一半
        return centerOfMass

    def separation(self, nearbyPredator):#分离向量（单位向量）
        separationVelocity = [0, 0, 0]
        for predator in nearbyPredator:
            dist = predator[1]  #nearbyPredator到self的距离
            predatorPos = [predator[0][0], predator[0][1], predator[0][2]]
            dv = [self.pos[0] - predatorPos[0], self.pos[1] - predatorPos[1]]
            if dist < avoidDist:
                dv = vector_div(dv, math.pow(dist, 2))  #除
                separationVelocity = vector_add(separationVelocity, dv)
        if vector_null(separationVelocity) == False:
            separationVelocity = vector_set_mag(separationVelocity, 1)#单位向量
        return separationVelocity

    def callPredatorRules(self):
        limPos[0] = min(limPos[0], self.pos[0])
        limPos[1] = max(limPos[1], self.pos[0])
        limPos[2] = min(limPos[2], self.pos[1])
        limPos[3] = max(limPos[3], self.pos[1])
        limPos[4] = min(limPos[4], self.pos[2])
        limPos[5] = max(limPos[5], self.pos[2])

        self.minDist = cohesionDist
        nearbyBoids, nearbyPredator = self.getNearbyBoids()

        newVelocity = [0, 0, 0]
        newVelocity = vector_add(newVelocity, self.boundCheckBorder())


        if vector_null(newVelocity) == True and self.landing == False:
            self.currRotationSpeed = self.baseRotationSpeed
            self.currSpeed = self.baseSpeed
            newVelocity = vector_add(newVelocity, vector_mult(self.case(nearbyBoids), 0.5))
            newVelocity = vector_add(newVelocity, self.separation(nearbyPredator))
        else:
            newVelocity = vector_add(newVelocity, self.separation(nearbyPredator))
        if vector_null(newVelocity) == False:
            newVelocity = vector_set_mag(newVelocity, 1)
            self.desiredDirection = newVelocity
        else:
            self.desiredDirection = self.currDirection

    def joystickControl(self):
        file = open("predator.txt", "r")
        data = file.readlines()
        for i in range(40):
            inputVector = data[i]
            if vector_mag(inputVector) > 0.2:
                self.desiredDirection = inputVector
                self.desiredDirection = vector_norm(self.desiredDirection)
                self.currSpeed = self.baseSpeed
            else:
                self.desiredDirection = [0, 0, 0]
            time.sleep(2)

    def move(self):
        if vector_null(self.desiredDirection) == False:
            self.currDirection = vector_norm(self.desiredDirection)
            self.pos = vector_add(self.pos, vector_mult(self.desiredDirection, self.currSpeed * 0.02))

        pos = [self.pos[0], self.pos[1], self.pos[2]]
        speed = self.currSpeed
        return pos, speed


class Boid(object):
    def __init__(self):
        self.baseRotationSpeed = 170    #旋转角速度
        self.currRotationSpeed = 170
        self.baseSpeed = 3
        self.currSpeed = 3
        self.pos = [random.uniform(left, right), random.uniform(front, back), height]
        self.currDirection = [1, 0, 0]
        self.desiredDirection = [1, 0, 0]
        self.minDist = cohesionDist * cohesionDist
        self.landing = False
        self.readyLanding = False
        self.time = 0.0

    def setData(self, position):
        self.pos = [
            position[0],
            position[1],
            height
        ]
    def getData(self):
        pos = [
            self.pos[0],
            self.pos[1],
            self.pos[2]
        ]
        vel = [
            self.currDirection[0] * self.currSpeed,
            self.currDirection[1] * self.currSpeed,
            self.currDirection[2] * self.currSpeed
        ]
        return pos, vel

    def getLandingFlag(self):
        flag = False
        flag = self.readyLanding
        return flag

    def getNearbyBoids(self):
        nearbyBoids = []
        nearbyPredator = []
        nearbyDisPow = math.pow(nearbyDis, 2)
        for i in range(9):
            boid=boidsList[i]
            pos1 = [boid.pos[0], boid.pos[1], boid.pos[2]]
            d = vector_sub(pos1, self.pos)
            dist = math.pow(d[0], 2) + math.pow(d[1], 2) + math.pow(d[2], 2)
            if dist > 0 and dist < nearbyDisPow:
                self.minDist = min(self.minDist, dist)
                nearbyBoids.append([boid.pos, dist, vector_mult(boid.currDirection, boid.currSpeed)])
        avoidPredatorDisPow = math.pow(avoidPredatorDis, 2)

        for predator in predatorsList:
            pos2 = [float(predator.pos[0]), float(predator.pos[1]), float(predator.pos[2])]
            d = vector_sub(pos2, self.pos)
            dist = math.pow(d[0], 2) + math.pow(d[1], 2) + math.pow(d[2], 2)
            if dist > 0 and dist < avoidPredatorDisPow:
                self.minDist = min(self.minDist, dist)
                nearbyPredator.append([pos2, dist])
        self.minDist = math.sqrt(self.minDist)
        return nearbyBoids, nearbyPredator

    def boundCheckBorder(self):
        if self.minDist < mustAvoidDist:
            return [0, 0, 0]
        radiusX = width / 2 + left
        radiusY = depth / 2 + front
        checker = 0.85
        center = [radiusX, radiusY, height]

        centerOffset = vector_sub(center, self.pos)

        checkX = abs(centerOffset[0]) / (width / 2)
        checkY = abs(centerOffset[1]) / (depth / 2)
        checkZ = abs(centerOffset[2]) / (height / 2)

        xOrYorZ = max(checkX, checkY, checkZ)

        if xOrYorZ < checker:
            return [0, 0, 0]

        if xOrYorZ <= 0.82:
            self.currRotationSpeed = 175
        elif xOrYorZ <= 0.85:
            self.currRotationSpeed = 200
        elif xOrYorZ <= 0.9:
            self.currRotationSpeed = 250
        elif xOrYorZ <= 0.92:
            self.currRotationSpeed = 300
        elif xOrYorZ <= 0.96:
            self.currRotationSpeed = 500
        elif xOrYorZ <= 0.98:
            self.currRotationSpeed = 1000
        else:
            self.currRotationSpeed = 1500

        return centerOffset

    def scatter(self, nearbyPredator):  # 躲避捕食者
        separationVelocity = [0, 0, 0]

        for predator in nearbyPredator:
            dist = predator[1]
            predatorPos = predator[0]
            if self.minDist < mustAvoidDist and dist >= mustAvoidDist:
                continue

            dist = dist / avoidPredatorDis

            if dist > 0.625:
                self.currSpeed = 0.85
                self.currRotationSpeed = 200
            elif dist > 0.5:
                self.currSpeed = 0.9
                self.currRotationSpeed = 320
            elif dist > 0.4:
                self.currSpeed = 0.95
                self.currRotationSpeed = 460
            elif dist > 0.3:
                self.currSpeed = 1
                self.currRotationSpeed = 500
            elif dist > 0.25:
                self.currSpeed = 1.1
                self.currRotationSpeed = 550
            else:
                self.currSpeed = 1.2
                self.currRotationSpeed = 550

            separationVelocity = vector_add(separationVelocity, vector_div(vector_mult(vector_sub(predatorPos, self.pos), -1), math.pow(dist, 3)))


        if vector_null(separationVelocity) == False:
            separationVelocity = vector_set_mag(separationVelocity, 1.5)
        return separationVelocity

    def separation(self, nearbyBoids):  # 分离
        separationVelocity = [0, 0, 0]
        if self.minDist < mustAvoidDist:
            self.currRotationSpeed = 1000
            for boid in nearbyBoids:
                dist = boid[1]
                boidPos = [boid[0][0], boid[0][1], boid[0][2]]
                dv = [self.pos[0] - boidPos[0], self.pos[1] - boidPos[1], self.pos[2] - boidPos[2]]
                if dist < mustAvoidDist:
                    dv = vector_div(dv, math.pow(dist, 3))
                    separationVelocity = vector_add(separationVelocity, dv)
                    if self.currSpeed < 1:
                        self.currSpeed = 1
        else:
            for boid in nearbyBoids:
                dist = boid[1]
                boidPos = [boid[0][0], boid[0][1], boid[0][2]]
                dv = [self.pos[0] - boidPos[0], self.pos[1] - boidPos[1], self.pos[2] - boidPos[2]]
                if dist < avoidDist:
                    dv = vector_div(dv, math.pow(dist, 3))
                    separationVelocity = vector_add(separationVelocity, dv)
        if vector_null(separationVelocity) == False:
            separationVelocity = vector_norm(separationVelocity)
        return separationVelocity

    def allignment(self, nearbyBoids):  # 对齐
        nearbyVelocity = [0, 0, 0]
        if self.minDist < mustAvoidDist:
            return [0, 0, 0]
        for boid in nearbyBoids:
            dist = boid[1]
            if dist < alignDist:
                nearbyVelocity = vector_add(nearbyVelocity, boid[2])

        if vector_null(nearbyVelocity) == False:
            nearbyVelocity = vector_norm(nearbyVelocity)
        return nearbyVelocity

    def cohesion(self, nearbyBoids):  # 聚集
        centerOfMass = [0, 0, 0]
        cnt = 0
        if self.minDist < mustAvoidDist:
            return [0, 0, 0]
        for boid in nearbyBoids:
            dist = boid[1]
            if dist >= alignDist:
                centerOfMass = vector_add(centerOfMass, boid[0])
                cnt += 1


        if cnt != 0:
            centerOfMass = vector_div(centerOfMass, cnt)
            centerOfMass = vector_sub(centerOfMass, self.pos)
            centerOfMass = vector_set_mag(centerOfMass, 0.5)

        return centerOfMass

    def callBoidRules(self):   #调用以上方法计算出 desiredDirection 的
        limPos[0] = min(limPos[0], self.pos[0])
        limPos[1] = max(limPos[1], self.pos[0])
        limPos[2] = min(limPos[2], self.pos[1])
        limPos[3] = max(limPos[3], self.pos[1])
        limPos[4] = min(limPos[4], self.pos[2])
        limPos[5] = max(limPos[5], self.pos[2])

        self.minDist = cohesionDist
        nearbyBoids, nearbyPredator = self.getNearbyBoids()

        newVelocity = [0, 0, 0]

        newVelocity = vector_add(newVelocity, self.boundCheckBorder())
        newVelocity = vector_add(newVelocity, vector_mult(self.scatter(nearbyPredator), 0.5))
        if vector_null(newVelocity) == True and self.landing == False:
            self.currRotationSpeed = self.baseRotationSpeed
            self.currSpeed = self.baseSpeed

            newVelocity = vector_add(newVelocity, self.separation(nearbyBoids))
            newVelocity = vector_add(newVelocity, self.allignment(nearbyBoids))
            newVelocity = vector_add(newVelocity, self.cohesion(nearbyBoids))
        else:
            newVelocity = vector_add(newVelocity, self.separation(nearbyBoids))


        newVelocity[2] = 0.0
        if vector_null(newVelocity) == False:
            newVelocity = vector_norm(newVelocity)
            self.desiredDirection = newVelocity
        else:
            self.desiredDirection = [self.currDirection[0], self.currDirection[1], self.currDirection[2]]

    def move(self):   #根据 speed、ω 和desiredDirection 计算出下一时刻个体前进方向
        dRot = vector_degree(self.desiredDirection, self.currDirection)
        rotIncVal = min(self.currRotationSpeed * 0.02, dRot)
        vec_k = vector_cross_product(self.currDirection, self.desiredDirection)
        vec_k = vector_norm(vec_k)
        rotIncVal = rotIncVal * math.pi / 180
        sinRot = math.sin(rotIncVal / 2)
        cosRot = math.cos(rotIncVal / 2)
        x = vec_k[0] * sinRot
        y = vec_k[1] * sinRot
        z = vec_k[2] * sinRot
        w = cosRot
        R = [[1 - 2 * y * y - 2 * x * x, 2 * (x * y - z * w), 2 * (x * z + y * w)],
             [2 * (x * y + z * w), 1 - 2 * x * x - 2 * z * z, 2 * (y * z - x * w)],
             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * x * x - 2 * y * y]]

        self.currDirection = matrix_mult(R, self.currDirection)
        self.currDirection = vector_norm(self.currDirection)
        self.pos = vector_add(self.pos, vector_mult(self.currDirection, self.currSpeed * 0.02))
        pos = [self.pos[0], self.pos[1], self.pos[2]]
        speed = self.currSpeed
        return pos, speed


lock = threading.Lock()
class BoidThread(threading.Thread):
    def __init__(self, boid, file1):
        threading.Thread.__init__(self)
        self.boid = boid
        self.file1 = file1
    def run(self):
        self.boid.callBoidRules()
        pos, speed = self.boid.move()
        with lock:
            for n in range(3):
                self.file1.write(str(pos[n]))
                self.file1.write('\n')


def Pos():
    predator = Predator()
    predatorsList.append(predator)
    predatorsList[0].setData(uPredatorPos[0])
    for i in range(9):
        boid = Boid()
        boidsList.append(boid)
    for i in range(9):
        boidsList[i].setData(uPos[i])
    file1 = open("predator.txt", "r")
    data = file1.readlines()
    file2 = open("boid.txt", "w")
    for k in range(80):
        datas = [data[2 * k], data[2 * k + 1]]
        predatorsList[0].setData(datas)
        threads = []
        for j in range(9):
            thread = BoidThread(boidsList[j], file2)
            threads.append(thread)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

if __name__== "__main__" :
    print('')
    Pos()
    fig = plt.figure()
    # syntax for 3-D projection
    ax = plt.axes(projection='3d')
    file1 = open("boid.txt", "r")
    data1 = file1.readlines()
    file2 = open("predator.txt", "r")
    data2 = file2.readlines()

    def update(i):
        ax.cla()
        ax.set_xlim(left=-2, right=2)
        ax.set_ylim(bottom=-2, top=2)
        ax.set_zlim(-2, 2)
        # ax.scatter(float(data2[2 * i]), float(data2[2 * i + 1]), float(0.5), color='red')
        time.sleep(0.02)
        threads = []
        for j in range(9):
            t = threading.Thread(target=scatter_boid, args=(ax, data1[9 * i + 3 * j:9 * i + 3 * j + 3]))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    def scatter_boid(ax, data):
        ax.scatter(float(data[0]), float(data[1]), float(data[2]), color='black')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    anim = FuncAnimation(fig, update, frames=np.arange(0, 50), interval=100)
    plt.show()


