# import packages
from copy import deepcopy
import os
from scipy.stats import levy
import numpy as np
from cec17_functions import cec17_test_func
import math


DimSize = 10  # the number of variables

DuckPopSize = 60
DuckPop = np.zeros((DuckPopSize, DimSize))
FitDuck = np.zeros(DuckPopSize)
BestDuck = np.zeros(DimSize)
FitBestDuck = 0

FishPopSize = 40
FishPop = np.zeros((FishPopSize,DimSize))
FitFish = np.zeros(FishPopSize)
BestFish = np.zeros(DimSize)
FitBestFish = 0

Prey = np.zeros(DimSize)

CurrentBest = np.zeros(DimSize)
FitCurrentBest = 0



TotalPopSize = DuckPopSize + FishPopSize

LB = [-100] * DimSize  # the maximum value of the variable range
UB = [100] * DimSize  # the minimum value of the variable range
Trials = 30  # the number of independent runs
MaxFEs = 1000 * DimSize  # the maximum number of fitness evaluations


Fun_num = 1  # the serial number of benchmark function
curIter = 0  # the current number of generations
MaxIter = math.ceil(MaxFEs / TotalPopSize)


FuncNum = 0
SuiteName = "CEC2020"


def fitness(X):
    global DimSize, FuncNum
    f = [0]
    cec17_test_func(X, f, DimSize, 1, FuncNum)
    return f[0]


def Initialization():
    global DuckPop,FitDuck,FishPop,FitFish,BestDuck, BestFish, Prey, CurrentBest, FitCurrentBest,FitBestFish,FitBestDuck,CurrentBest,FitCurrentBest
    DuckPop = np.zeros((DuckPopSize, DimSize))
    FitDuck = np.zeros(DuckPopSize)
    BestDuck = np.zeros(DimSize)
    FishPop = np.zeros((FishPopSize, DimSize))
    FitFish = np.zeros(FishPopSize)
    BestFish = np.zeros(DimSize)
    Prey = np.zeros(DimSize)
    CurrentBest = np.zeros(DimSize)
    FitCurrentBest = 0
    # randomly generate individuals
    for i in range(DuckPopSize):
        for j in range(DimSize):
            DuckPop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
            # calculate the fitness of the i-th individual
        FitDuck[i] = fitness(DuckPop[i])
    for i in range(FishPopSize):
        for j in range(DimSize):
            FishPop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
            # calculate the fitness of the i-th individual
        FitFish[i] = fitness(FishPop[i])
    BestDuck = DuckPop[np.argmin(FitDuck)]
    FitBestDuck = np.min(FitDuck)

    BestFish = FishPop[np.argmin(FitFish)]
    FitBestFish = np.min(FitFish)
    if FitBestFish > FitBestDuck:
        CurrentBest = np.copy(BestDuck)
        FitCurrentBest = FitBestDuck
    else:
        CurrentBest = np.copy(BestFish)
        FitCurrentBest = FitBestFish


def Evaluation():
    global DuckPop,FitDuck,BestDuck,FitBestDuck, FishPop,FitFish, BestFish, FitBestFish, Prey, CurrentBest, FitCurrentBest
    #find the best duck and fish
    BestDuck = DuckPop[np.argmin(FitDuck)]
    FitBestDuck = np.min(FitDuck)

    BestFish = FishPop[np.argmin(FitFish)]
    FitBestFish = np.min(FitFish)

    Off = np.zeros(DimSize)

    if FitBestFish > FitBestDuck:
        CurrentBest = np.copy(BestDuck)
        FitCurrentBest = FitBestDuck
    else:
        CurrentBest = np.copy(BestFish)
        FitCurrentBest = FitBestFish


    if np.random.random() < 0.5:
        Prey = np.copy(BestFish)
    else:
        Prey = np.copy(BestDuck)
    for i in range(DuckPopSize):
        index = np.random.randint(0, FishPopSize)
        Avg_FitFish = FitFish[index]
        diversity = np.mean(np.std(DuckPop, axis=0))
        if FitDuck[i] < Avg_FitFish:
            Off = DuckPop[i] + np.random.random()*(Prey - DuckPop[i]) * np.sin(2*np.pi*np.random.random())*(1 - curIter / MaxIter) * diversity
        else:

            for j in range(DimSize):
                if np.random.random() < 0.5:
                    Off[j] = DuckPop[i][j] + np.random.normal()*(1 - curIter / MaxIter) * diversity
                else:
                    Off[j] = FishPop[index][j] + np.random.normal()*(1 - curIter / MaxIter) * diversity
        Off = np.clip(Off, LB,UB)
        FitOff = fitness(Off)
        if FitOff < FitDuck[i]:
            DuckPop[i] = np.copy(Off)
            FitDuck[i] = FitOff
        if FitOff < FitBestDuck:
            BestDuck = np.copy(Off)
            FitBestDuck = FitOff
    for i in range(FishPopSize):
        Off = BestFish - 0.2 * np.random.uniform(-1,1) * (BestDuck - FishPop[i]) + 0.2*np.random.uniform() * (DuckPop[np.random.randint(0, DuckPopSize)] - FishPop[i])
        Off = np.clip(Off,LB,UB)
        FitOff = fitness(Off)
        if FitOff < FitFish[i]:
            FishPop[i] = np.copy(Off)
            FitFish[i] = FitOff
        if FitOff < FitBestFish:
            BestFish = np.copy(Off)
            FitBestFish = FitOff
    if FitBestFish > FitBestDuck:
        CurrentBest = np.copy(BestDuck)
        FitCurrentBest = FitBestDuck
    else:
        CurrentBest = np.copy(BestFish)
        FitCurrentBest = FitBestFish


def Run():
    global curIter, MaxFEs, TrialRuns, DimSize
    Trace = []
    curIter = 0
    Initialization()
    curIter = 1
    print("Iter: ", curIter, "Best: ", FitCurrentBest)
    Trace.append(FitCurrentBest)
    while curIter < MaxIter:
        Evaluation()
        curIter += 1
        Trace.append(FitCurrentBest)
        print("Iter: ", curIter, "Best: ", FitCurrentBest)
    return Trace


def main(dim):
    global DimSize, LB, UB, MaxFEs, MaxIter, Trials,FuncNum
    DimSize = dim
    LB = [-100] * dim
    UB = [100] * dim

    PopSize = 100
    MaxFEs = 1000 * dim
    MaxIter = int(MaxFEs / PopSize)

    for i in range(1, 31):
        FuncNum = i
        if i == 2:
            continue
        All_Trial_Best = []
        for j in range(Trials):
            np.random.seed(2023 + 920 * j)
            Trace = Run()
            All_Trial_Best.append(Trace)
        np.savetxt("./CIA_Data/CEC2017/F" + str(i) + "_" + str(dim) + "D.csv", All_Trial_Best, delimiter=",")


if __name__ == "__main__":
    if os.path.exists('CIA_Data/CEC2017') == False:
        os.makedirs('CIA_Data/CEC2017')
    Dims = [10, 30, 50, 100]
    for dim in Dims:
        main(dim)