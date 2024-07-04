# coding:UTF-8
'''
Created by Yuefeng XU (xyf20070623@gmail.com) on October 1, 2023
benchmark function: 10 functions of the CEC2020 test suite (https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2017/CEC2017.htm)
'''

# import packages
import os
import math
from opfunu.cec_based.cec2020 import *
import numpy as np
import requests


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
TrialRuns = 30  # the number of independent runs
MaxFEs = 1000 * DimSize  # the maximum number of fitness evaluations


Fun_num = 1  # the serial number of benchmark function
curIter = 0  # the current number of generations
MaxIter = math.ceil(MaxFEs / TotalPopSize)


FuncNum = 0
SuiteName = "CEC2020"


def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def Initialization(func):
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
        FitDuck[i] = func.evaluate(DuckPop[i])
    for i in range(FishPopSize):
        for j in range(DimSize):
            FishPop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
            # calculate the fitness of the i-th individual
        FitFish[i] = func.evaluate(FishPop[i])
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


def Evaluation(func):
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
        print(diversity)
        if FitDuck[i] < Avg_FitFish:
            Off = DuckPop[i] + np.random.random()*(Prey - DuckPop[i]) * np.sin(2*np.pi*np.random.random())*(1 - curIter / MaxIter) * diversity
        else:

            for j in range(DimSize):
                if np.random.random() < 0.5:
                    Off[j] = DuckPop[i][j] + np.random.normal()*(1 - curIter / MaxIter) * diversity
                else:
                    Off[j] = FishPop[index][j] + np.random.normal()*(1 - curIter / MaxIter) * diversity
        Off = np.clip(Off, LB,UB)
        FitOff = func.evaluate(Off)
        if FitOff < FitDuck[i]:
            DuckPop[i] = np.copy(Off)
            FitDuck[i] = FitOff
        if FitOff < FitBestDuck:
            BestDuck = np.copy(Off)
            FitBestDuck = FitOff
    for i in range(FishPopSize):
        Off = BestFish - 0.2 * np.random.uniform(-1,1) * (BestDuck - FishPop[i]) + 0.2*np.random.uniform() * (DuckPop[np.random.randint(0, DuckPopSize)] - FishPop[i])
        Off = np.clip(Off,LB,UB)
        FitOff = func.evaluate(Off)
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


def Run(func):
    global curIter, MaxFEs, TrialRuns, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        np.random.seed(2023 + 920 * i)
        Best_list = []
        curIter = 0
        Initialization(func)
        curIter = 1
        print("Iter: ", curIter, "Best: ", FitCurrentBest)
        Best_list.append(FitCurrentBest)

        while curIter < MaxIter:
            Evaluation(func)
            curIter += 1
            Best_list.append(FitCurrentBest)
            print("Iter: ", curIter, "Best: ", FitCurrentBest)
        All_Trial_Best.append(Best_list)
    np.savetxt("./CIA_Data/CEC2020/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best,
               delimiter=",")


def main(dim):
    global FuncNum, DimSize, MaxFEs, MaxIter,SuiteName, LB, UB
    DimSize = dim
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / TotalPopSize)
    LB = [-100] * dim
    UB = [100] * dim

    CEC2020 = [F12020(DimSize), F22020(DimSize), F32020(DimSize), F42020(DimSize), F52020(DimSize), F62020(DimSize),
               F72020(DimSize), F82020(DimSize), F92020(DimSize), F102020(DimSize)]

    FuncNum = 0
    for i in range(len(CEC2020)):
        FuncNum = i + 1
        Run(CEC2020[i])


if __name__ == "__main__":
    if os.path.exists('./CIA_Data/CEC2020') == False:
        os.makedirs('./CIA_Data/CEC2020')
    Dims = [10, 30, 50, 100]
    for Dim in Dims:
        main(Dim)


