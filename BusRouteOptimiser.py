import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import ReadData
import csv

fit=[0,0]
class RouteOptimizer:
    
    def __init__(self):
        self.busStops=None
        self.fleetSize=None
        self.linkData=None
        self.demandMat=None
        self.currPopulation=None
        self.maxRoute=None
        self.genCount=None
        self.startPopulation=None
        self.shortestTimeMat=None
        self.mutationProb=None
        self.copyCount=None
        
        
    def initialization(self):

        #making a list of tuples for activity of each bus stop
        activityLev=[[0,x] for x in range(len(self.busStops))]
        for x in range(len(self.busStops)):
            for i in range(len(self.busStops)):
                if i != x:
                    activityLev[x][0]+=self.demandMat[i][x]+self.demandMat[x][i]
        activityLev.sort()
        
        startPopulationSize=self.startPopulation
        self.currPopulation=[[] for x in range(startPopulationSize)]
        hyper_param_1=min(15,self.fleetSize)
        
        INS=activityLev[-(hyper_param_1):]
        probablity=[0 for x in range(len(INS))]
        sum=0
        for x in range(len(INS)):
            sum+=INS[x][0]
        for x in range(len(INS)):
            probablity[x]=INS[x][0]/sum
            
        INS_1D=[INS[x][1] for x in range(len(INS))]
        
        for x in range(startPopulationSize):
            currRoute=[[] for i in range(self.fleetSize)]
            probCopy=[x for x in probablity]
            sumCopy=sum
            for i in range(self.fleetSize):
                    currChoice=np.random.choice(INS_1D,1,p=probCopy)
                    currRoute[i].append(int(currChoice))
                    indx=INS_1D.index(currChoice)
                    sumCopy-=INS[indx][0]
                    probCopy[indx]=0
                    for y in range(len(INS)):
                        if probCopy[y]!=0:
                            probCopy[y]=INS[y][0]/sumCopy
                    if (i+1)%len(INS)==0:
                        probCopy=[x for x in probablity]
                        sumCopy=sum

            for i in range(self.fleetSize):
                currRouteSize=random.randint(math.floor(self.maxRoute*(3/4)),self.maxRoute)
                for y in range(1,currRouteSize):
                    tempActivityLevel=[[0,z] for z in range(len(self.busStops))]
                    for z in range(len(self.busStops)):
                        for k in range(len(self.busStops)):
                                if z != k and z not in currRoute[i]:
                                    tempActivityLevel[z][0]+=self.demandMat[k][z]+self.demandMat[z][k]

                    tempActivityLevel.sort()
                    VNS=tempActivityLevel[-(hyper_param_1):]
                    tempProb=[0 for z in range(len(VNS))]
                    sum_vns=0
                    for z in range(len(VNS)):
                        sum_vns+=VNS[z][0]
                    if sum_vns==0:
                        for z in range(len(self.busStops)):
                            for k in range(len(self.busStops)):
                                if z != k:
                                    tempActivityLevel[z][0]+=self.demandMat[k][z]+self.demandMat[z][k]
                        
                        tempActivityLevel.sort()
                        VNS=tempActivityLevel[-(hyper_param_1):]
                        tempProb=[0 for z in range(len(VNS))]
                        sum_vns=0
                        for z in range(len(VNS)):
                            sum_vns+=VNS[z][0]
                    for z in range(len(VNS)):
                        tempProb[z]=VNS[z][0]/sum_vns
                    VNS_1D=[VNS[y][1] for y in range(len(VNS))]
                    currChoice=np.random.choice(VNS_1D,1,p=tempProb)
                    currRoute[i].append(int(currChoice))
                        
            self.currPopulation[x]=currRoute
            
            m=len(self.demandMat)
            self.shortestTimeMat=[0 for x in range(m)]
            for i in range(m):
                k=i
                cost=[[0 for x in range(m)] for x in range(1)]
                offsets = []
                offsets.append(k)
                elepos=0
                for j in range(m):
                    cost[0][j]=self.linkData[k][j]
                mini=999
                for x in range (m-1):
                    mini=999
                    for j in range (m):
                            if cost[0][j]<=mini and j not in offsets:
                                    mini=cost[0][j]
                                    elepos=j
                    offsets.append(elepos)
                    for j in range (m):
                        if cost[0][j] >cost[0][elepos]+self.linkData[elepos][j]:
                            cost[0][j]=cost[0][elepos]+self.linkData[elepos][j]
                self.shortestTimeMat[i]=cost     
                    
    def evaluation(self,routeSet,store=0):
        
        totalFit=0
        l=len(routeSet)
        
        demandFulfilled=0
        totalDemand=0
        
        #calculating number of people who fulfilled their transport need
        n=len(self.demandMat)
        for x in range(n):
            for y in range(n):
                totalDemand+=self.demandMat[x][y]
                if self.demandMat[x][y] == 0 or x==y:
                    continue
                else:
                    for z in range(l):
                        routeLength=len(routeSet[z])
                        startInd3x=[i for i in range(routeLength) if routeSet[z][i]==x]
                        endIndex=[i for i in range(routeLength) if routeSet[z][i]==y]
                        
                        if len(startInd3x)==0 or len(endIndex)==0:
                            continue
                        elif startInd3x[0]<endIndex[0]:
                            demandFulfilled+=self.demandMat[x][y]
                            
        totalFit+=(demandFulfilled/totalDemand)*37              
        if store==1:
            fit[0]=(demandFulfilled/totalDemand)*100
        #calculating time fitness
        totalTime=0
        for x in range(n):
            for y in range(n):
                if self.demandMat[x][y]==0 or x==y:
                    continue
                minTime=100000
                for z in range(l):
                    currMinTime=0
                    routeLength=len(routeSet[z])
                    if routeLength==0:
                        continue
                    startInd3x=[i for i in range(routeLength) if routeSet[z][i]==x]
                    endIndex=[i for i in range(routeLength) if routeSet[z][i]==y]
                    if len(startInd3x)==0 or len(endIndex)==0:
                            continue
                    else:
                        for i in range(startInd3x[0],endIndex[0]):
                            currMinTime+=self.linkData[i][i+1]
                        if currMinTime<minTime and currMinTime!=0:
                            minTime=currMinTime
                    
                if minTime != 100000:
                    totalTime+=(10-abs(self.shortestTimeMat[x][0][y]-minTime)/self.shortestTimeMat[x][0][y])
        totalFit+=totalTime    
        if store==1:
            fit[1]=totalTime
                
        if store==0:
            return totalFit
        else:
            return fit
        
    def modify(self):
        l=len(self.currPopulation)
        #inter route crossover
        for x in range(l):
            cointoss=random.randint(0,1)
            if cointoss:
                secondStringIndex=random.randint(0,l-1)
                if(len(self.currPopulation[x])-1<=0 or len(self.currPopulation[secondStringIndex])-1<=0):
                    continue
                firstStringSite=random.randint(0,len(self.currPopulation[x])-1)
                secondStringSite=random.randint(0,len(self.currPopulation[secondStringIndex])-1)
                self.currPopulation[x][firstStringSite],self.currPopulation[secondStringIndex][secondStringSite]=self.currPopulation[secondStringIndex][secondStringSite],self.currPopulation[x][firstStringSite]
            
        #intra route crossover
        copyCount=self.copyCount
        for x in range(l):
            for i in range(copyCount-1):
                self.currPopulation+=[self.currPopulation[i]]
        
        for x in range(l*copyCount):
            if(len(self.currPopulation[x])-1<=0):
                continue
            first_string_indx=random.randint(0,len(self.currPopulation[x])-1)
            secondStringIndex=random.randint(0,len(self.currPopulation[x])-1)
            minLength=min(len(self.currPopulation[x][first_string_indx]),len(self.currPopulation[x][secondStringIndex]))
            minLength=random.randint(1,math.floor(minLength*4/5))
            tempString=[self.currPopulation[x][first_string_indx][i] for i in range(len(self.currPopulation[x][first_string_indx]))]
            for y in range(minLength):
                self.currPopulation[x][first_string_indx][y]=self.currPopulation[x][secondStringIndex][y]
                self.currPopulation[x][secondStringIndex][y]=tempString[y]

        #reproduction
        fitnessRank=[[self.evaluation(self.currPopulation[x]),x] for x in range(l*copyCount)]
        tot_fit=0
        for x in range(l*copyCount):
            tot_fit+=fitnessRank[x][0]
        probability=[0 for x in range(l*copyCount)]
        for x in range(l*copyCount):
            probability[x]=fitnessRank[x][0]/tot_fit
        
        tempPopulation=[[] for x in range(l)]
    
        for x in range(l):
            currChoice=np.random.choice(l*copyCount,1,p=probability)
            tot_fit-=fitnessRank[int(currChoice)][0]
            probability[int(currChoice)]=0
            for y in range(l*copyCount):
                    if probability[y]!=0:
                        probability[y]=fitnessRank[y][0]/tot_fit
            tempPopulation[x]=self.currPopulation[int(currChoice)]
        
        #Mutation 
        for x in range(l):
            y=random.random()
            if y<self.mutationProb:
                i=random.randint(0,len(tempPopulation[x])-1)
                indx1=random.randint(0,len(tempPopulation[x][i])-1)
                indx2=random.randint(0,len(tempPopulation[x][i])-1)
                tempPopulation[x][i][indx1],tempPopulation[x][i][indx2]=tempPopulation[x][i][indx2],tempPopulation[x][i][indx1]
        self.currPopulation=tempPopulation
        
    def optimize(self):
        self.initialization()
        
        maxFitX,avgFitX=[],[]
        for g in range(self.genCount):
            self.modify()
            fitnessRank=[self.evaluation(self.currPopulation[i]) for i in range(len(self.currPopulation))]
            maxFit=-1
            indx=-1
            avgFitness=0
            for x in range(len(self.currPopulation)):
                if maxFit<fitnessRank[x]:
                    maxFit,indx=fitnessRank[x],x
                avgFitness+=fitnessRank[x]
            maxFitX.append(maxFit)
            avgFitX.append(avgFitness/len(self.currPopulation))
            
        return self.currPopulation,maxFitX,avgFitX
    
def model_run(data):
    fleetSize,busStopData,demandMatr,linkMat=ReadData.dataTransfer()
    GA=RouteOptimizer()
    GA.maxRoute=data[3]
    GA.genCount=data[0]
    GA.startPopulation=data[1]
    GA.fleetSize=fleetSize
    GA.busStops=busStopData
    GA.demandMat=demandMatr
    GA.linkData=linkMat
    GA.mutationProb=data[2]/100
    GA.copyCount=data[4]
    GA.initialization()
    finalPop,maxFitX,avgFitX=GA.optimize()
    fitnessRank=[GA.evaluation(finalPop[i]) for i in range(len(finalPop))]
    indx=-1
    maxFit=-1
    for x in range(len(finalPop)):
        if maxFit<fitnessRank[x]:
            maxFit,indx=fitnessRank[x],x
    
    with open('generatedRoutes.csv','w',newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(['Route_No','Route_Detail'])
        for x in range(len(finalPop[indx])):
            routeString=''
            for y in range(len(finalPop[indx][x])-1):
                routeString+=str(busStopData[finalPop[indx][x][y]][1])+', '
            routeString+=str(busStopData[finalPop[indx][x][len(finalPop[indx][x])-1]][1])
            writer.writerow([x+1,routeString])

    lastFit= GA.evaluation(finalPop[indx],1)
   
    return finalPop[indx],maxFitX,avgFitX,lastFit[0],lastFit[1]
