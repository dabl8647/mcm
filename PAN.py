import numpy as np
import pickle

def save_obj(obj, name ):
    with open('obj/'+name+'.pkl','wb') as f:
        pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)

        
def load_obj(name):
    with open('obj/'+name+'.pkl','rb')as f:
        return pickle.load(f)
    
def simData(n,Age = 0, infoConst = 0, infoAlpha = False):
    domainInfo = []
    indChar = []
    infoSec = []
    for i in range(n):
        if(Age == 0):
            u = np.random.uniform(0,1)
            if (u <= .068):
                Age = np.random.uniform(0,5)
            elif (u <= (.068+.189)):
                Age = np.random.uniform(5,17)
            elif (u <= (.257+.096)):
                Age = np.random.uniform(17,24)
            elif (u <= (.257+.097+.302)):
                Age = np.random.uniform(24,44)
            elif (u <= (.399+.257+.22)):
                Age = np.random.uniform(44,64)
            else:
                Age = np.random.uniform(64,100)
        gender = np.random.uniform(0,1)
        #Political individual Characteristic
        politics  = np.random.uniform(0,1)
        tempChar = [Age/100,gender,politics]
        indChar.append(tempChar)
        #Pos 0 = Financial, 1 = social, 2 = health
        domainData = []
        domainData.append(np.random.beta(10*Age/100, 4.6))
        domainData.append(np.random.beta(4.6,10*Age/100))
        domainData.append(np.random.beta(10*Age/100, 3.4))
        domainInfo.append(domainData)
        #Information Security
        if(infoAlpha == False):
            infoSec.append(infoConst)
        else:
            infoSec.append(np.random.beta((10-infoAlpha),3.7)+infoConst)
    return[indChar,domainInfo,infoSec]

def ageSim(network,age,n):
    ageData = {}
    for a in age:
        print(a)
        ageData.update({a: {'diff': [], 'tot': [],'char': []}})
        for i in range(n):
            data = simData(1,a)
            indChar = data[0]
            domain = data[1]
            infoSec = data[2]
            network.addNode('test',indChar,domain,infoSec)
            impactCoeff = network.calcImpact('test')
            benVec = [0, 2*132.47, 2*62500]
            damVec = [2*9812, 2*4920, 2*29336]
            benefit = calcBenefit(network,'test',benVec)
            damage = calcDamage(network,impactCoeff,damVec,'test')
            diff = benefit-damage
            tot = np.sum(diff)
            ageData[a]['diff'].append(diff)
            ageData[a]['tot'].append(tot)
            ageData[a]['char'].append(data)
    return ageData

        
                      
def checkSimilarity(node1, node2):
    #This is a function to calculate the similarity between two nodes
    #Importantly, all features MUST be normalized to values between 0 and 1
    indChar1 = np.array(node1.indChar)
    domain1 = np.array(node1.domain)
    
    indChar2 = np.array(node2.indChar)
    domain2 = np.array(node2.domain)
    diffInd = np.abs(indChar1-indChar2)
    diffDom = np.abs(domain1-domain2)
    diff = diffInd+diffDom #Get the difference
    tot = np.sum(np.abs(diff))#Take the absolute value and sum to get total distance
    return tot


def calcDamage(network,impactCoeffs, params, nodeInfo):
    #Value of info is the total number of connections person has
    moneyVec = params
    ToTimpact = np.dot(impactCoeffs,moneyVec)
    domainImpact = np.multiply(impactCoeffs,moneyVec)
    return domainImpact
def calcBenefit(network, nodeInfo, P ):
    node = network.nodeList[nodeInfo]
    firstVec = np.multiply(node.domain, P)
    secondVec = np.zeros(len(P),)
    for i in network.adjList[nodeInfo]:
        ind = network.adjList[i].index(nodeInfo)
        params = network.nodeList[i].domain
        weightVec = network.weightList[i][ind]
        tempVec = np.multiply(params,P)
        secondVec = secondVec+np.multiply(tempVec,weightVec)
        
    benefit = firstVec+secondVec
    return benefit

class Node:
    def __init__(self,name, indChar, domain, infSec):
        #Node Structure
        self.name = name
        self.indChar = indChar
        self.domain = domain
        self.infoSec = infSec
        
class Network:
    def __init__(self,a,MD,c, adjList = {}, nodeList = {}, weightList = {}):
        #Init a network with size n and inherent connection rate a
        self.adjList = adjList #Adj list for representing graph
        self.nodeList = nodeList #list of added Nodes
        self.weightList = weightList #Dictionary of connection weights labeled by edge that it corresponds to
        self.a = a
        self.c = c #Average number of connections we expect to see
        self.maxDist = MD #Number of params
        self.numParams = MD
        
    def addNode(self,name,indChar,domain,infoSec,c = 0):
        #Add the node to the adjacency list
        if(c == 0):
            c = self.c
        self.adjList.update({name: []})
        #Add the node to the node list
        self.nodeList.update({name: Node(name,indChar,domain,infoSec)})
        self.weightList.update({name: []})
        for n,a in self.adjList.items():
            if(n != name): #Don't want node to connect to itself
                currNode = Node(name,indChar,domain,infoSec); #Get the node we are adding
                checkNode = self.nodeList[n] #Get the node we are seeing if it connects to
                tot = checkSimilarity(currNode,checkNode) #Calculate the similarity
                Q = (1-tot/self.maxDist)/(1-self.a)#Convert sim to a prob
                self.addEdge(Q,currNode,checkNode,c) #Check and see if the edge gets added

                
    def addEdge(self,Q,node1,node2,c):
        size = len(self.nodeList)
        prob = (Q+self.a)*(c/size) #Add inherent prob to sim prob
        u = np.random.uniform(0,1) 
        name1 = node1.name
        name2 = node2.name
        if(u <= prob):
            #If the node is to be added, add to both parts of the adjacency list
            self.adjList[name2].append(name1)
            self.adjList[name1].append(name2)
            #Calculate the absolute value of domain vector differences
            diffVec = np.array(node1.domain)-np.array(node2.domain)
            diffVec = np.absolute(diffVec)
            diffVec = 1-diffVec
            self.weightList[name2].append(diffVec)#Add the connection strength for each parameter
            self.weightList[name1].append(diffVec)
    
    def calcImpact(self,name1):
		degreeSum = np.sum(self.weightList[name1])
        node = self.nodeList[name1]
        #Multiply the sum(connectionWeights) and the node's individual parameter values element-wise
        impactVec = (1 - node.infoSec[0]) * node.domain * degreeSum
        return impactVec

if __name__=="__main__":
	pass
