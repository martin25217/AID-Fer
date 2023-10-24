import networkx as nx;
import numpy as np;
from pybdm import BDM;
import random;
import matplotlib.pyplot as plt;

def permuteGraph(adjecencyMatrix : np.ndarray, iteration : list) -> np.ndarray:

    for i in range(len(iteration)):
        adjecencyMatrix[i, :], adjecencyMatrix[iteration[i], :] = adjecencyMatrix[iteration[i], :], adjecencyMatrix[i,:].copy()
        adjecencyMatrix[:,i], adjecencyMatrix[:,iteration[i]] = adjecencyMatrix[:, iteration[i]], adjecencyMatrix[:,i].copy()
    
    return adjecencyMatrix


vertexNumber = 64
probability = 0.1
numOfRepetitions = 100

#myGraph = nx.newman_watts_strogatz_graph(vertexNumber,5, probability,)
#myGraph = nx.erdos_renyi_graph(vertexNumber,probability)
#myGraph = nx.read_adjlist("/home/demonlord/project_eta/AID/com-amazon.ungraph.txt.gz", nodetype=int)
myGraph = nx.complete_graph(64)

adjMatrix = nx.to_numpy_array(myGraph, dtype=int)
permutingList = list()
arrayInt = adjMatrix.astype(int)

nx.draw(myGraph, with_labels=True)
plt.savefig("graph.png")

bdm = BDM(ndim=2, nsymbols=2)

entVals = np.empty(numOfRepetitions)
bdmVals = np.empty(numOfRepetitions)

print(arrayInt.shape)

val1 = bdm.nbdm(np.array(arrayInt))
val2 = bdm.nent(np.array(arrayInt)) 

print("Algoritmic complexity is " + str(val1))
print("Shanon entropy is " + str(val2))
print("The difference is " + str(val2-val1))

print("\n")

for i in range(0,vertexNumber):
    permutingList.append(i)


counter = 0

for i in range(numOfRepetitions):

    random.shuffle(permutingList)
    iteratedGraph = permuteGraph(arrayInt,permutingList)
    rmemberOld = list(permutingList)


    bdmVals[counter] = bdm.nbdm(iteratedGraph)
    entVals[counter] = bdm.nent(iteratedGraph)

    counter = counter + 1

entMean = np.mean(entVals)
bdmMean = np.mean(bdmVals)

entMedian = np.median(entVals)
bdmMedian = np.median(bdmVals)

entStd = np.std(entVals)
bdmStd = np.std(bdmVals)

print("Mean of bdm : " + str(bdmMean))
print("Median of bdm : " + str(bdmMedian))
print("Standard deviation of bdm : " + str(bdmStd))

print("\n")

print("Mean of statistical entropy : " + str(entMean))
print("Median of entropy : " + str(entMedian))
print("Standard deviation of entropy : " + str(entStd))

plt.clf()
plt.hist(entVals)
plt.savefig("entHist.png")

plt.clf()
plt.hist(bdmVals)
plt.savefig("bdmHist.png")
