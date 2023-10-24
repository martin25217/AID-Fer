import networkx as nx;
import matplotlib.pyplot as plt;
import itertools;
import numpy as np;
from pybdm import BDM;



myGraph = nx.Graph()
myGraph.add_edge(0,1)
myGraph.add_edge(0,2)
myGraph.add_edge(1,2)

nx.draw(myGraph, with_labels=True)
plt.savefig("graphs/newGraph.png")

myList = [0,1,2]
counter = 0

bdm = BDM(ndim = 1, nsymbols = 2)

for i in itertools.combinations(myList,2):
    plt.clf()
    myGraph.remove_edge(i[0], i[1])
    nx.draw(myGraph, with_labels=True)
    plt.savefig("graphs/newGraph" + str(counter) + ".png")
    counter = counter + 1
    adjecencyMatrix = nx.to_numpy_array(myGraph)  
    print(adjecencyMatrix)
    print(type(adjecencyMatrix))
    #newAdjecencyMatrix = adjecencyMatrix.astype(int)
    print(bdm.bdm(adjecencyMatrix))
    myGraph.add_edge(i[0], i[1])









