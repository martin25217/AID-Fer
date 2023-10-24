import networkx as nx;
import numpy as np;
from pybdm import BDM;
import random;
import matplotlib.pyplot as plt;
import time;

start = time.time()

#myGraph = nx.read_adjlist("/home/demonlord/project_eta/AID/com-amazon.ungraph.txt.gz", nodetype=int)
#myGraph = nx.convert_node_labels_to_integers(myGraph)

myGraph = nx.complete_graph(64)

bdm = BDM(ndim=1, nsymbols=2)

print(myGraph.number_of_nodes())

sumOfEntvals = 0
sumOfBDMvals = 0
numOfNodes = myGraph.number_of_nodes()



for node1 in myGraph.nodes():
    
    array = np.empty(shape=numOfNodes, dtype=int)
    for node2 in myGraph:
        if(node2 in myGraph.neighbors(node1)):
            array[node2] = 1
        else:
            array[node2] = 0

    sumOfEntvals += bdm.nent(array)
    sumOfBDMvals += bdm.nbdm(array)

entVal = sumOfEntvals/numOfNodes
bdmVal = sumOfBDMvals/numOfNodes

print("BDM is: " + str(bdmVal))
print("Entropy is: " + str(entVal))

end = time.time()

print(end-start)