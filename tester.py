# Tester file for trying out solutions
import ast
import networkx as nx
import pandas as pd

#Data paths
train_set = 'datasets/train.csv'

data = pd.read_csv(train_set, index_col=False)

def centrality_calc(x):
    #Evaluates the string given as a python list for loading into the graph
    x = ast.literal_eval(x)
    g = nx.Graph(x)
    cent = nx.degree_centrality(g)
    cent = max(cent, key=cent.get)
    print(f"For Sentence {x} \nthe central node is: {cent}")


data['edgelist'].apply(centrality_calc)