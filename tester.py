# Tester file for trying out solutions
import ast
import networkx as nx
import pandas
import pandas as pd

#Data paths
train_set = 'datasets/train.csv'

def centralities(edgelist):
    """
     - edgelist is a list of node pairs e.g. [(7,2),(1,7),(1,9),...]
     - returns a dictionary of vertex -> (centrality values)
    """
    T = nx.from_edgelist(edgelist)
    dc = nx.degree_centrality(T)
    cc = nx.harmonic_centrality(T)
    bc = nx.betweenness_centrality(T)
    pc = nx.pagerank(T)
    return {v: (dc[v], cc[v], bc[v], pc[v]) for v in T}


def centrality_calc(x):
    #Evaluates the string given as a python list for loading into the graph
    x = ast.literal_eval(x)
    cent = centralities(x)
    cent = dict(sorted(cent.items()))
    return cent

def reestructure_dataset(data):
    rows = []
    headers = ['language', 'sentence', 'sent_n',
               'v', 'deg_cent', 'harm_cent',
               'btwn_cent', 'pgrnk_cent', 'is_root']
    for index, row in data.iterrows():
        language = row['language']
        sentence = row['sentence']
        sent_n = row['n']
        cents = centrality_calc(row['edgelist'])
        root = row['root']

        for v in cents:
            deg_cent = cents[v][0]
            harm_cent = cents[v][0]
            btwn_cent = cents[v][0]
            pgrnk_cent = cents[v][0]
            is_root = 1 if v == root else 0
            row = [language, sentence, sent_n,
                    v, deg_cent, harm_cent,
                    btwn_cent,pgrnk_cent, is_root]
            rows.append(row)
    df = pandas.DataFrame(data=rows, columns=headers)
    print(df.head(50))

def main():
    data = pd.read_csv(train_set, index_col=False)
    data = data.head(100)
    reestructure_dataset(data)

if __name__ == '__main__':
    main()