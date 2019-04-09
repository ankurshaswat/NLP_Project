import pickle
from grapher import Graph

import matplotlib.pyplot as plt
import sys

"""
arg 1 is dataset ('NELL' or 'Wiki')
arg 2 is results filename (eg-'NELL_test_results')

Expected format of results file-
a dict storing results for each set of support triples,
results for each triple stored in the form of dict,
mapping head entity to (ground_truth, rank, top_5 predictions)
"""

def best_predictions(results, cutoff=2):
    print("BEST PREDICTIONS-")
    for support_triples in results:
        triple_stats=results[support_triples]
        for head in triple_stats:
            tail,rank,top5=triple_stats[head]
            if(rank<=2):
                print("**********")
                print("SUPPORT TRIPLES: {}".format(support_triples))
                print("HEAD: {}, TAIL: {}, RANK: {} ".format(head,tail,rank))

def degree_vs_ranks(graph,results):

    for support_triples in results:
        triple_stats=results[support_triples]

        head_degree_ranks=[]
        tail_degree_ranks=[]


        for head in triple_stats:
            tail,rank,top5=triple_stats[head]
            if(rank>5):
                 continue
            head_deg=len(graph.connections[head])+len(graph._connections[head])
            tail_deg=len(graph.connections[tail])+len(graph._connections[tail])
            head_degree_ranks.append((head_deg,rank))
            tail_degree_ranks.append((tail_deg,rank))


        rank_count={}
        for h,r in head_degree_ranks:
            j=h-(h%5) #bins of 5
            if j not in rank_count:
                rank_count[j]=1
            else:
                rank_count[j]+=1

        if(len(rank_count.keys())<5):
             continue


        x,y=[],[]
        for i in sorted(rank_count.keys()):
            # print(i)
            # print(ranks[i])
            x.append(i)
            y.append(rank_count[i])
        # plt.scatter(x,y)
        plt.plot(x,y)
        plt.show()




if __name__=='__main__':

    dataset=sys.argv[1]
    results_file=sys.argv[2]
    with open(results_file, "rb") as input_file:
        results = pickle.load(input_file)
    graph=Graph(dataset)
    best_predictions(results)
    # degree_vs_mrr(graph,results)    