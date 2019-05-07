import sys

import pickle
import matplotlib.pyplot as plt
import ast

from grapher import Graph
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
        triple_stats = results[support_triples]
        for head in triple_stats:
            tail, rank, top5 = triple_stats[head]
            if(rank <= cutoff):
                print("**********")
                print("SUPPORT TRIPLES: {}".format(support_triples))
                print("HEAD: {}, TAIL: {}, RANK: {} ".format(head, tail, rank))


def degree_vs_ranks(graph, results, rank_cutoff=5, degree_threshold=50):

    relation_wise_stats = {}
    for support_triples in results:
        triple_stats = results[support_triples]

        print(support_triples)

        sup_trip = ast.literal_eval(support_triples)
        # print(sup_trip)
        rel = sup_trip[0][1]
        # print("RELATION: {}".format(rel))

        if(rel not in relation_wise_stats):
            relation_wise_stats[rel] = {}
            relation_wise_stats[rel]['low_degree_entities'] = ([], [])
            relation_wise_stats[rel]['total_entities'] = 0
            relation_wise_stats[rel]['deg_rank'] = []

        head_degree_ranks = []
        tail_degree_ranks = []

        scarce_heads = []
        scarce_tails = []

        for head in triple_stats:
            tail, rank, top5, _, _ = triple_stats[head]

            head_deg = len(graph.connections[head]) + \
                len(graph._connections[head])
            tail_deg = len(graph.connections[tail]) + \
                len(graph._connections[tail])
            neighbour_degs = 0
            for x in [head, tail]:
                for i in graph.connections[x]:
                    neighbour_degs += len(graph.connections[i]) + \
                        len(graph._connections[i])
                for i in graph._connections[x]:
                    neighbour_degs += len(graph.connections[i]) + \
                        len(graph._connections[i])
            head_degree_ranks.append((head_deg+tail_deg+neighbour_degs, rank))
            tail_degree_ranks.append((tail_deg, rank))

            # if(rank <= rank_cutoff):
            #     if(head_deg < degree_threshold):
            scarce_heads.append(head)
            # if(tail_deg < degree_threshold):
            scarce_tails.append(tail)

            relation_wise_stats[rel]['total_entities'] += 1

        relation_wise_stats[rel]['deg_rank'] += head_degree_ranks

        rank_count = {}
        for h, r in head_degree_ranks:
            j = h-(h % 5)  # bins of 5
            if j not in rank_count:
                rank_count[j] = 1
            else:
                rank_count[j] += 1

        if(len(rank_count.keys()) < 5):
            continue

        h_c, t_c = relation_wise_stats[rel]['low_degree_entities']
        h_c += scarce_heads
        t_c += scarce_tails

        # print("\n")
        print(len(h_c), len(t_c))

        # x,y=[],[]
        # for i in sorted(rank_count.keys()):
        #     # print(i)
        #     # print(ranks[i])
        #     x.append(i)
        #     y.append(rank_count[i])
        # plt.scatter(x,y)
        # plt.plot(x,y)
        # plt.show()

    for rel in relation_wise_stats:
        print("\nStats for relation {}".format(rel))
        h_c, t_c = relation_wise_stats[rel]['low_degree_entities']
        tot = relation_wise_stats[rel]['total_entities']
        print("Total hits@{}: {}".format(rank_cutoff, tot))

        print("h_c: {}, t_c: {}".format(len(h_c)/(tot+1), len(t_c)/(tot+1)))

        x, y = [], []
        rank_deg_stats = relation_wise_stats[rel]['deg_rank']
        for deg, r in rank_deg_stats:
            # print(i)
            # print(ranks[i])
            # print(r,deg)

            # if(r>50):
                # continue
            x.append(r)
            # y.append(deg)
            # y.append(min(deg, 2*degree_threshold))
            y.append(deg)

        plt.xlabel('Rank')
        plt.ylabel('Size of neighbourhood')
        plt.title('Relation: {}'.format(rel))
        plt.scatter(x, y)
        plt.show()
        plt.clf()


# def translate(results):
#     results_copy={}
#     for support_triples in results.keys():
#         support_triples_copy=ast.literal_eval(support_triples)
#         for i in range(len(support_triples)):
#             e=get_string_desc(support_triples[i])
#             support_triples[i]=e
#         results_copy[str(support_triples_copy)]={}

#         triple_stats=results[support_triples]
#         triple_stats_copy={}
#         for head in triple_stats:
#             tail,rank,top5=triple_stats[head]
#             h_c=get_string_desc(head)
#             t_c=get_string_desc(tail)
#             top5_c=[get_string_desc(i) for i in top5]
#             triple_stats_copy[h_c]=(t_c,rank,top5_c)

#         return results_copy

# if __name__ == '__main__':

#     dataset = sys.argv[1]
#     results_file = sys.argv[2]
#     with open(results_file, "rb") as input_file:
#         results = pickle.load(input_file)

#     # print(results)
#     # if(dataset=='Wiki'):
#     #     results=translate(results)

#     graph = Graph(dataset)
#     # best_predictions(results,cutoff=5)
#     degree_vs_ranks(graph, results)


def compare(dataset, results1, results2):
    for support_triples in results1:

        triple_stats1 = results1[support_triples]
        triple_stats2 = results2[support_triples]

        for head in triple_stats1:
            tail1, rank1, top5_1, degree_head, degree_true = triple_stats1[head]
            tail2, rank2, top5_2, degree_head, degree_true = triple_stats2[head]

            if(abs(rank1-rank2) < 10):
                continue

            if(rank1 < rank2):
                print("\nModel 1 Did Better")
                rank = rank1
                tail = tail1
            elif(rank1 > rank2):
                print("\nModel 2 Did Better")
                rank = rank2
                tail = tail2
            else:
                print("Equally Good")
            print("DIFFERENCE: ", abs(rank1-rank2))
            print("SUPPORT TRIPLES: {}".format(support_triples))
            print("HEAD: {}, TAIL: {}, RANK: {}, DEGREE_HEAD: {}, DEGREE_TRUE: {}".format(
                head, tail, rank, degree_head, degree_true))


if __name__ == '__main__':

    dataset = sys.argv[1]
    results_file1 = sys.argv[2]
    results_file2 = sys.argv[3]

    with open(results_file1, "rb") as input_file:
        results1 = pickle.load(input_file)

    with open(results_file2, 'rb') as input_file:
        results2 = pickle.load(input_file)

    compare(dataset, results1, results2)
