import numpy as np
import torch
import json
import argparse
import logging
import random

from collections import defaultdict
from torch.autograd import Variable
from tqdm import tqdm

from matcher import EmbedMatcher
from grapher import Graph

class Application(object):

    def __init__(self, arg):
        super(Application, self).__init__()
        for k, v in vars(arg).items():
            setattr(self, k, v)

        self.meta = not self.no_meta

        logging.info('LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        self.load_symbol2id_ent2id_id2symbol()
        self.use_pretrain = False

        self.num_symbols = len(self.symbol2id.keys()) - 1  # one for 'PAD'
        self.pad_id = self.num_symbols
        self.matcher = EmbedMatcher(self.embed_dim, self.num_symbols, use_pretrain=self.use_pretrain, embed=self.symbol2vec, dropout=self.dropout,
                                    batch_size=self.batch_size, process_steps=self.process_steps, finetune=self.fine_tune, aggregate=self.aggregate)
        self.matcher.cuda()

        self.batch_nums = 0
        self.writer = None

        self.parameters = filter(
            lambda p: p.requires_grad, self.matcher.parameters())

        self.num_ents = len(self.ent2id.keys())


        logging.info('BUILDING CONNECTION MATRIX')
        self.build_connection(max_=self.max_neighbor)

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(
            open(self.dataset + '/rel2candidates.json'))

        # load answer dict
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

        # Create Graph object (for querying paths later on)
        logging.info('BUILDING GRAPH OBJECT FOR {} DATASET'.format(arg.dataset))
        self.graph=Graph(arg.dataset)
        


    def load_symbol2id_ent2id_id2symbol(self):
        symbol_id = {}
        id_symbol = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))
        i = 0
        for key in rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                id_symbol[i] = key
                i += 1

        self.relation_sym_range = (0, i-1)

        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                id_symbol[i] = key
                i += 1

        self.ent_sym_range = (self.relation_sym_range[1]+1, i-1)

        symbol_id['PAD'] = i
        id_symbol[i] = 'PAD'

        self.symbol2id = symbol_id
        self.id2symbol = id_symbol
        self.ent2id = ent2id
        self.symbol2vec = None

    def build_connection(self, max_=100):

        self.connections = (np.ones((self.num_ents, max_, 2))
                            * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        with open(self.dataset + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2 = line.rstrip().split()
                self.e1_rele2[e1].append(
                    (self.symbol2id[rel], self.symbol2id[e2]))
                self.e1_rele2[e2].append(
                    (self.symbol2id[rel+'_inv'], self.symbol2id[e1]))

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)  # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]

        return degrees

    def load(self):
        self.matcher.load_state_dict(torch.load(self.save_path))

    def get_meta(self, left, right):
        left_connections = Variable(torch.LongTensor(
            np.stack([self.connections[_, :, :] for _ in left], axis=0))).cuda()
        left_degrees = Variable(torch.FloatTensor(
            [self.e1_degrees[_] for _ in left])).cuda()
        right_connections = Variable(torch.LongTensor(
            np.stack([self.connections[_, :, :] for _ in right], axis=0))).cuda()
        right_degrees = Variable(torch.FloatTensor(
            [self.e1_degrees[_] for _ in right])).cuda()
        return (left_connections, left_degrees, right_connections, right_degrees)

    def run(self, mode='new_rel', meta=False):
        self.matcher.eval()

        symbol2id = self.symbol2id
        few = self.few

        logging.info('EVALUATING ON QUERY DATA')
        tasks = json.load(open(self.query_file))

        rel2candidates = self.rel2candidates

        for query_ in tasks.keys():

            if (mode == 'old_rel'):
                candidates = rel2candidates[query_]
            elif(mode == 'new_rel'):
                candidates = []
                # for index in range(self.ent_sym_range[0],self.ent_sym_range[1]):
                #     candidates.append(self.id2symbol[index])

                candidates += rel2candidates[query_]
                candidates=candidates[:4000]
                print("\n\nQUERY: {}".format(query_))
                print("\n\n CANDIDATES: ",candidates)

                while(len(candidates) < 500):
                    sample = random.randint(
                        self.ent_sym_range[0], self.ent_sym_range[1])
                    if(self.id2symbol[sample] not in candidates):
                        candidates.append(self.id2symbol[sample])

            support_triples = tasks[query_][:few]
            support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]]
                             for triple in support_triples]

            if meta:
                support_left = [self.ent2id[triple[0]]
                                for triple in support_triples]
                support_right = [self.ent2id[triple[2]]
                                 for triple in support_triples]
                support_meta = self.get_meta(support_left, support_right)

            support = Variable(torch.LongTensor(support_pairs)).cuda()

            for triple in tasks[query_][few:]:

                print("\nExisting Connecntions of query head")
                neighbors_of_top = self.e1_rele2[triple[0]]
                for rel, e2 in neighbors_of_top:
                    print(triple[0], self.id2symbol[rel], self.id2symbol[e2])

                query_pairs = []

                if meta:
                    query_left = []
                    query_right = []

                for ent in candidates:
                    query_pairs.append(
                        [symbol2id[triple[0]], symbol2id[ent]])
                    if meta:
                        query_left.append(self.ent2id[triple[0]])
                        query_right.append(self.ent2id[ent])

                query = Variable(torch.LongTensor(query_pairs)).cuda()

                if meta:
                    query_meta = self.get_meta(query_left, query_right)
                    scores = self.matcher(
                        query, support, query_meta, support_meta)
                    scores.detach()
                    scores = scores.data
                else:
                    scores = self.matcher(query, support)
                    scores.detach()
                    scores = scores.data

                scores = scores.cpu().numpy()
                sort = list(np.argsort(scores))[::-1]



                rel = self.id2symbol[query_pairs[sort[0]][0]]
                top_e = self.id2symbol[query_pairs[sort[0]][1]]

                print("\nTop 10 Results")
                for target_rank in range(10):
                    index = sort[target_rank]
                    query_pair = query_pairs[index]
                    print('Rank', target_rank+1, ': Head=', self.id2symbol[query_pair[0]][8:], 'Relation=', query_[
                          8:], 'Tail=', self.id2symbol[query_pair[1]][8:])

                    # if(target_rank==2):
                        # top_e=query_pair[1]

                print("\nExisting Connections of top result")
                neighbors_of_top = self.e1_rele2[top_e]
                for rel, e2 in neighbors_of_top:
                    print(top_e, self.id2symbol[rel], self.id2symbol[e2])

                path_k=600
                path_depth=2
                print("\nFinding paths for k={} and depth={}".format(path_k,path_depth))
                e2=triple[0]
                e1=triple[2]
                paths=self.graph.pair_feature([e1,e2],k=path_k,depth=path_depth)
                # e1=top_e
                # e2=self.id2symbol[neighbors_of_top[0][1]]
                # paths=self.graph.pair_feature([e2,e1],k=path_k,depth=path_depth)                
                print("\nFound {} paths between {} & {}".format(len(paths),e1,e2))
                for path in paths:
                    print("*********")
                    print(path)



    def run_(self, mode):
        self.load()
        logging.info('Pre-trained model loaded')
        self.run(mode, meta=self.meta)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NELL", type=str)
    parser.add_argument("--embed_dim", default=100, type=int)
    parser.add_argument("--few", default=1, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--process_steps", default=2, type=int)
    parser.add_argument("--fine_tune", action='store_true')
    parser.add_argument("--aggregate", default='max', type=str)
    parser.add_argument("--max_neighbor", default=200, type=int)
    parser.add_argument("--no_meta", action='store_true')
    parser.add_argument("--prefix", default='intial', type=str)

    parser.add_argument("--seed", default='19940419', type=int)

    # parser.add_argument("--log_every", default=50, type=int)
    # parser.add_argument("--eval_every", default=10000, type=int)
    # parser.add_argument("--neg_num", default=1, type=int)
    # parser.add_argument("--random_embed", action='store_true')
    # parser.add_argument("--train_few", default=1, type=int)
    # parser.add_argument("--lr", default=0.001, type=float)
    # parser.add_argument("--margin", default=5.0, type=float)
    # parser.add_argument("--max_batches", default=1000000, type=int)
    # parser.add_argument("--test", action='store_true')
    # parser.add_argument("--grad_clip", default=5.0, type=float)
    # parser.add_argument("--weight_decay", default=0.0, type=float)
    # parser.add_argument("--embed_model", default='ComplEx', type=str)

    parser.add_argument("--query_file", default='queries/query.json', type=str)
    parser.add_argument("--app_mode", default=1, type=int, choices=[1, 2])

    args = parser.parse_args()
    args.save_path = 'models/' + args.prefix

    print("------HYPERPARAMETERS-------")
    for k, v in vars(args).items():
        print(k + ': ' + str(v))
    print("----------------------------")

    return args


if __name__ == '__main__':
    args = read_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    # setup random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    app = Application(args)
    app.run_(mode='new_rel')
