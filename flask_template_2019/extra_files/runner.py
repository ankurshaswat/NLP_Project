import json
# import logging
import numpy as np
# import pickle
# import torch
# import torch.nn.functional as F
# import datetime

from collections import defaultdict
# from collections import deque
from torch import optim
# from torch.autograd import Variable
from tqdm import tqdm

# from args import read_options
# from data_loader import *
from matcher import EmbedMatcher
# from tensorboardX import SummaryWriter

# import datetime
# from grapher import Graph


class Trainer(object):

    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in arg.items():
        # for k, v in vars(arg).items():
            setattr(self, k, v)

        if self.app_mode == 'train':

            with open('models/'+self.prefix+'_params.json', 'w') as outfile:
                json.dump(vars(arg), outfile)

        self.test = (self.app_mode == 'test')
        self.add_extra_neighbours = (self.max_extra_neighbor_depth > 0)

        self.meta = not self.no_meta

        if self.random_embed:
            use_pretrain = False
        else:
            use_pretrain = True

        #logging.info('LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        if self.test or self.random_embed:
            self.load_symbol2id()
            use_pretrain = False
        else:
            self.load_embed()
        self.use_pretrain = use_pretrain

        self.num_symbols = len(self.symbol2id.keys()) - 1  # one for 'PAD'
        self.pad_id = self.num_symbols
        self.matcher = EmbedMatcher(self.embed_dim, self.num_symbols, use_pretrain=self.use_pretrain, embed=self.symbol2vec, dropout=self.dropout,
                                    batch_size=self.batch_size, process_steps=self.process_steps, finetune=self.fine_tune,
                                    aggregate=self.aggregate, attend_neighbours=self.attend_neighbours)
        self.matcher#.cuda()

        self.batch_nums = 0

        self.writer = None

        self.parameters = filter(
            lambda p: p.requires_grad, self.matcher.parameters())
        self.optim = optim.Adam(
            self.parameters, lr=self.lr, weight_decay=self.weight_decay)

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=[200000], gamma=0.5)

        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.num_ents = len(self.ent2id.keys())

        #logging.info('BUILDING CONNECTION MATRIX')
        degrees = self.build_connection(max_=self.max_neighbor)

        #logging.info('LOADING CANDIDATES ENTITIES')
        # self.rel2candidates = json.load(
        #     open(self.dataset + '/rel2candidates.json'))

        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

        #logging.info(
            # 'BUILDING GRAPH OBJECT FOR {} DATASET'.format(arg.dataset))
        # self.graph = Graph(arg.dataset)

    def load_symbol2id(self):
        id_symbol = {}

        symbol_id = {}
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
        self.symbol2vec = None
        self.id2symbol = id_symbol

    def load_embed(self):

        symbol_id = {}
        id_symbol = {}

        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))

        #logging.info('LOADING PRE-TRAINED EMBEDDING')
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
            ent_embed = np.loadtxt(
                self.dataset + '/entity2vec.' + self.embed_model)
            rel_embed = np.loadtxt(
                self.dataset + '/relation2vec.' + self.embed_model)

            if self.embed_model == 'ComplEx':
                # normalize the complex embeddings
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

            assert ent_embed.shape[0] == len(ent2id.keys())
            assert rel_embed.shape[0] == len(rel2id.keys())

            i = 0
            embeddings = []
            for key in rel2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    id_symbol[i] = key

                    i += 1
                    embeddings.append(list(rel_embed[rel2id[key], :]))
            self.relation_sym_range = (0, i-1)

            for key in ent2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    id_symbol[i] = key
                    i += 1
                    embeddings.append(list(ent_embed[ent2id[key], :]))
            self.ent_sym_range = (self.relation_sym_range[1]+1, i-1)

            symbol_id['PAD'] = i
            id_symbol[i] = 'PAD'

            embeddings.append(list(np.zeros((rel_embed.shape[1],))))
            embeddings = np.array(embeddings)
            assert embeddings.shape[0] == len(symbol_id.keys())

            self.symbol2id = symbol_id
            self.id2symbol = id_symbol

            self.symbol2vec = embeddings

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

        def temp_func(x):
            return len(self.e1_rele2[self.id2symbol[x[1]]])

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                if self.sort_neighbours:
                    # print('1',list(map(temp_func, neighbors)))
                    neighbors_sorted = sorted(
                        neighbors, key=temp_func, reverse=True)
                    new_neighbours = []
                    entities_added = []
                    for neighbour in neighbors_sorted:
                        if neighbour[1] in entities_added:
                            continue
                        else:
                            new_neighbours.append(neighbour)
                            entities_added.append(neighbour[1])

                    if(len(new_neighbours) < max_):
                        # while(len(new_neigbors)<max_):
                        for neighbour in neighbors_sorted:
                            if neighbour not in new_neighbours:
                                new_neighbours.append(neighbour)
                                if(len(new_neighbours) == max_):
                                    break

                    neighbors = new_neighbours[:max_]
                else:
                    neighbors = neighbors[:max_]

                # print(list(map(temp_func, neighbors)))
            # degrees.append(len(neighbors))
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)  # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]

        # Creating extended connections in pre processing
        if self.add_extra_neighbours:
            #logging.info('BUILDING EXTENDED NEIGHBOUR CONNECTION MATRIX')

            self.connections_extended = (np.ones((self.num_ents, max_, 2))
                                         * self.pad_id).astype(int)
            self.e1_degrees_extended = defaultdict(int)

            for ent, id_ in tqdm(self.ent2id.items()):

                for i in range(self.e1_degrees[id_]):
                    self.connections_extended[id_, i,
                                              0] = self.connections[id_, i, 0]
                    self.connections_extended[id_, i,
                                              1] = self.connections[id_, i, 1]

                self.e1_degrees_extended[id_] = self.e1_degrees[id_]

                pos_2_add = self.e1_degrees[id_]
                degree_threshold = self.e1_degrees[id_]

                if pos_2_add >= max_:
                    continue

                depth = 0
                for i in range(max_):

                    neigbour_ent = self.connections_extended[id_, i, 1]

                    neigbour_ent = self.ent2id[self.id2symbol[neigbour_ent]]

                    new_connections = self.connections[neigbour_ent, :, :]

                    for j in range(self.e1_degrees[neigbour_ent]):
                        self.connections_extended[id_,
                                                  pos_2_add, :] = new_connections[j, :]

                        pos_2_add += 1
                        self.e1_degrees_extended[id_] += 1

                        if pos_2_add >= max_:
                            break

                    if i == degree_threshold-1:
                        depth += 1
                        degree_threshold = self.e1_degrees_extended[id_]

                    if (depth >= self.max_extra_neighbor_depth) or pos_2_add >= max_:
                        break

        return degrees

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.matcher.state_dict(), path)

    def load(self):
        self.matcher.load_state_dict(torch.load(self.save_path))

    def get_meta(self, left, right):
        if self.add_extra_neighbours:

            left_connections = Variable(torch.LongTensor(
                np.stack([self.connections_extended[_, :, :] for _ in left], axis=0)))#.cuda()
            left_degrees = Variable(torch.FloatTensor(
                [self.e1_degrees_extended[_] for _ in left]))#.cuda()
            right_connections = Variable(torch.LongTensor(
                np.stack([self.connections_extended[_, :, :] for _ in right], axis=0)))#.cuda()
            right_degrees = Variable(torch.FloatTensor(
                [self.e1_degrees_extended[_] for _ in right]))#.cuda()

        else:

            left_connections = Variable(torch.LongTensor(
                np.stack([self.connections[_, :, :] for _ in left], axis=0)))#.cuda()
            left_degrees = Variable(torch.FloatTensor(
                [self.e1_degrees[_] for _ in left]))#.cuda()
            right_connections = Variable(torch.LongTensor(
                np.stack([self.connections[_, :, :] for _ in right], axis=0)))#.cuda()
            right_degrees = Variable(torch.FloatTensor(
                [self.e1_degrees[_] for _ in right]))#.cuda()

        return (left_connections, left_degrees, right_connections, right_degrees)

    def rank(self,support,candidates):
        self.matcher.eval()

        symbol2id = self.symbol2id
        few = self.few

        results = {}

        tasks = {}

        rel2candidates = self.rel2candidates

        support_triples = [support]
        support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]]
                            for triple in support_triples]

        if meta:
            support_left = [self.ent2id[triple[0]]
                            for triple in support_triples]
            support_right = [self.ent2id[triple[2]]
                                for triple in support_triples]
            support_meta = self.get_meta(support_left, support_right)

        support = Variable(torch.LongTensor(support_pairs))#.cuda()

        results[query_] = []

        neighbors_of_top = self.e1_rele2[triple[0]]

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

        query = Variable(torch.LongTensor(query_pairs))#.cuda()

        param = None

        if meta:
            query_meta = self.get_meta(query_left, query_right)
            scores = self.matcher(
                query, support, query_meta, support_meta, id2ent=param)
            scores.detach()
            scores = scores.data
        else:
            scores = self.matcher(query, support, id2ent=param)
            scores.detach()
            scores = scores.data

        scores = scores.numpy()
        sort = list(np.argsort(scores))[::-1]

        all_in_rank_order = []
        for i in sort:
            all_in_rank_order.append(self.id2symbol[query_pairs[i]])

        return all_in_rank_order[:10]