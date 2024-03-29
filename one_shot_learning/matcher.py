import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from modules import *
from torch.autograd import Variable


class EmbedMatcher(nn.Module):
    """
    Matching metric based on KB Embeddings
    """

    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout=0.2, batch_size=64, process_steps=4, finetune=False, aggregate='max', attend_neighbours=0):
        super(EmbedMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(
            num_symbols + 1, embed_dim, padding_idx=num_symbols)
        self.aggregate = aggregate
        self.num_symbols = num_symbols

        self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

        self.dropout = nn.Dropout(0.5)

        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)

        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            # emb_np = np.loadtxt(embed_path)
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        d_model = self.embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

        # for attention
        self.attend_neighbours = attend_neighbours
        if(attend_neighbours==1):
            self.query_nbr_attention = ScaledDotProductAttention(d_model)
        else:
            self.query_nbr_attention = ScaledDotProductAttention2(d_model)
        

        # save attention results (updated every forward pass)
        self.attention_results=[0]*2 #for left (head) and right(tail)
        


    def neighbor_encoder(self, connections, num_neighbors):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:, :, 0].squeeze(-1)
        entities = connections[:, :, 1].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(
            relations))  # (batch, 200, embed_dim)
        ent_embeds = self.dropout(self.symbol_emb(
            entities))  # (batch, 200, embed_dim)

        # (batch, 200, 2*embed_dim)
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)

        out = self.gcn_w(concat_embeds)

        out = torch.sum(out, dim=1)  # (batch, embed_dim)
        out = out / num_neighbors
        return out.tanh()


    def neighbor_encoder_attn(self, connections, num_neighbors, support, id2ent=None, head_or_tail=0 ):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:, :, 0].squeeze(-1)
        entities = connections[:, :, 1].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(
            relations))  # (batch, 200, embed_dim)
        ent_embeds = self.dropout(self.symbol_emb(
            entities))  # (batch, 200, embed_dim)

        # (batch, 200, 2*embed_dim)
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)

        out = concat_embeds

        batch_size, num_neighbors = list(out.size())[:2]
        
        # print("Query(support) shape before: {}".format(support.shape))
        
        support = support.repeat(batch_size, num_neighbors, 1)

        # print("Batch size:{}, Number of neighbours: {}".format(batch_size,num_neighbors))
        # print("Query(support) shape: {}, Key/Value(neighbours) shape: {}".format(support.shape,out.shape))

        # print("Input value:",out[-1,1,:])

        out, attn_wts = self.query_nbr_attention(support, out, out)

        # print("Attention module output shape: {}".format(out.shape))
        
        # print("Output value:",out[-1,1,:])


        if(id2ent is not None):
            attn_wts=attn_wts.cpu().detach().numpy()
            # print("\nAttn_wts shape:", attn_wts.shape)
            print("\nAttn_layer output shape:", out.shape)
            

            sorted_indices= [np.argsort(attn_wts[i,0,:]) for i in range(batch_size) ]

            results=[0]*batch_size

            for ind,l in enumerate(sorted_indices):
                ent_list= [id2ent[ entities[ind][i].item() ] for i in l]
                rel_list= [id2ent[ relations[ind][i].item() ] for i in l]
                
                ent_list=list(reversed(ent_list)) [:] #most attended to least attended
                rel_list=list(reversed(rel_list)) [:]

                results[ind]=(ent_list,rel_list)

            self.attention_results[head_or_tail]=results



            # best=[id2ent[ entities[sel_candidate][i].item() ] for i in best]
            # worst=[id2ent[ entities[sel_candidate][i].item() ] for i in worst]
            # print("\nHigh attention neighbours: ",best)
            # print("\nLow attention neighbours: ",worst)
            



        # out = self.gcn_w(concat_embeds)
        out = self.gcn_w(out)

        # print("GCN layer output shape: {}".format(out.shape))

        out = torch.sum(out, dim=1)  # (batch, embed_dim)

        out = out / num_neighbors
        return out.tanh()







    def forward(self, query, support, query_meta=None, support_meta=None, neighbour_encode_only=False, id2ent=None):
        '''
        query: (batch_size, 2)
        support: (few, 2)
        return: (batch_size, )
        '''

        if neighbour_encode_only:
            # print('In Neigbour encoder only')
            return self.forward_neighbour_encoder(query, support, query_meta, support_meta)
        # print('Not In Neigbour encoder only')

        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        support_left = self.neighbor_encoder(
            support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(
            support_right_connections, support_right_degrees)

        support_neighbor = torch.cat(
            (support_left, support_right), dim=-1)  # tanh

        support = support_neighbor

        if(self.attend_neighbours==0):
            query_left = self.neighbor_encoder(
                query_left_connections, query_left_degrees)
            query_right = self.neighbor_encoder(
                query_right_connections, query_right_degrees)
            query_neighbor = torch.cat(
                (query_left, query_right), dim=-1)  # tanh
        else:
            query_left = self.neighbor_encoder_attn(
                query_left_connections, query_left_degrees, support, id2ent=id2ent,head_or_tail=0)
            query_right = self.neighbor_encoder_attn(
                query_right_connections, query_right_degrees, support, id2ent=id2ent,head_or_tail=1)
            query_neighbor = torch.cat(
                (query_left, query_right), dim=-1)  # tanh



        query = query_neighbor

        support_g = self.support_encoder(support)  # 1 * 100
        query_g = self.support_encoder(query)

        support_g = torch.mean(support_g, dim=0, keepdim=True)
        # support_g = support
        # query_g = query

        query_f = self.query_encoder(support_g, query_g)  # 128 * 100

        # cosine similarity
        matching_scores = torch.matmul(query_f, support_g.t()).squeeze()

        return matching_scores

    def forward_neighbour_encoder(self, query, support, query_meta=None, support_meta=None):
        '''
        Function to do forward pass only till neighbour encoder
        query: (batch_size, 2)
        support: (few, 2)
        return: (batch_size, )
        '''
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_left = self.neighbor_encoder(
            query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(
            query_right_connections, query_right_degrees)

        support_left = self.neighbor_encoder(
            support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(
            support_right_connections, support_right_degrees)

        query_neighbor = torch.cat((query_left, query_right), dim=-1)  # tanh
        support_neighbor = torch.cat(
            (support_left, support_right), dim=-1)  # tanh

        support = support_neighbor
        query = query_neighbor

        simple_matching_scores = torch.matmul(query, support.t()).squeeze()

        return simple_matching_scores

    def forward_(self, query_meta, support_meta):
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_left = self.neighbor_encoder(
            query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(
            query_right_connections, query_right_degrees)
        support_left = self.neighbor_encoder(
            support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(
            support_right_connections, support_right_degrees)

        query = torch.cat((query_left, query_right), dim=-1)  # tanh
        support = torch.cat((support_left, support_right), dim=-1)  # tanh

        support_expand = support.expand_as(query)

        distances = F.sigmoid(self.siamese(
            torch.abs(support_expand - query))).squeeze()
        return distances


class RescalMatcher(nn.Module):
    """
    Matching based on KB Embeddings
    """

    def __init__(self, embed_dim, num_ents, num_rels, use_pretrain=True, ent_embed=None, rel_matrices=None, dropout=0.1, attn_layers=1, n_head=4, batch_size=64, process_steps=4, finetune=False, aggregate='max'):
        super(RescalMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.ent_emb = nn.Embedding(
            num_ents + 1, embed_dim, padding_idx=num_ents)
        self.rel_matrices = nn.Embedding(
            num_rels + 1, embed_dim * embed_dim, padding_idx=num_rels)

        self.aggregate = aggregate

        self.gcn_w = nn.Linear(self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

        # self.gcn_self_r = nn.Parameter(torch.FloatTensor(1, self.embed_dim))

        self.dropout = nn.Dropout(0.5)

        init.xavier_normal(self.gcn_w.weight)
        init.constant(self.gcn_b, 0)

        if use_pretrain:
            print('LOADING KB EMBEDDINGS')
            # emb_np = np.loadtxt(embed_path)
            self.ent_emb.weight.data.copy_(torch.from_numpy(ent_embed))
            self.rel_matrices.weight.data.copy_(torch.from_numpy(rel_matrices))
            if not finetune:
                print('FIX KB EMBEDDING')
                self.ent_emb.weight.requires_grad = False
                self.rel_matrices.weight.requires_grad = False

        d_model = embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout)
        self.query_encoder = QueryEncoder(d_model, process_steps)

        # self.slf_neighbor = ContextAwareEncoder(attn_layers, d_model, d_inner_hid, n_head, d_k, d_v, dropout)

        # extra parameters for Siamese Neural Networks
        # self.siamese = nn.Linear(d_model, 1)
        # init.xavier_normal(self.siamese.weight)

    def neighbor_encoder(self, connections, num_neighbors):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:, :, 0].squeeze(-1)
        entities = connections[:, :, 1].squeeze(-1)
        # (batch, 200, embed_dim*embed_dim)
        rel_embeds = self.dropout(self.rel_matrices(relations))
        ent_embeds = self.dropout(self.ent_emb(
            entities))  # (batch, 200, embed_dim)

        batch_size = rel_embeds.size()[0]
        max_neighbors = rel_embeds.size()[1]
        rel_embeds = rel_embeds.view(-1, self.embed_dim, self.embed_dim)
        ent_embeds = ent_embeds.view(-1, self.embed_dim).unsqueeze(2)

        # concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1) # (batch, 200, 2*embed_dim)
        concat_embeds = torch.bmm(rel_embeds, ent_embeds).squeeze().view(
            batch_size, max_neighbors, -1)
        # print concat_embeds.size()

        # concat_embeds = self.slf_neighbor(concat_embeds)

        out = self.gcn_w(concat_embeds) + self.gcn_b  # (batch, 200, embed_dim)
        out = torch.sum(out, dim=1)  # (batch, embed_dim)
        out = out / num_neighbors
        # out = F.relu(out)
        out = F.tanh(out)
        # out = F.sigmoid(out)
        return out

    def forward(self, query, support, query_meta=None, support_meta=None):
        '''
        query: (batch_size, 2)
        support: (few, 2)
        return: (batch_size, )
        '''
        if query_meta == None:
            support = self.dropout(self.symbol_emb(
                support)).view(-1, 2*self.embed_dim)
            query = self.dropout(self.symbol_emb(
                query)).view(-1, 2*self.embed_dim)
            support = support.unsqueeze(0)
            support_g = self.support_encoder(support).squeeze(0)
            query_f = self.query_encoder(support_g, query)
            matching_scores = torch.matmul(query_f, support_g.t())
            if self.aggregate == 'max':
                query_scores = torch.max(matching_scores, dim=1)[0]
            elif self.aggregate == 'mean':
                query_scores = torch.mean(matching_scores, dim=1)
            elif self.aggregate == 'sum':
                query_scores = torch.sum(matching_scores, dim=1)
            return query_scores

        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_left = self.neighbor_encoder(
            query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(
            query_right_connections, query_right_degrees)

        support_left = self.neighbor_encoder(
            support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(
            support_right_connections, support_right_degrees)

        query_neighbor = torch.cat((query_left, query_right), dim=-1)  # tanh
        support_neighbor = torch.cat(
            (support_left, support_right), dim=-1)  # tanh

        support = support_neighbor
        query = query_neighbor

        support_g = self.support_encoder(support)  # 1 * 100
        query_g = self.support_encoder(query)

        query_f = self.query_encoder(support_g, query_g)  # 128 * 100

        # cosine similarity
        matching_scores = torch.matmul(query_f, support_g.t()).squeeze()

        # if self.aggregate == 'max':
        #     matching_scores = torch.max(matching_scores, dim=1)[0]
        # elif self.aggregate == 'mean':
        #     matching_scores = torch.mean(matching_scores, dim=1)
        # elif self.aggregate == 'sum':
        #     matching_scores = torch.sum(matching_scores, dim=1)

        # query_scores = F.sigmoid(query_scores)
        return matching_scores

    def forward_(self, query_meta, support_meta):
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_left = self.neighbor_encoder(
            query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(
            query_right_connections, query_right_degrees)
        support_left = self.neighbor_encoder(
            support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(
            support_right_connections, support_right_degrees)

        query = torch.cat((query_left, query_right), dim=-1)  # tanh
        support = torch.cat((support_left, support_right), dim=-1)  # tanh

        support_expand = support.expand_as(query)

        distances = F.sigmoid(self.siamese(
            torch.abs(support_expand - query))).squeeze()
        return distances


if __name__ == '__main__':

    query = Variable(torch.ones(64, 2).long()) * 100
    support = Variable(torch.ones(40, 2).long())
    matcher = EmbedMatcher(40, 200, use_pretrain=False)
    print(matcher(query, support).size())
