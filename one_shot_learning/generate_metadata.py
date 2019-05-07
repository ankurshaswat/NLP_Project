import sys
import json

dataset_path = sys.argv[1]

rel2id = json.load(open(dataset_path + '/relation2ids'))
ent2id = json.load(open(dataset_path + '/ent2ids'))

rels = []

symbol2ids = {}
id2symbols = {}

i = 0


for rel in rel2id:
    if rel not in ['', 'OOV']:
        symbol2ids[rel] = i
        id2symbols[i] = rel
        i += 1

    rel_parts = rel.split(':')

    if(len(rel_parts) < 2):
        continue

    if rel_parts[1] not in rels:
        rels.append(rel_parts[1])

ents = {}

ent_list=[]


for ent in ent2id:
    if ent not in ['', 'OOV']:
        symbol2ids[ent] = i
        id2symbols[i] = ent
        i += 1

    ent_parts = ent.split(':')

    if(len(ent_parts) < 3):
        continue

    if ent_parts[1] not in ents:
        ents[ent_parts[1]] = []

    ents[ent_parts[1]].append(ent_parts[2])

    ent_list.append(ent_parts[1]+":"+ent_parts[2])

symbol2ids['PAD'] = i
id2symbols[i] = 'PAD'

options = {'ents': ents, 'rels': rels}

with open(dataset_path+'ent_list.json', 'w') as outfile:
    json.dump(sorted(ent_list), outfile)

with open(dataset_path+'options.json', 'w') as outfile:
    json.dump(options, outfile)

with open(dataset_path+'id2symbols.json', 'w') as outfile:
    json.dump(id2symbols, outfile)

with open(dataset_path+'symbol2ids.json', 'w') as outfile:
    json.dump(symbol2ids, outfile)
