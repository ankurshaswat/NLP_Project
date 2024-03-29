from flask import Flask, request, jsonify, abort
import json
import sys
from tqdm import tqdm
from collections import defaultdict
from trainer import Trainer
from args import read_options

app = Flask(__name__)


# api = Api(app)

# api.add_resource(HelloWorld, '/')


'''
    Send complete entity name to this
'''
@app.route('/api/neighbours', methods=['POST'])
def get_neighbours():

    if not request.json or not 'ent' in request.json:
        abort(400)

    ent = request.json['ent']
    neighbours = graph[symbols2ids[ent]]

    decoder = {}

    for neighbour in neighbours:
        decoder[neighbour[0]] = id2symbols[str(neighbour[0])]
        decoder[neighbour[1]] = id2symbols[str(neighbour[1])]

    result = {'decoder': decoder, 'neighbours': neighbours}

    return jsonify(result), 201


'''
    Send learning relation and query
'''
@app.route('/api/predict', methods=['POST'])
def get_predictions():

    print(request.json)

    if not request.json or not 'query' in request.json:
        abort(400)

    # support = request.json['support']
    query = request.json['query']

    # query_object = {'support':support,'query':query}

    result_predictions = trainer.run_(query)

    decoder = {}

    for rel in result_predictions.keys():
        for query in result_predictions[rel]:
            for ranked_result in query:
                for idx in ranked_result:
                    if idx not in decoder:
                        decoder[idx] = id2symbols[str(idx)]

    return jsonify({'results': result_predictions, 'decoder': decoder}),201

if __name__ == '__main__':
    args = read_options()
    # print(args)

    data_path = 'data/' + args.dataset + '/'

    id2symbols = json.load(open(data_path + '/id2symbols.json'))
    symbols2ids = json.load(open(data_path + '/symbol2ids.json'))

    graph = defaultdict(list)

    with open(data_path+'path_graph') as f:
        lines = f.readlines()
        print("Generating graph")
        for line in tqdm(lines):
            e1, rel, e2 = line.rstrip().split()

            e1_id = symbols2ids[e1]
            rel_id = symbols2ids[rel]
            rel_inv_id = symbols2ids[rel+'_inv']
            e2_id = symbols2ids[e2]

            graph[e1_id].append((rel_id, e2_id))
            graph[e2_id].append((rel_inv_id, e1_id))

    trainer = Trainer(args)

    app.run(debug=True)
