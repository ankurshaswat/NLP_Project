#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#

from flask import Flask, render_template, request
# from flask.ext.sqlalchemy import SQLAlchemy
import logging
from logging import Formatter, FileHandler
from forms import *
import os
import json
from extra_files.runner import Trainer

from flask import jsonify

#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object('config')

### Load your model over here ###
args = {
"dataset":"NELL",
"embed_dim":100,
"few":1,
"batch_size":128,
"neg_num":1,
"train_few":1,
"lr":0.001,
"margin":5.0,
"max_batches":1000000,
"dropout":0.2,
"process_steps":2,
"log_every":50,
"eval_every":10000,
"aggregate":'max',
"max_neighbor":50,
"grad_clip":5.0,
"weight_decay":0.0,
"embed_model":'ComplEx',
"prefix":'NELL_attn_bestHits10',
"seed":19940419,
"query_file":'queries/query.json',
"max_extra_neighbor_depth":0,
"app_mode":'query_new_rel',
"attend_neighbours":1,
"no_meta":False,
"random_embed":False,
"fine_tune":False,
"sort_neighbours":False
}

entity_list=[]
with open('extra_files/NELLent_list.json') as handle:
    entity_list=json.loads(handle.read())

rel2candidates = json.load(open('extra_files/rel2candidates.json'))


with open('extra_files/NELLoptions.json') as handle:
    options = json.loads(handle.read())
model = Trainer(args)

def predict(input):

    for ent in [input["e1"] , input["e2"], input["query"]]:
        if ent not in entity_list:
            return [ent+" not in Knowledge Base!!"]

    support = [
            'concept:'+input["e1"],
            'concept:'+input["rel"],
            'concept:'+input["e2"]
            # "concept:automobilemodel:windstar",
            # "concept:producedby",
            # "concept:company:ford001"
        ]


    head = 'concept:'+input["query"]
    # head = "concept:product:wii_console"

    candidate_type = support[2].split(":")[1]

    raw_candidates = options['ents'][candidate_type][:]

    candidates = []

    # if("conceppt:"+input["rel"] in rel2candidates):
    #     print("found in rel2candidates")
    #     candidates = rel2candidates["concept:"+input["rel"]]
    # else: 
    for cand in raw_candidates:
        candidates.append("concept:"+candidate_type+":"+cand)

    output = model.rank(support, head,candidates)
    
    for i in range(len(output)):
        output[i]=output[i].replace('concept:','')
    
    # print(output)
    return output

#----------------------------------------------------------------------------#
# Controllers.
#----------------------------------------------------------------------------#
@app.route('/', methods=['GET','POST'])
def home():
    global entity_list
    if request.method == 'POST':
        ## Called after submit button is clicked
        output = predict(request.form)
        template = render_template('project.html', results=output, entity_list=entity_list,
            e1=request.form["e1"],e2=request.form["e2"], rel=request.form["rel"], query=request.form["query"] )
        # print(template)
        return template

    if request.method == 'GET':
        # print(entity_list[:2])
        return render_template('project.html',entity_list=entity_list , 
            e1="automobilemodel:windstar", e2="company:ford001", rel="producedby", query="product:outlook_email")


if not app.debug:
    file_handler = FileHandler('error.log')
    file_handler.setFormatter(
        Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info('errors')

#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

# Default port:
if __name__ == '__main__':
    app.run()

# Or specify port manually:
'''
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
'''

