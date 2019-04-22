# NLP_Project

NLP Project - Spring 2019

Run 'python grapher.py' first to generated 'symbol2ids' file first for NELL dataset

For Training
CUDA_VISIBLE_DEVICES=0 python trainer.py --max_neighbor 50 --fine_tune

For Testing
CUDA_VISIBLE_DEVICES=0 python trainer.py --max_neighbor 50 --fine_tune --app_mode test

For Training with extra neighbours (till depth 2)
CUDA_VISIBLE_DEVICES=0 python trainer.py --max_neighbor 50 --fine_tune --add_extra_neighbours --max_extra_neighbor_depth 1 --prefix NELL

For Testing queries with extra neighbours (till depth 2)
CUDA_VISIBLE_DEVICES=0 python trainer.py --max_neighbor 51 --fine_tune --prefix NELL --query_file queries/query.json --add_extra_neighbours --max_extra_neighbor_depth 1 --app_mode query_new_rel


For flask app

python flask_app.py --prefix NELL --app_mode query_new_rel

APIs setup for predict and neighbours