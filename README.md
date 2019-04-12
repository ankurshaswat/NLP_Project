# NLP_Project

NLP Project - Spring 2019

Run 'python grapher.py' first to generated 'symbol2ids' file first for NELL dataset 

CUDA_VISIBLE_DEVICES=0 python application_backend.py --max_neighbor 50 --fine_tune

CUDA_VISIBLE_DEVICES=0 python trainer.py --max_neighbor 50 --fine_tune --test

For training with extra neighbours

CUDA_VISIBLE_DEVICES=0 python trainer.py --max_neighbor 50 --fine_tune --add_extra_neighbours --max_extra_neighbor_depth 1 --prefix NELL

for testing with extra neighbours

CUDA_VISIBLE_DEVICES=0 python application_backend.py --max_neighbor 51 --fine_tune --prefix NELL --query_file queries/query.json --add_extra_neighbours --max_extra_neighbor_depth 1