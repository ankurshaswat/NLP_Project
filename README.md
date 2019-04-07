# NLP_Project

NLP Project - Spring 2019

Run 'python grapher.py' first to generated 'symbol2ids' file first for NELL dataset 

CUDA_VISIBLE_DEVICES=0 python application_backend.py --max_neighbor 50 --fine_tune

CUDA_VISIBLE_DEVICES=0 python trainer.py --max_neighbor 50 --fine_tune --test

