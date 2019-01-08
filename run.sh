#!/usr/bin/env bash
python mnist_eval_co_training.py --sup  &
python mnist_eval_co_training.py --sup --js &
python mnist_eval_co_training.py --sup --js --adv
