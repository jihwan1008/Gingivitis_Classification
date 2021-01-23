import os
import random as rand
import numpy as np
import shutil

path_train = os.getcwd() + '/data/' + 'train'
path_test = os.getcwd() + '/data/' + 'test'

if os.path.exists(path_train + '/True'):
    pass
else:
    os.makedirs(path_train + '/True')

if os.path.exists(path_train + '/False'):
    pass
else:
    os.makedirs(path_train + '/False')

if os.path.exists(path_test + '/True'):
    pass
else:
    os.makedirs(path_test + '/True')

if os.path.exists(path_test + '/False'):
    pass
else:
    os.makedirs(path_test + '/False')


# Split the train and test data with ratio of 7:3
for i, item in enumerate(os.listdir(os.getcwd() + '/data/Aggregated_True')):
    if i < 103:
        shutil.copy(os.getcwd() + '/data/Aggregated_True/' + item, os.getcwd() + '/data/train/True/' + item)
    else:
        shutil.copy(os.getcwd() + '/data/Aggregated_True/' + item, os.getcwd() + '/data/test/True/' + item)

for i, item in enumerate(os.listdir(os.getcwd() + '/data/Aggregated_False')):
    if i < 103:
        shutil.copy(os.getcwd() + '/data/Aggregated_False/' + item, os.getcwd() + '/data/train/False/' + item)
    else:
        shutil.copy(os.getcwd() + '/data/Aggregated_False/' + item, os.getcwd() + '/data/test/False/' + item)
