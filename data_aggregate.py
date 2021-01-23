import shutil
import os
import sys

name_list = ['affine', 'color', 'color2', 'color3', 'gaussian', 'gray', 'same']
name_list_2 = ['True', 'False']
root = os.getcwd() + '/' + 'Gingivitis_'

if os.path.exists(os.getcwd() + '/data/Aggregated_True'):
    pass
else:
    os.makedirs(os.getcwd() + '/data/Aggregated_True')

if os.path.exists(os.getcwd() + '/data/Aggregated_False'):
    pass
else:
    os.makedirs(os.getcwd() + '/data/Aggregated_False')

def index(num):

    result = str()

    if int(num) < 10:
        result = '00' + str(num)
    elif int(num) > 10 and int(num) < 100:
        result = '0' + str(num)
    else:
        result = str(num)

    return result


# for i, item in enumerate(os.listdir(os.getcwd() + '/data/Gingivitis_True')):
#     shutil.move(os.getcwd() + '/data/Gingivitis_True/' + item, os.getcwd() + '/data/Gingivitis_True/' + 'image_' +
#                 str(index(i)) + '.jpg')
#
# for i, item in enumerate(os.listdir(os.getcwd() + '/data/Gingivitis_False')):
#     shutil.move(os.getcwd() + '/data/Gingivitis_False/' + item, os.getcwd() + '/data/Gingivitis_False/' + 'image_' +
#                 str(index(i)) + '.jpg')

for i, item in enumerate(name_list):
    for elem in name_list_2:
        for j, image in enumerate(os.listdir(root + elem + '_' + item)):
            shutil.copy(root + elem + '_' + item + '/' + image, os.getcwd() + '/data/Aggregated_' + elem + '/' +
                        'image_' + str(index(i * 21 + j)) + '.jpg')