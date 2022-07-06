# -*- coding: utf-8 -*-
"""
@author: Miguel Dalmau Casañal
"""

import os
import shutil
import tarfile   # treatment of files .tar
import glob
from sklearn.model_selection import train_test_split

""" download the database in our directory"""

my_tar = tarfile.open('dsgdb9nsd.xyz.tar')
my_tar.extractall('C:/Users/Portatil/Documents/Master/TFM/Trabajo/MoleculeDB/Ficheros_xyz')
my_tar.close()

""" split the database into train (60%), validate(20%) and test (20%) sets """

os.mkdir('./train')
os.mkdir('./validate')
os.mkdir('./test')

files = glob.glob('C:/Users/Portatil/Documents/Master/TFM/Trabajo/MoleculeDB/Ficheros_xyz/dsgdb9nsd_*.xyz')

print('Total number of entries: '+repr(len(files)))

reminder_set, test = train_test_split( files, test_size = 0.2, random_state = 42 )

size = len( test )
# print(test)

train, validate = train_test_split( reminder_set, test_size = size, random_state = 42 )

print('test_size = '+repr(size))
print('validate_size = '+repr(len(validate)))
print('train_size = '+repr(len(train)))

total = size + len(validate) + len( train )

print('total_size = '+repr(total))

# move test files to test
    
for file in test:
    shutil.move(file, "C:/Users/Portatil/Documents/Master/TFM/Trabajo/MoleculeDB/test")

# move validate files to validate

for file in validate:
    shutil.move(file, "C:/Users/Portatil/Documents/Master/TFM/Trabajo/MoleculeDB/validate")
    
# move train files to train

for file in train:
    shutil.move(file, "C:/Users/Portatil/Documents/Master/TFM/Trabajo/MoleculeDB/train")