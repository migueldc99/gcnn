import os
from glob import glob
from sklearn.model_selection import train_test_split

""" split the database into train (60%), validate(20%) and test (20%) sets """

os.mkdir('./train')
os.mkdir('./validate')
os.mkdir('./test')

files = glob('*.xyz')

print('Total number of entries: '+repr(len(files)))

reminder_set, test = train_test_split( files, test_size = 0.2, random_state = 42 )

size = len( test )

train, validate = train_test_split( reminder_set, test_size = 0.2, random_state = 42 )
# train, validate = train_test_split( reminder_set, test_size = size, random_state = 42 )

print('test_size = '+repr(size))
print('validate_size = '+repr(len(validate)))
print('train_size = '+repr(len(train)))

total = size + len(validate) + len( train )

print('total_size = '+repr(total))

# move test files to test
    
for file in test:
    command = 'mv -f '+file+' ./test'
    os.system(command)

# move validate files to validate

for file in validate:
    command = 'mv -f '+file+' ./validate'
    os.system(command)

# move train files to train

for file in train:
    command = 'mv -f '+file+' ./train'
    os.system(command)
