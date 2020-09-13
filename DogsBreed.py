import os
import pandas as pd

base_dir = './'

train_dir = os.path.join(base_dir, 'train')
new_train_dir = os.path.join(base_dir, 'trainDirectory')
test_dir = os.path.join(base_dir, 'test')

labels = pd.read_csv('./labels.csv', header = 0)

uniqueValues = labels.breed.unique()

try:
	os.mkdir(new_train_dir)
except OSError:
	print('Error in creating directory')

for i in uniqueValues:
	try:
		os.mkdir(new_train_dir+'/'+i)
	except OSError:
		print('Error in creating directory '+i)


for i in range(0, len(labels['id'])):
	sourceFolder = train_dir +'/'+ labels.loc[i,'id']+'.jpg' 
	destinationFolder = new_train_dir+'/'+ labels.loc[i,'breed'] + '/' + labels.loc[i,'id']+'.jpg'
	
	os.replace(sourceFolder, destinationFolder)


