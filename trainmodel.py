import pandas as pd
import sys
import os

train=pd.read_csv('trainingdataset.csv')
print(train.head())
print(train['image_name'].nunique())
print(train['type'].value_counts())

data=pd.DataFrame()
data['format']=train['image_name']



for i in range(data.shape[0]):
    data['format'][i]=data['format'][i]+','+str(train['xmin'][i])+','\
                      +str(train['ymin'][i])+','+str(train['xmax'][i])+','+str(train['ymax'][i])+','+train['type'][i]
data.to_csv('annotate.txt',header=None,index=None,sep=' ')

#os.system('cd keras-frcnn')
os.system('python ./keras-frcnn/train_frcnn.py -o simple -p annotate.txt')
