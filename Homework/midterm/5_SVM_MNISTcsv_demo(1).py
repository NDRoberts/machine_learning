
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data=np.array(pd.read_csv('./MNIST.csv'))
print("[INFO] evaluating classifier...")
trainX=data[0:1000,1:]
trainY=data[0:1000,0]
testX=data[1000:1100,1:]
testY=data[1000:1100,0]
 
 
Gamma=0.001
C=1#0.001
model =  svm.SVC(kernel='poly', C=C,gamma=Gamma)
#model = LogisticRegression()
#model =  DecisionTreeClassifier()
model.fit(trainX, trainY)
predY=model.predict(testX)
print(classification_report(testY, predY))      
#
Showlist=np.arange(10)
for i in Showlist:
    sample=testX[i]
    sample=sample.reshape((28,28))
    plt.imshow(sample,cmap='gray')
    plt.title('The prediction:'+str(predY[i]))
    plt.show()