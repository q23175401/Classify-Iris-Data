from sklearn import datasets as DS
from sklearn.model_selection import train_test_split as TTS
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import pydot
from sklearn.externals.six import StringIO

iris = DS.load_iris()
iris_data = iris.data
# iris_data = preprocessing.scale(iris_data)
iris_label = iris.target

train_data , test_data , train_label , test_label = TTS(iris_data,iris_label,test_size=0.3,random_state=3)

#KNN Accuracy
KNN_Accuracy=[]
for k in range(1,101):
    knn = KNC(n_neighbors=k)
    knn.fit(train_data,train_label)
    result=knn.predict(test_data)
    KNN_Accuracy.append(knn.score(test_data,test_label))
    
plt.plot(KNN_Accuracy)
plt.title('Accuracy influenced by different neighbers in KNN')
plt.xlabel('neighbers(1-100)')
plt.ylabel('accuracy(%)')
plt.show()

NB=GaussianNB()
NB.fit(train_data,train_label)

fig=plt.figure(figsize=(12,15))
fig.tight_layout()
plt.subplots_adjust(wspace =0, hspace =0.4)

for fea in range(0,4):
    ax=plt.subplot(4,1,fea+1)
    if(fea==0):
        ax.set_title('classes influenced by different features',fontsize=20)
    ax.set_xlabel(iris.feature_names[fea])
    ax.set_ylabel('distribution')
    for clss in range(0,3):
        mu=NB.theta_[clss][fea] 
        sigma=NB.sigma_[clss][fea]
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 200)
#         y=PC.minmax_scale(norm.pdf(x, mu, sigma),feature_range=(0,1))
        y=norm.pdf(x, mu, sigma)
        plt.plot(x,y,label=iris.target_names[clss])
        plt.legend()
plt.show()

DTC=DecisionTreeClassifier()
DTC.fit(train_data,train_label)

dot_data=StringIO()
tree.export_graphviz(DTC,out_file=dot_data)

graph=pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf('DecisionTree.pdf')

print(NB.score(test_data,test_label))
print(DTC.score(test_data,test_label))