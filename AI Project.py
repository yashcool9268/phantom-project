import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('C:\\Users\\acer\\Desktop\\shortrev.csv')

dataset['Text'][2]

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

dataset['Text'][0]

clean_review=[]

for i in range(5000):
     text=re.sub('[^a-zA-Z]', ' ',dataset['Text'][i])
     text=text.lower()
     text=text.split()
     text=[ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
     text=" ".join(text)
     clean_review.append(text)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=400)
X=cv.fit_transform(clean_review) 
X=X.toarray()

dataset['label'] = np.nan
for i in range(0,5000):
    if( dataset['Score'][i]>=3):
      dataset['label'][i]=1
    else:
       dataset['label'][i]=0
    


y=dataset['label'].values

############## Naive Bayes Algorithm ###############
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X,y)
nb.score(X,y)

############# Logistic Regression #################

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X,y)
log_reg.score(X,y)


############# Printing all the relevant features for classification ############

print(cv.get_feature_names())   