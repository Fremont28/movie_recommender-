#import libraries 
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer

#news data
paper=pd.read_csv("third_eye.csv")
paper['text']=paper['text'].dropna(how="all")
paper['text'].head() 

#tf-idf analysis 
#define tf-idf vectorizer and remove stopwords
tfidf=TfidfVectorizer(stop_words='english')
#replace Nan con una empty string
paper['text']=paper['text'].fillna('')

#create a tf-idf matrix
tfidf_matrix=tfidf.fit_transform(paper['text'])
tfidf_matrix.shape 

#cosine similarity matrix 
cosine_similarity=linear_kernel(tfidf_matrix,tfidf_matrix)
cosine_similarity

#create a reverse map of indices
indices=pd.Series(paper.index,index=paper['channel'])

#tv network similarity algorithm 
paper.info() 
paper['channel'].values 

def news_filter(channel,cosine_similarity=cosine_similarity):
    #get the index of news channel
    idx=indices[channel]
    #pairwise similarity score
    sim_scores=list(enumerate(cosine_similarity[idx]))
    #sort channels on similarity
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    #similar channels?
    sim_scores=sim_scores[1:3]
    #movie indices
    channel_indices=[i[0] for i in sim_scores]
    return paper['channel'].iloc[channel_indices]

news_filter('CNNW') #finds similar networks to CNN World (based on text shown on screen)


