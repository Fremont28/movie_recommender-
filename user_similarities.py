#This code is uses K-Nearest Neighbors (KNN) and matrix factorization for finding users that
#like similar movies 

import pandas as pd 
import numpy as np 
from scipy.sparse import csr_matrix 
import sklearn 
from sklearn.decomposition import TruncatedSVD 
from sklearn.neighbors import NearestNeighbors

#import movies and users dataset 
movies=pd.read_csv("movies.csv",sep=":")
movies=movies[["1","1193","5","978300760"]]
movies.columns=['user_id','movie_id','rating','algo']

#1. 
##collaborative filter algorithm (KNN)
movies.head(5)

#count movies and users 
movies1=movies.dropna(axis=0,subset=["movie_id"])
movie_counts=movies1['movie_id'].value_counts()
movie_counts=pd.DataFrame(movie_counts)
movie_counts.head(4)
type(movie_counts)
movie_counts.reset_index(level=0,inplace=True)
movie_counts.columns=['movie_id','movie_count']

#combine movie and user counts 
movie_counts.head(3)
moviesX=movies1.merge(movie_counts,on="movie_id")
moviesX.movie_count.max() #268 
moviesX.head(4)

#distributioon of movie ratings 
moviesX['movie_count'].describe() #mean 59.1, 75% 81, 25% 22 
moviesX['movie_count'].quantile(0.95) #166 

#movies with at least 59 reviews 
movies_pop=moviesX[moviesX['movie_count']>59]
movies_pop.shape 

movies_pop_pivot=movies_pop.pivot(index='movie_id',columns='user_id',values='rating').fillna(0)
movies_pop_matrix=csr_matrix(movies_pop_pivot.values) 

#KNN (convert table to 2d matrix)-fill missing values with zeros 
knn_model=NearestNeighbors(metric='cosine',algorithm='brute')
knn_model.fit(movies_pop_matrix)

#test out the knn model for recommendations 
query_index=np.random.choice(movies_pop_pivot.shape[0]) #spits out a random movie selection
distances,indices=knn_model.kneighbors(movies_pop_pivot.iloc[query_index,:].reshape(1,-1),n_neighbors=10)
movies_similarities=np.hstack((indices,distances))

#2. 
##collaborative filtering recommendations (using matrix factorization)
movies_pop_pivot_matrix
movies_pop_pivot.head(4)

SVD=TruncatedSVD(n_components=14,random_state=5)
matrix=SVD.fit_transform(movies_pop_matrix)
matrix.shape #(266,14) 

#calculate a pearson's correlation coefficient for each user-movie pair in the matrix
corr=np.corrcoef(matrix)
corr.shape 

#comparing users based on movie similarities 
#top 3 user similartiies (based on correlations)
user_2=corr[:,1]
user_2.shape 
user_2_scores=user_2.argsort()[-3:][::-1]

user_3=corr[:,2]
user_3_scores=user_3.argsort()[-3:][::-1]

user_4=corr[:,4]
user_4_scores=user_4.argsort()[-3:][::-1]

user_200=corr[:,199]
user_200_scores=user_200.argsort()[-3:][::-1]
