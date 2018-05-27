#5/27/18 
cinema_subset=cinema[["original_title","revenue","runtime","vote_average","vote_count"]]
cinema_subset=cinema_subset.fillna(0)

# movie weighted score 
vc=cinema_subset['vote_count']
vc_mean=cinema_subset['vote_count'].mean() 
v_mean=cinema_subset['vote_average']

def score_weight(x,vc_mean=vc_mean):
    vc=x['vote_count']
    v_mean=x['vote_average']
    return (vc/(vc_mean))*(v_mean)

cinema_subset['weighted_score']=cinema_subset.apply(score_weight,axis=1) #leggero: https://www.datacamp.com/community/tutorials/recommender-systems-python

#split the data into train and test sets
train,test=train_test_split(cinema_subset,test_size=0.25,random_state=97)
weighted_scores=train.sort_values('weighted_score',ascending=False)
weighted_scores[['original_title','weighted_score']].head(15) #top movies are Inception, Dark Knight, and Interstellar 


#subset movies (vote count above at least the 50% quantile)
cinema_50=cinema['vote_count'].quantile(0.50) #235 votes is the 50% quantile
cinema_sub1=cinema[cinema['vote_count']>235.0]

#quick data visualization
import matplotlib.pyplot as plt

# histogram (vote_average) for all movies 
cinema.vote_average.hist(bins=[0,2,4,6,8,10])
plt.title('Vote Average Distribution\n')
plt.xlabel('Average Rating')
plt.ylabel('Count')
plt.savefig('vote_avg.png', bbox_inches='tight')
plt.show()

#subset the movie data 
cinema_xx=cinema[['original_title','id','revenue','runtime','vote_average','vote_count']]
cinema_xx.shape #4803,5

#create pivot table for movie titles
cinema_xx=cinema_xx.drop_duplicates(['original_title','id'])
movie_title_pivot=cinema_xx.pivot(index='original_title',columns='id',values='vote_average').fillna(0) 
movie_title_pivot.shape #4801,4803
movie_title_matrix=csr_matrix(movie_title_pivot.values)
movie_title_matrix 

knn_model=NearestNeighbors(algorithm='brute',metric='cosine')
knn_model.fit(movie_title_matrix)

#test the knn model (for movie recommendations)
query_index=np.random.choice(movie_title_pivot.shape[0]) 
distances, indices = model_knn.kneighbors(movie_title_pivot.iloc[query_index, :].reshape(1, -1), n_neighbors = 9) 

### matrix factorization 
movie_title_pivot 
movie_title_pivot.head() #5 rows, 4803 columns 
movie_title_pivot.shape 

X=movie_title_pivot.shape 
X=movie_title_pivot.values.T 
X.shape #4803,4801 

import sklearn 
from sklearn.decomposition import TruncatedSVD

SVD=TruncatedSVD(n_components=12,random_state=0)
matrix=SVD.fit_transform(X)
matrix.shape #4803,12 (reduces dimensions)

import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)

corr=np.corrcoef(matrix)
corr.shape #4803,4803 

movie_titles=movie_title_pivot.columns 
movie_titles.shape #4803, 
movie_titles_list=list(movie_titles)
movie_titles[0:100]

#finding movie indexes
cinema_xx['original_title'].iloc[98] #the hobbit movie 

#find a movie similar to the The Hobbit
movie_sims=movie_titles_list.index(98)
movie_sims #index 41
cinema_xx['original_title'].iloc[41] #green lantern (is similar to the hobbit movie)



























