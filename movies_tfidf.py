#5/23/18 
#tf-idf for movie recommendations

#import imdb movie dataset 
cinema=pd.read_csv("tmdb_5000_movies.csv",encoding="latin-1")
cinema.head(3)
len(cinema)

#count movies 
movie_grouped=cinema.groupby(['original_title']).agg({'popularity':'count'}).reset_index() 

#count number of unique movies
movie_count=cinema['original_title'].unique() 
len(movie_count)
lengua_count=cinema['original_language'].unique() 
lengua_count

#tf-idf algorithm (for movie recommendations)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(stop_words='english')
cinema['overview']=cinema['overview'].fillna('')
#create the tf-idf matrix por fitting the transforming data
tfidf_matrix=tfidf.fit_transform(cinema['overview']) 
tfidf_matrix.shape 

#cosine similarity matrix
cosine_sim=linear_kernel(tfidf_matrix,tfidf_matrix) 
#movie id indices 
indices=pd.Series(cinema.index,index=cinema['original_title']).drop_duplicates() #for indexing movies

#create an algorithm that links 20 most similar movies/shows  
def get_recs(title,cosine_sim=cosine_sim):
    #get the index de movie que matches the title
    idx=indices[title]
    #pairwise similarity score 
    sim_scores=list(enumerate(cosine_sim[idx])) #enumumerate cosine sim. based on the movie index 
    #sort movies based on similarity? 
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    #get scores de the 10 most sim. movies
    sim_scores=sim_scores[1:20]
    #get the movie indices
    movie_indices=[i[0] for i in sim_scores] 
    #return top 20 movies 
    return cinema['original_title'].iloc[movie_indices]

get_recs("No Reservations") 




