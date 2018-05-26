#5/25/18 ******
import pandas as pd 
import scipy
import numpy as np  

cinema #dataset that contains movie details 
cinema.info() 
cinema_sub=cinema[['id','original_title','revenue','runtime','vote_average','vote_count']].values 

# Avatar movie 
cinema_uno=cinema_sub.iloc[0].values 
cinema_uno.shape 
cinema_unox=cinema_uno[2:6]

#No Reservations tv show 
no_res=cinema[cinema['original_title'].str.match('No Reservations')]
no_res1=no_res[['id','original_title','revenue','runtime','vote_average','vote_count']].values
no_res2=no_res1[:,2:6]

#pirates of the caribbean (world's end)
pirates=cinema_sub[1]
pirates=pirates[2:6]
cinema_subx=cinema_sub[:,2:6]

#comparing movies based on the Euclidean distance 
#based on movie revenue, runtime, average votes, and vote count (popularity)

#simple euclidean distance function (returns movie simialarity
def movie_ratings_sim(x,y):
    x=np.asarray(x)
    y=np.asarray(y)
    return np.sqrt(np.sum((x-y)**2))

movie_ratings_sim(pirates,no_res2) 

# No Reservations vs. Avatar -euclidean distance= 2695364037.0244184
# No Reservations vs. Pirates of the Caribbean (World's End)- euclidean distance= 868398950.0100288

#Anthony Bourdain's No Reservations is more similar to Pirates of the Caribbean World's End than Avatar 