import pandas as pd
import copy
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#returns name(s) of director(s)
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    #Return NaN if director no available
    return np.nan

#returns name(s) of writer(s)
def get_writer(x):
    for i in x:
        if i['job'] == 'Writer':
            return i['name']
    # Return NaN if director no available
    return np.nan

#returns list of top 3 actors, keywords and genres
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #If more than 3 elements exist, return only first three.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

#converts strings to lower case and removes spaces
def clean_data(x):

    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #if director/writer doesnt exist, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

#join all required columns seperated by space
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' '.join(x['writer']) + ' ' + ' '.join(x['genres'])

def extractData(metadata):

    # apply takes a function and applies it to all values of pandas series.
    ## Parse the stringified features into their corresponding python objects
    features = ['cast', 'crew', 'keywords', 'genres']

    for feature in features:
        metadata[feature] = metadata[feature].apply(literal_eval)

    # Define new director, writer, cast, genres and keywords features that are in a suitable form.
    metadata['director'] = metadata['crew'].apply(get_director)
    metadata['writer'] = metadata['crew'].apply(get_writer)
    features = ['cast', 'keywords', 'genres']
    for feature in features:
        metadata[feature] = metadata[feature].apply(get_list)

    features = ['cast', 'keywords', 'director', 'writer', 'genres']
    for feature in features:
        metadata[feature] = metadata[feature].apply(clean_data)

    # create a string having all metadata we need
    return metadata.apply(create_soup, axis=1)

def computeSimilarity(metadata):

    #CountVectorizer counts word frequecies and removes stop words like 'the, 'and'
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(metadata['soup'])

    simMatrix = []
    for i in range (len(metadata)):
        cosine_sim = cosine_similarity(count_matrix[i], count_matrix)
        sim_scores = list(enumerate(cosine_sim[0]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get the scores of the 50 most similar movies
        sim_scores = sim_scores[0:50]

        simMatrix.append(sim_scores)

    return simMatrix

def main():

    # Load data
    metadata = pd.read_csv('smovies_metadata.csv', low_memory=False)
    credits = pd.read_csv('scredits.csv')
    keywords = pd.read_csv('skeywords.csv')

    # Remove rows with bad IDs.
    #metadata = metadata.drop([19730, 29503, 35587])

    # convert ids to int for merging
    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    metadata['id'] = metadata['id'].astype('int')

    # Merge keywords and credits into your main metadata dataframe
    metadata = metadata.merge(credits, on='id')
    metadata = metadata.merge(keywords, on='id')

    metadata['soup'] = extractData(metadata)
    #returns similarity matrix of each movie with the rest of the movies
    simMatrix = computeSimilarity(metadata)

    # reverse mapping of movie titles and DataFrame indices.
    # a mechanism to identify the index of a movie in your metadata DataFrame, given its title.
    metadata = metadata.reset_index()
    indices = pd.Series(metadata.index, index=metadata['title'])

    stop = 0
    flag = 0
    while int(stop) != 1:
        flag = 0
        movie = input("Enter the exact name of the movie you want recommendations for: ")

        for i in range(len(metadata)):
            if movie == metadata['title'].loc[i]:
                size = input("How many recommendations would you like? (Max 50): ")
                while int(size) > 50:
                    size = input("Please input a number less than 50: ")
                idx = indices[movie]
                # Get the movie indices
                movie_indices = [i[0] for i in simMatrix[idx]]
                movies = movie_indices[0:int(size)]
                print((metadata['title'].iloc[movies]))
                stop = input("Press 1 to exit. Press 0 to continue: ")
                flag = 1
                break
        if int(stop) == 1:
             break
        if flag == 0:
            stop = input("Incorrect name. Press 1 to exit. Press 0 to continue: ")

if __name__=="__main__":
    main()
