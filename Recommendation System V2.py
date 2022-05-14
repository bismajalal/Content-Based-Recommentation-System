import pandas as pd
import copy
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


# returns name(s) of director(s)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    # Return NaN if director no available
    return np.nan


# returns name(s) of writer(s)
def get_writer(x):
    for i in x:
        if i['job'] == 'Writer':
            return i['name']
    # Return NaN if director no available
    return np.nan


# returns list of top 3 actors, keywords and genres
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # If more than 3 elements exist, return only first three.
        if len(names) > 3:
            names = names[:3]
        return names

    # Return empty list in case of missing/malformed data
    return []


# converts strings to lower case and removes spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # if director/writer doesnt exist, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# join all required columns seperated by space
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' '.join(
        x['writer']) + ' ' + ' '.join(x['genres'])


def extractData(data):
    # apply takes a function and applies it to all values of pandas series.
    ## Parse the stringified features into their corresponding python objects
    features = ['cast', 'crew', 'keywords', 'genres']

    for feature in features:
        data[feature] = data[feature].apply(literal_eval)

    # Define new director, writer, cast, genres and keywords features that are in a suitable form.
    data['director'] = data['crew'].apply(get_director)
    data['writer'] = data['crew'].apply(get_writer)
    features = ['cast', 'keywords', 'genres']

    for feature in features:
        data[feature] = data[feature].apply(get_list)

    features = ['cast', 'keywords', 'director', 'writer', 'genres']
    for feature in features:
        data[feature] = data[feature].apply(clean_data)

    # create a string having all metadata we need
    return data.apply(create_soup, axis=1)


def computeSimilarity(metadata):
    # CountVectorizer counts word frequecies and removes stop words like 'the, 'and'
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(metadata['soup'])

    simMatrix = []

    sim_scores = []
    truncated = []
    start = 0
    end = 100
    while (start <= (len(metadata) - 100)):
        # cosine_sim = cosine_similarity(count_matrix[i], count_matrix[i+1: len(metadata)])
        cosine_sim = cosine_similarity(count_matrix[start:end], count_matrix)

        for i in range(100):
            sim_scores = list(enumerate(cosine_sim[i]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Get the scores of the 50 most similar movies
            sim_scores = sim_scores[0:50]
            simMatrix.append(sim_scores)

        # print(start,end)
        start = end
        end = end + 100

    cosine_sim = cosine_similarity(count_matrix[start:len(metadata)], count_matrix)

    for i in range(len(metadata) - start):
        sim_scores = list(enumerate(cosine_sim[i]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get the scores of the 50 most similar movies
        sim_scores = sim_scores[0:50]
        simMatrix.append(sim_scores)

    return simMatrix


def main():
    # Load data
    metadata = pd.read_csv('Data/smovies_metadata.csv', low_memory=False)
    credits = pd.read_csv('Data/scredits.csv')
    keywords = pd.read_csv('Data/skeywords.csv')

    # Remove rows with bad IDs.
    # metadata = metadata.drop([19730, 29503, 35587])

    # convert ids to int for merging
    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    metadata['id'] = metadata['id'].astype('int')

    # Merge keywords and credits into your main metadata dataframe
    metadata = metadata.merge(credits, on='id')
    metadata = metadata.merge(keywords, on='id')

    data = metadata.filter(['genres', 'id', 'title', 'cast', 'crew', 'keywords'], axis=1)
    data['soup'] = extractData(data)
    data = data.filter(['id', 'title', 'soup'], axis=1)
    # returns similarity matrix of each movie with the rest of the movies
    simMatrix = computeSimilarity(data)

    # open a file, where you ant to store the data
    file = open('matrix', 'wb')
    # dump information to that file
    pickle.dump(simMatrix, file)

    data = data.filter(['id', 'title'], axis=1)
    file = open('metadata', 'wb')
    pickle.dump(data, file)

    # reverse mapping of movie titles and DataFrame indices.
    # a mechanism to identify the index of a movie in your metadata DataFrame, given its title.
    data = data.reset_index()
    indices = pd.Series(data.index, index=data['title'])

    stop = 0
    flag = 0

    while int(stop) != 1:
        flag = 0
        movie = input("Enter the exact name of the movie you want recommendations for: ")

        for i in range(len(data)):
            if movie == data['title'].loc[i]:
                size = input("How many recommendations would you like? (Max 50): ")
                while int(size) > 50:
                    size = input("Please input a number less than 50: ")
                idx = indices[movie]
                # Get the movie indices

                movie_indices = [i[0] for i in simMatrix[idx]]
                movies = movie_indices[0:int(size)]
                print((data['title'].iloc[movies]))
                stop = input("Press 1 to exit. Press 0 to continue: ")
                flag = 1
                break
        if int(stop) == 1:
            break
        if flag == 0:
            stop = input("Incorrect name. Press 1 to exit. Press 0 to continue: ")


if __name__ == "__main__":
    main()
