# Content-Based-Recommentation-System
A Content-Based Movie Recommendation System which takes a movie name as input and retrieves at most 50 recommendations

Project Name: Content-Based Movie Recommendation System

Business Problem: Implement a Content-Based Movie Recommendation System which takes a movie name as input and retrieves at most 50 recommendations. 

Technologies:
•	PyCharm
•	Sklearn
•	Pandas
•	Numpy
•	Pickle

Solution: The model asks the user for the name of the movie and the number of recommendations, N. Using the cosine similarity, the model finds N movies most similar to the input in terms of genre, actors, director(s), writer(s) and keywords. 

In the first version, the cosine similarity found the similarity of one movie to all 45k other movies and returned a 1x45k matrix. Therefore, the consine_similarity() was called 45k times. Then the movies were sorted based on their similarities. The running time was around 50 minutes. 

If we compute similarities of all 45k movies with all the other movies returning,  consine_similarity() would be called only once. This is a faster approach but requires a lot of memory.

To reduce the time complexity, the concept of chunking was used. Similarities of chunks of 100 movies with the rest of the movies were found returning 100x45k matrix. This took 30 minutes.

Model: Another improvement in the version 2 is that the similarity matrix and metadata is saved using pickle. In GetRecommendations.py these files are loaded and the user is prompted to give movie names as input and is given movie recommendations as output. As a result of saving the files, it takes only a few seconds for the recommendations to be displayed.

Future Improvements/Releases: This is a very basic content based recommendation system which does not take into account the rating’s user gave to other movies. This recommendation system can be combined with a collaborative recommendation system to make a hybrid one. The is also room for improvement in terms of the time complexity of this model

Dataset: The dataset used has 7 files containing data about 45000 movies. The dataset used for this model can be found here. https://www.kaggle.com/rounakbanik/the-movies-dataset/home?select=movies_metadata.csv
For faster testing while development, a subset of each file in the dataset was used. 
