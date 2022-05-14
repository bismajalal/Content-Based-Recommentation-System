import pickle
import pandas as pd
import tensorflow as tf

# open a file, where you stored the pickled data
file = open('matrix', 'rb')
# dump information to that file
simMatrix = pickle.load(file)
# close the file
file.close()

file = open('metadata', 'rb')
metadata = pickle.load(file)
file.close()

# Save the model in SavedModel format.
tf.saved_model.save(simMatrix)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(simMatrix)
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.gfile.GFile('model.tflite', 'wb') as f:
  f.write(tflite_model)

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