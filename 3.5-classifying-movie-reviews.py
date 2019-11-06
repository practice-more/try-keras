import keras
print(keras.__version__)

from keras.datasets import imdb
# local data: ~/.keras/datasets/"fname".tar.gz

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data.shape)

print(train_data[0][:10])
print(train_labels[0])

print('max value of data is {}'.format(max([max(s) for s in train_data])))

word_index_dict = imdb.get_word_index()
#print([(key,value) for (key, value ) in word_index_dict.items()][:10] )

reverse_word_index_dict = dict([(value, key) for (key, value) in word_index_dict.items()])
#print([(key,value) for (key, value ) in reverse_word_index_dict.items()][:10] )

decoded_review = ' '.join([reverse_word_index_dict.get(i -3, '?') for i in train_data[0]])
print(decoded_review)


# preparing the data

import numpy as np
from keras.utils import np_utils

def vectorize_sequences(sequences, dimension=10000):
    result = np.zeros((len(sequences), dimension))
    print(result.shape)
    print(sequences.shape)
    for i, sentence in enumerate(sequences):
        if i is 0:
            print("i:{}, sequences:{}".format(str(i), ' '.join([str(s) for s in sentence])))
        result[i, sentence] = 1  #[dim,slice_index]
    return result

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print(x_train.shape)
print(x_train[0])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# train
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
                optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

# draw

import matplotlib.pyplot as plt

print(history.history.keys())
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.show()



