from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(train_data[0,:])

# train data is value of the following classes
#1. Per capita crime rate.
#2. Proportion of residential land zoned for lots over 25,000 square feet.
#3. Proportion of non-retail business acres per town.
#4. Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
#5. Nitric oxides concentration (parts per 10 million).
#6. Average number of rooms per dwelling.
#7. Proportion of owner-occupied units built prior to 1940.
#8. Weighted distances to five Boston employment centres.
#9. Index of accessibility to radial highways.
#10. Full-value property-tax rate per $10,000.
#11. Pupil-teacher ratio by town.
#12. 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
#13. % lower status of the population.

# train target is the price

mean = train_data.mean(axis=0)
train_data -= mean

std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1], )))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# use k-fold validation

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
all_mae_history = []

for i in range(k):
    print("process folder #", i)
    val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
    val_target = train_targets[i * num_val_samples: (i+1) * num_val_samples]

    partial_train_data = np.concatenate(
        [
            train_data[:i * num_val_samples],
            train_data[(i+1) * num_val_samples:]
        ],
        axis=0
    )

    partial_train_targets = np.concatenate(
        [
            train_targets[:i * num_val_samples],
            train_targets[(i+1) * num_val_samples:]
        ],
        axis=0
    )

    model = build_model()

    history = model.fit(
        partial_train_data, partial_train_targets,
        validation_data=(val_data, val_target),
        epochs=num_epochs, batch_size=1, verbose=0)
    
    mae_history = history.history['val_mae']
    all_mae_history.append(mae_history)

    #val_mes, val_mae = model.evaluate(val_data, val_target, verbose=0)
    #all_scores.append(val_mae)

average_mae_history = [
    np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)
]

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points


average_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()



