import numpy as np
import keras
from keras import Sequential
from keras.layers import Dense

def matrix2board(matrix):
    out_matrix = np.zeros((8,8), dtype=int)
    for i in range(12):
        out_matrix[:] += matrix[i*64:(i+1)*64].reshape((8,8))*(i+1)
    out_matrix[out_matrix > 6] -= 13
    return out_matrix

input_boards = np.load("all_boards_large.npy")
output_moves = np.load("all_moves_large.npy")
print(input_boards.shape, output_moves.shape)

model = Sequential()
model.add(Dense(128, input_shape=(768,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(input_boards, output_moves, epochs=40, batch_size=4096*16)

model.save("model_deeper128")

# input_boards = np.load("all_boards2.npy")
# output_moves = np.load("all_moves2.npy")


# model = keras.models.load_model("model")
# prediction = model.predict(input_boards[:100])

# for i in range(10):
#     print(matrix2board(input_boards[i]))
#     print(np.unravel_index(np.argmax(prediction[i,:64].reshape((8,8))), (8,8)))
#     print(np.unravel_index(np.argmax(prediction[i,64:].reshape((8,8))), (8,8)))