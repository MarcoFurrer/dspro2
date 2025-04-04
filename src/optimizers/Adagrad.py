from keras.optimizers import Adagrad

optimizer = Adagrad(learning_rate=0.01, initial_accumulator_value=0.1)
