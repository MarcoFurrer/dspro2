from keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001, 
                 beta_1=0.9, beta_2=0.999, 
                 name="Adam"
                 )