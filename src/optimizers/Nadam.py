from keras.optimizers import Nadam

optimizer = Nadam(learning_rate=0.002, 
                  beta_1=0.9, 
                  beta_2=0.999, 
                  name="Nadam"
                  )
