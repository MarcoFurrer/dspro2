import tensorflow as tf
from keras.optimizers import Adam

# Create a learning rate scheduler with decay
initial_learning_rate = 0.001
decay_rate = 0.9
decay_steps = 500

# Learning rate schedule with exponential decay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)

# Create an optimizer with decay and various improvements
optimizer = Adam(
    learning_rate=lr_schedule,  # Dynamic learning rate
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=True,  # Use AMSGrad variant for better convergence
    weight_decay=1e-6  # Add weight decay for additional regularization
)