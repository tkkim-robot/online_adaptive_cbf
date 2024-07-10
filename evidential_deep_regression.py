import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data
df = pd.read_csv('/mnt/data/data_generation_results_7datapoint.csv')

# Define input features and outputs
X = df[['Distance', 'Velocity', 'Theta', 'Gamma1', 'Gamma2']].values
y_safety_loss = df['Safety Loss'].values
y_deadlock_time = df['Deadlock Time'].values

# Split the data into training and testing sets
X_train, X_test, y_train_safety_loss, y_test_safety_loss, y_train_deadlock_time, y_test_deadlock_time = train_test_split(X, y_safety_loss, y_deadlock_time, test_size=0.2, random_state=42)

# Define the evidential model
def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(4, activation=None)  # The output layer has 4 units for evidential regression
    ])
    return model

model_safety_loss = create_model(X_train.shape[1])
model_deadlock_time = create_model(X_train.shape[1])

# Define evidential loss functions (adapted from the repository)
def evidential_loss(true, pred):
    # Unpack the outputs
    gamma, v, alpha, beta = tf.split(pred, 4, axis=-1)
    # Calculate the NIG loss (negative log-likelihood)
    true = tf.cast(true, tf.float32)
    loss = 0.5 * tf.reduce_sum(gamma * (true - v)**2, axis=-1) + 0.5 * tf.reduce_sum(tf.math.log(beta), axis=-1)
    return loss

# Compile the models
model_safety_loss.compile(optimizer='adam', loss=evidential_loss)
model_deadlock_time.compile(optimizer='adam', loss=evidential_loss)

# Train the models
history_safety_loss = model_safety_loss.fit(X_train, y_train_safety_loss, epochs=100, batch_size=32, validation_split=0.2)
history_deadlock_time = model_deadlock_time.fit(X_train, y_train_deadlock_time, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the models
loss_safety_loss = model_safety_loss.evaluate(X_test, y_test_safety_loss)
loss_deadlock_time = model_deadlock_time.evaluate(X_test, y_test_deadlock_time)
print(f'Safety Loss Model Loss: {loss_safety_loss}')
print(f'Deadlock Time Model Loss: {loss_deadlock_time}')

# Save the models
model_safety_loss.save('model_safety_loss.h5')
model_deadlock_time.save('model_deadlock_time.h5')
