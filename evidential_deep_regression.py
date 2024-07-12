import pandas as pd
import tensorflow as tf
import numpy as np
import evidential_deep_learning as edl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class EvidentialDeepRegression:
    def __init__(self, data_file, epochs=300, batch_size=32):
        self.data_file = data_file
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = None
        self.history = None

    def load_and_preprocess_data(self):
        # Load the CSV file
        df = pd.read_csv(self.data_file)

        # Extract inputs and outputs
        X = df[['Distance', 'Velocity', 'Theta', 'Gamma1', 'Gamma2']].values
        y_safety_loss = df['Safety Loss'].values.reshape(-1, 1)
        y_deadlock_time = df['Deadlock Time'].values.reshape(-1, 1)

        # Transform Theta into sine and cosine components
        Theta = X[:, 2]
        X_sin_cos = np.column_stack((X[:, :2], np.sin(Theta), np.cos(Theta), X[:, 3:]))

        # Normalize the inputs
        X_scaled = self.scaler.fit_transform(X_sin_cos)
        
        return X_scaled, y_safety_loss, y_deadlock_time
    
    def EvidentialRegressionLoss(self, true, pred):
        # Custom loss function to handle the custom regularizer coefficient
        return edl.losses.EvidentialRegression(true, pred, coeff=1e-2)

    def build_and_compile_model(self, input_shape):
        # Define the evidential deep learning model for both outputs
        input_layer = tf.keras.layers.Input(shape=(input_shape,))
        dense_1 = tf.keras.layers.Dense(128, activation="relu")(input_layer)
        dense_2 = tf.keras.layers.Dense(128, activation="relu")(dense_1)
        dense_3 = tf.keras.layers.Dense(64, activation="relu")(dense_2)
        dense_4 = tf.keras.layers.Dense(64, activation="relu")(dense_3)

        # Output layer for safety loss
        output_safety_loss = edl.layers.DenseNormalGamma(1)(dense_4)

        # Output layer for deadlock time
        output_deadlock_time = edl.layers.DenseNormalGamma(1)(dense_4)

        # Define the model with two outputs
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=[output_safety_loss, output_deadlock_time])


        # Compile the model with multiple losses
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss={
                'dense_normal_gamma': self.EvidentialRegressionLoss,
                'dense_normal_gamma_1': self.EvidentialRegressionLoss
            }
        )

    def train_model(self, X_scaled, y_safety_loss, y_deadlock_time):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_scaled, [y_safety_loss, y_deadlock_time],
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            validation_split=0.2,
            callbacks=[early_stopping]
        )

    def save_model(self, model_name):
        self.model.save(model_name)

    def load_saved_model(self, model_name):
        self.model = tf.keras.models.load_model(
            model_name, 
            custom_objects={
                'DenseNormalGamma': edl.layers.DenseNormalGamma, 
                'EvidentialRegression': EvidentialDeepRegression.EvidentialRegressionLoss
            }
        )

        
def plot_training_history(history):
    plt.figure(figsize=(12, 6))

    # Safety Loss Model Training History
    plt.subplot(1, 2, 1)
    plt.plot(history.history['dense_normal_gamma_loss'], label='Training Loss - Safety Loss')
    plt.plot(history.history['val_dense_normal_gamma_loss'], label='Validation Loss - Safety Loss')
    plt.title('Safety Loss Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Deadlock Time Model Training History
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dense_normal_gamma_1_loss'], label='Training Loss - Deadlock Time')
    plt.plot(history.history['val_dense_normal_gamma_1_loss'], label='Validation Loss - Deadlock Time')
    plt.title('Deadlock Time Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def evaluate_predictions(y_true, y_pred, name):
    mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
    mu = mu.numpy()[:, 0]

    mse = mean_squared_error(y_true, mu)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, mu)

    print(f"Evaluation metrics for {name}:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print()



if __name__ == "__main__":
    epochs = 300
    batch_size = 32
    Train_model = True
    model_name = 'EDR_model.h5'
    data_file = 'data_generation_results_7datapoint.csv'
    
    edr = EvidentialDeepRegression(data_file, epochs=epochs, batch_size=batch_size)
    X_scaled, y_safety_loss, y_deadlock_time = edr.load_and_preprocess_data()
    
    # Registering the custom loss function in TensorFlow's serialization framework:
    tf.keras.utils.get_custom_objects().update({
        'EvidentialRegressionLoss': EvidentialDeepRegression.EvidentialRegressionLoss
    })
    
    if Train_model:
        edr.build_and_compile_model(X_scaled.shape[1])
        edr.train_model(X_scaled, y_safety_loss, y_deadlock_time)
        edr.save_model(model_name)
        plot_training_history(edr.history)
    else:
        edr.load_saved_model(model_name)    

    # Predict and plot using the trained model
    y_pred_safety_loss, y_pred_deadlock_time = edr.model.predict(X_scaled)

    # Evaluate predictions for safety loss
    evaluate_predictions(y_safety_loss, y_pred_safety_loss, 'Safety Loss')

    # Evaluate predictions for deadlock time
    evaluate_predictions(y_deadlock_time, y_pred_deadlock_time, 'Deadlock Time')
    
    