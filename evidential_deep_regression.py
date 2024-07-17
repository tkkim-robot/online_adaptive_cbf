import pandas as pd
import tensorflow as tf
import numpy as np
import evidential_deep_learning as edl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt




class EvidentialDeepRegression:
    def __init__(self, data_file, epochs=1000, batch_size=32):
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
        dropout_1 = tf.keras.layers.Dropout(0.2)(dense_2)
        dense_3 = tf.keras.layers.Dense(64, activation="relu")(dropout_1)
        dropout_2 = tf.keras.layers.Dropout(0.2)(dense_3)
        dense_4 = tf.keras.layers.Dense(64, activation="relu")(dropout_2)

        # Output layer for safety loss
        output_safety_loss = edl.layers.DenseNormalGamma(1)(dense_4)

        # Output layer for deadlock time
        output_deadlock_time = edl.layers.DenseNormalGamma(1)(dense_4)

        # Define the model with two outputs
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=[output_safety_loss, output_deadlock_time])

        # Compile the model with multiple losses
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(2e-6),
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

        reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-8
        )

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='edr_model_best.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        self.history = self.model.fit(
            X_scaled, [y_safety_loss, y_deadlock_time],
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr_on_plateau, model_checkpoint]
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

    def calculate_uncertainties(self, y_pred):
        mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
        mu = mu.numpy()[:, 0]
        v = v.numpy()[:, 0]
        alpha = alpha.numpy()[:, 0]
        beta = beta.numpy()[:, 0]
        aleatoric_uncertainty = beta / (alpha - 1)
        epistemic_uncertainty = beta / (v * (alpha - 1))
        return mu, aleatoric_uncertainty, epistemic_uncertainty
    
    
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

def plot_gaussian_with_different_gammas(edr):
    import seaborn as sns
    from scipy.stats import norm    
    
    # Define three different combinations of gamma1 and gamma2
    gamma_combinations = [
        (0.01, 0.01),
        (0.5, 0.5),
        (0.99, 0.99)
    ]
    
    mu_safety_all = []
    sigma_safety_all = []
    mu_deadlock_all = []
    sigma_deadlock_all = []
    
    for gamma1, gamma2 in gamma_combinations:
        # X = df[['Distance', 'Velocity', 'Theta', 'Gamma1', 'Gamma2']].values
        X_nonscaled = [0.516, 0.0825, 0.0017, 0.9999, gamma1, gamma2]
        X_batch = np.tile(X_nonscaled, (32, 1)) 
        X_batch_normed = edr.scaler.transform(X_batch)

        # Predict using the EDR model
        y_pred_safety_loss, y_pred_deadlock_time = edr.model.predict(X_batch_normed)
        
        # Calculate uncertainties for safety loss
        mu_safety_loss, aleatoric_uncertainty_safety_loss, epistemic_uncertainty_safety_loss = edr.calculate_uncertainties(y_pred_safety_loss)
        mu_deadlock_time, aleatoric_uncertainty_deadlock_time, epistemic_uncertainty_deadlock_time = edr.calculate_uncertainties(y_pred_deadlock_time)
        
        # Append results to the lists for safety loss
        mu_safety_all.append(mu_safety_loss[0])
        sigma_safety_all.append(np.sqrt(aleatoric_uncertainty_safety_loss[0] + epistemic_uncertainty_safety_loss[0]))
        
        # Append results to the lists for deadlock time
        mu_deadlock_all.append(mu_deadlock_time[0])
        sigma_deadlock_all.append(np.sqrt(aleatoric_uncertainty_deadlock_time[0] + epistemic_uncertainty_deadlock_time[0]))
    
    # Plot GMM for safety loss
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'orange', 'green']
    labels = ['γ1 = 0.01, γ2 = 0.01', 'γ1 = 0.5, γ2 = 0.5', 'γ1 = 0.99, γ2 = 0.99']
    
    for i, (mu, sigma, color, label) in enumerate(zip(mu_safety_all, sigma_safety_all, colors, labels)):
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        p = norm.pdf(x, mu, sigma)
        ax[0].plot(x, p, color=color, linestyle='dashed', label=label)
    
    ax[0].legend()
    ax[0].set_xlabel('Safety Loss Prediction (Mu)')
    ax[0].set_ylabel('Probability Density')
    ax[0].set_title('Safety Loss GMM of EDR Predictions')
    
    # Plot GMM for deadlock time
    for i, (mu, sigma, color, label) in enumerate(zip(mu_deadlock_all, sigma_deadlock_all, colors, labels)):
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        p = norm.pdf(x, mu, sigma)
        ax[1].plot(x, p, color=color, linestyle='dashed', label=label)
    
    ax[1].legend()
    ax[1].set_xlabel('Deadlock Time Prediction (Mu)')
    ax[1].set_ylabel('Probability Density')
    ax[1].set_title('Deadlock Time GMM of EDR Predictions')
    
    plt.show()

def plot_gmm_with_different_gammas(edr):
    import seaborn as sns
    from sklearn.mixture import GaussianMixture
        
    # Define three different combinations of gamma1 and gamma2
    gamma_combinations = [
        (0.01, 0.01),
        (0.5, 0.5),
        (0.99, 0.99)
    ]
    
    mu_safety_all = []
    mu_deadlock_all = []
    
    for gamma1, gamma2 in gamma_combinations:
        # Generate varied samples
        X_nonscaled = np.array([0.516, 0.0825, 0.0017, 0.9999])
        gamma_values = np.tile([gamma1, gamma2], (32, 1))
        noise = np.random.normal(0, 0.01, size=(32, 4))  # Add some noise to the features
        X_batch = np.hstack((X_nonscaled + noise, gamma_values))  # Create a batch of varied samples

        X_batch_normed = edr.scaler.transform(X_batch)

        # Predict using the EDR model
        y_pred_safety_loss, y_pred_deadlock_time = edr.model.predict(X_batch_normed)
        
        # Calculate uncertainties for safety loss
        mu_safety_loss, aleatoric_uncertainty_safety_loss, epistemic_uncertainty_safety_loss = edr.calculate_uncertainties(y_pred_safety_loss)
        mu_deadlock_time, aleatoric_uncertainty_deadlock_time, epistemic_uncertainty_deadlock_time = edr.calculate_uncertainties(y_pred_deadlock_time)
        
        # Append results to the lists for safety loss
        mu_safety_all.extend(mu_safety_loss)
        
        # Append results to the lists for deadlock time
        mu_deadlock_all.extend(mu_deadlock_time)
    
    # Convert lists to numpy arrays for GMM fitting
    mu_safety_all = np.array(mu_safety_all).reshape(-1, 1)
    mu_deadlock_all = np.array(mu_deadlock_all).reshape(-1, 1)
    
    # Fit GMM and plot for safety loss
    gmm_safety = GaussianMixture(n_components=3).fit(mu_safety_all)
    x_safety = np.linspace(mu_safety_all.min() - 1, mu_safety_all.max() + 1, 1000)
    logprob_safety = gmm_safety.score_samples(x_safety.reshape(-1, 1))
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(mu_safety_all.flatten(), kde=False, stat='density', bins=30, label='Data', ax=ax[0])
    ax[0].plot(x_safety, np.exp(logprob_safety), color='black', linestyle='dashed', label='GMM')
    ax[0].legend()
    ax[0].set_xlabel('Safety Loss Prediction (Mu)')
    ax[0].set_ylabel('Probability Density')
    ax[0].set_title('Safety Loss GMM of EDR Predictions')
    
    # Fit GMM and plot for deadlock time
    gmm_deadlock = GaussianMixture(n_components=3).fit(mu_deadlock_all)
    x_deadlock = np.linspace(mu_deadlock_all.min() - 1, mu_deadlock_all.max() + 1, 1000)
    logprob_deadlock = gmm_deadlock.score_samples(x_deadlock.reshape(-1, 1))
    
    sns.histplot(mu_deadlock_all.flatten(), kde=False, stat='density', bins=30, label='Data', ax=ax[1])
    ax[1].plot(x_deadlock, np.exp(logprob_deadlock), color='black', linestyle='dashed', label='GMM')
    ax[1].legend()
    ax[1].set_xlabel('Deadlock Time Prediction (Mu)')
    ax[1].set_ylabel('Probability Density')
    ax[1].set_title('Deadlock Time GMM of EDR Predictions')
    
    plt.show()

def plot_overlay_gaussian_gmm_with_different_gammas(edr):
    import seaborn as sns
    from scipy.stats import norm    
    from sklearn.mixture import GaussianMixture
    
    # Define three different combinations of gamma1 and gamma2
    gamma_combinations = [
        (0.01, 0.01),
        (0.5, 0.5),
        (0.99, 0.99)
    ]
    
    mu_safety_all = []
    sigma_safety_all = []
    mu_deadlock_all = []
    sigma_deadlock_all = []
    
    for gamma1, gamma2 in gamma_combinations:
        # Generate varied samples
        X_nonscaled = np.array([0.516, 0.0825, 0.0017, 0.9999])
        gamma_values = np.tile([gamma1, gamma2], (32, 1))
        noise = np.random.normal(0, 0.01, size=(32, 4))  # Add some noise to the features
        X_batch = np.hstack((X_nonscaled + noise, gamma_values))  # Create a batch of varied samples

        X_batch_normed = edr.scaler.transform(X_batch)

        # Predict using the EDR model
        y_pred_safety_loss, y_pred_deadlock_time = edr.model.predict(X_batch_normed)
        
        # Calculate uncertainties for safety loss
        mu_safety_loss, aleatoric_uncertainty_safety_loss, epistemic_uncertainty_safety_loss = edr.calculate_uncertainties(y_pred_safety_loss)
        mu_deadlock_time, aleatoric_uncertainty_deadlock_time, epistemic_uncertainty_deadlock_time = edr.calculate_uncertainties(y_pred_deadlock_time)
        
        # Append results to the lists for safety loss
        mu_safety_all.extend(mu_safety_loss)
        sigma_safety_all.extend(np.sqrt(aleatoric_uncertainty_safety_loss + epistemic_uncertainty_safety_loss))
        
        # Append results to the lists for deadlock time
        mu_deadlock_all.extend(mu_deadlock_time)
        sigma_deadlock_all.extend(np.sqrt(aleatoric_uncertainty_deadlock_time + epistemic_uncertainty_deadlock_time))
    
    # Convert lists to numpy arrays for GMM fitting
    mu_safety_all = np.array(mu_safety_all).reshape(-1, 1)
    sigma_safety_all = np.array(sigma_safety_all).reshape(-1, 1)
    mu_deadlock_all = np.array(mu_deadlock_all).reshape(-1, 1)
    sigma_deadlock_all = np.array(sigma_deadlock_all).reshape(-1, 1)
    
    # Fit GMM and plot for safety loss
    gmm_safety = GaussianMixture(n_components=3).fit(mu_safety_all)
    x_safety = np.linspace(mu_safety_all.min() - 1, mu_safety_all.max() + 1, 1000)
    logprob_safety = gmm_safety.score_samples(x_safety.reshape(-1, 1))
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.histplot(mu_safety_all.flatten(), kde=False, stat='density', bins=30, label='Data', ax=ax[0])
    ax[0].plot(x_safety, np.exp(logprob_safety), color='black', linestyle='dashed', label='GMM')

    colors = ['blue', 'orange', 'green']
    labels = ['γ1 = 0.01, γ2 = 0.01', 'γ1 = 0.5, γ2 = 0.5', 'γ1 = 0.99, γ2 = 0.99']
    
    for i, (mu, sigma, color, label) in enumerate(zip(mu_safety_all[::32], sigma_safety_all[::32], colors, labels)):
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        p = norm.pdf(x, mu, sigma)
        ax[0].plot(x, p, color=color, linestyle='dashed', label=label)
    
    ax[0].legend()
    ax[0].set_xlabel('Safety Loss Prediction (Mu)')
    ax[0].set_ylabel('Probability Density')
    ax[0].set_title('Safety Loss GMM of EDR Predictions')
    
    # Fit GMM and plot for deadlock time
    gmm_deadlock = GaussianMixture(n_components=3).fit(mu_deadlock_all)
    x_deadlock = np.linspace(mu_deadlock_all.min() - 1, mu_deadlock_all.max() + 1, 1000)
    logprob_deadlock = gmm_deadlock.score_samples(x_deadlock.reshape(-1, 1))
    
    sns.histplot(mu_deadlock_all.flatten(), kde=False, stat='density', bins=30, label='Data', ax=ax[1])
    ax[1].plot(x_deadlock, np.exp(logprob_deadlock), color='black', linestyle='dashed', label='GMM')
    
    for i, (mu, sigma, color, label) in enumerate(zip(mu_deadlock_all[::32], sigma_deadlock_all[::32], colors, labels)):
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        p = norm.pdf(x, mu, sigma)
        ax[1].plot(x, p, color=color, linestyle='dashed', label=label)
    
    ax[1].legend()
    ax[1].set_xlabel('Deadlock Time Prediction (Mu)')
    ax[1].set_ylabel('Probability Density')
    ax[1].set_title('Deadlock Time GMM of EDR Predictions')
    
    plt.show()


if __name__ == "__main__":
    Train_model = False
    model_name = 'edr_model_best_0713.h5'
    data_file = 'data_generation_results_8datapoint.csv'
    
    batch_size = 128
    edr = EvidentialDeepRegression(data_file, batch_size=batch_size)
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

    # # Predict and plot using the trained model
    # y_pred_safety_loss, y_pred_deadlock_time = edr.model.predict(X_scaled)

    # evaluate_predictions(y_safety_loss, y_pred_safety_loss, 'Safety Loss')
    # evaluate_predictions(y_deadlock_time, y_pred_deadlock_time, 'Deadlock Time')
            
    # # Calculate uncertainties for safety loss
    # mu_safety_loss, aleatoric_uncertainty_safety_loss, epistemic_uncertainty_safety_loss = edr.calculate_uncertainties(y_pred_safety_loss)
    # print("Safety Loss Prediction and Uncertainties:")
    # print("Prediction:", mu_safety_loss)
    # print("Aleatoric Uncertainty:", aleatoric_uncertainty_safety_loss)
    # print("Epistemic Uncertainty:", epistemic_uncertainty_safety_loss)
    
    # # Calculate uncertainties for deadlock time
    # mu_deadlock_time, aleatoric_uncertainty_deadlock_time, epistemic_uncertainty_deadlock_time = edr.calculate_uncertainties(y_pred_deadlock_time)
    # print("Deadlock Time Prediction and Uncertainties:")
    # print("Prediction:", mu_deadlock_time)
    # print("Aleatoric Uncertainty:", aleatoric_uncertainty_deadlock_time)
    # print("Epistemic Uncertainty:", epistemic_uncertainty_deadlock_time)
            
            
    plot_gaussian_with_different_gammas(edr)
    plot_gmm_with_different_gammas(edr)
    plot_overlay_gaussian_gmm_with_different_gammas(edr)




