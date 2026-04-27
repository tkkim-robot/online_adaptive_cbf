import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from nn_model.penn.nn_iccbf_predict import ProbabilisticEnsembleNN  


def generate_new_dataset(gamma_pairs, distance_range, velocity_range, theta_range, num_samples_per_param=10):
    """
    Generates a dataset with all combinations of gamma1 and gamma2 pairs,
    and samples distances, velocities, and thetas within the specified ranges.
    """
    gamma1_vals, gamma2_vals = zip(*gamma_pairs)
    distance_vals = np.linspace(distance_range[0], distance_range[1], num_samples_per_param)
    velocity_vals = np.linspace(velocity_range[0], velocity_range[1], num_samples_per_param)
    theta_vals = np.linspace(theta_range[0], theta_range[1], num_samples_per_param)

    dataset = []
    for gamma1, gamma2 in gamma_pairs:
        for distance in distance_vals:
            for velocity in velocity_vals:
                for theta in theta_vals:
                    dataset.append([distance, velocity, theta, gamma1, gamma2])
    return np.array(dataset)

def plot_3d_grid_mean_predictions(X_original, mean_predictions, gamma1, gamma2, subplot_position, vmin, vmax, cmap='RdYlBu_r', tolerance=1e-4):
    """
    Plots a single 3D scatter plot for given gamma1 and gamma2 with approximate masking,
    using a consistent color scale across all subplots.
    """
    ax = plt.subplot(3, 3, subplot_position, projection='3d')
    mask = (
        np.isclose(X_original[:, 3], gamma1, atol=tolerance) &
        np.isclose(X_original[:, 4], gamma2, atol=tolerance)
    )
    x = X_original[mask][:, 0]  # Distance
    y = X_original[mask][:, 1]  # Velocity
    z = X_original[mask][:, 2]  # Theta
    preds = mean_predictions[mask]  # Mean predictions

    if len(x) == 0:
        ax.text(0.5, 0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    else:
        scatter = ax.scatter(x, y, z, c=preds, cmap=cmap, alpha=0.6, vmin=vmin, vmax=vmax)
        ax.set_xlabel('Distance [m]', fontsize=7)
        ax.set_ylabel('Velocity [m/s]', fontsize=7)
        ax.set_zlabel('Relative Angle [rad]', fontsize=7)
        ax.set_title(f'Gamma1: {gamma1:.2f}, Gamma2: {gamma2:.2f}', fontsize=8)

        ax.invert_xaxis()



# model_path = 'nn_model/checkpoint/penn_model_0907.pth' 
# scaler_path = 'nn_model/checkpoint/scaler_0907.save'  
model_path = 'nn_model/checkpoint/penn_model_0921.pth' 
scaler_path = 'nn_model/checkpoint/scaler_0921.save'  

penn = ProbabilisticEnsembleNN()
penn.load_model(model_path)
penn.load_scaler(scaler_path)
penn.model.eval()

gamma_pairs = [
    (0.03, 0.03), (0.03, 0.10), (0.03, 0.18),
    (0.10, 0.03), (0.10, 0.10), (0.10, 0.18),
    (0.18, 0.03), (0.18, 0.10), (0.18, 0.18)
]

distance_range = (0.6, 3.0)
velocity_range = (0.01, 1.0)
theta_range = (0.01, np.pi / 4)

new_dataset = generate_new_dataset(
    gamma_pairs=gamma_pairs,
    distance_range=distance_range,
    velocity_range=velocity_range,
    theta_range=theta_range,
    num_samples_per_param=10 
)
print(f"Generated dataset shape: {new_dataset.shape}")

# Verify the unique gamma pairs in the dataset
unique_gamma_pairs = np.unique(new_dataset[:, 3:5], axis=0)
print("Unique Gamma1 and Gamma2 pairs in the dataset:")
for pair in unique_gamma_pairs:
    print(f"Gamma1: {pair[0]:.2f}, Gamma2: {pair[1]:.2f}")

# Make predictions using the PENN model
with torch.no_grad():
    y_pred_safety_loss, y_pred_deadlock_time, _ = penn.predict(new_dataset)

# Convert y_pred_safety_loss and y_pred_deadlock_time to NumPy arrays for vectorized operations
# Each y_pred_safety_loss[i] is a list of [mu, sigma_sq] for each ensemble member
safety_loss_mu = np.array([[pred[0] for pred in sample] for sample in y_pred_safety_loss])  # Shape: (num_samples, n_ensemble)
mean_safety_loss = np.mean(safety_loss_mu, axis=1)  # Mean across ensembles

# Similarly, extract mu values for deadlock time across ensembles if needed
deadlock_time_mu = np.array([[pred[0] for pred in sample] for sample in y_pred_deadlock_time])  # Shape: (num_samples, n_ensemble)
mean_deadlock_time = np.mean(deadlock_time_mu, axis=1)  # Mean across ensembles

# Extract original input features
X_original = new_dataset  # Shape: (N, 5) where columns are [Distance, Velocity, Theta, Gamma1, Gamma2]

# Determine global min and max for the color scale
global_min = mean_safety_loss.min()
# global_max = mean_safety_loss.max()
global_max = 0.40
print(f"Global min safety loss: {global_min}")
print(f"Global max safety loss: {global_max}")

fig = plt.figure(figsize=(10, 10))

# Sort gamma2 and gamma1 in ascending order
sorted_gamma2 = sorted(set(pair[1] for pair in gamma_pairs))  # [0.03, 0.10, 0.18]
sorted_gamma1 = sorted(set(pair[0] for pair in gamma_pairs))  # [0.03, 0.10, 0.18]

for gamma1 in sorted_gamma1:
    for gamma2 in sorted_gamma2:
        # Determine the subplot position
        # Subplots are numbered from 1 to 9, left to right, top to bottom
        # (gamma1, gamma2) arranged from left bottom, increasing gamma1 left to right and gamma2 bottom to top
        # Hence, gamma2=0.03 (bottom row), gamma2=0.10 (middle row), gamma2=0.18 (top row)
        # gamma1=0.03 (left column), gamma1=0.10 (middle column), gamma1=0.18 (right column)
        # Calculate row index: 0 (bottom) to 2 (top)
        row = sorted_gamma2.index(gamma2)
        # Calculate column index: 0 (left) to 2 (right)
        col = sorted_gamma1.index(gamma1)
        # Calculate subplot position
        subplot_position = (len(sorted_gamma2) - row - 1) * 3 + col + 1  # Rows are top to bottom

        # Plot mean safety loss predictions
        plot_3d_grid_mean_predictions(
            X_original=X_original,
            mean_predictions=mean_safety_loss,
            gamma1=gamma1,
            gamma2=gamma2,
            subplot_position=subplot_position,
            vmin=global_min,
            vmax=global_max,
            cmap='RdYlBu_r',
            tolerance=1e-4
        )

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
norm = plt.Normalize(vmin=global_min, vmax=global_max)
sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Predicted Safety Loss', fontsize=9)
plt.subplots_adjust(wspace=0.4, hspace=0.4, left=0.05, right=0.8, top=0.95, bottom=0.05)
plt.suptitle('Mean Safety Loss Predictions for Gamma Pairs', fontsize=10)
plt.show()
