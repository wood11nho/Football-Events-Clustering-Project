import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
import matplotlib.patches as patches

# Load the dataset
data_path = '../data/events.csv'  # Update the path to your file
data = pd.read_csv(data_path)

# Filter data to include only rows with valid shot_place
shots_data = data.dropna(subset=['shot_place'])

# Ensure at least 1000 samples
if len(shots_data) < 1000:
    raise ValueError("Dataset must have at least 1000 samples after filtering.")

# Reduce the dataset size for quicker execution (e.g., 1000 samples)
shots_data = shots_data.sample(n=1000, random_state=42)
print("Reduced dataset shape:", shots_data.shape)

# Feature Engineering: Add derived features
shots_data['time_since_last_event'] = shots_data.groupby('id_odsp')['time'].diff().fillna(0)

# Define features to include
categorical_columns = ['side', 'bodypart', 'assist_method', 'situation', 'fast_break', 'event_team', 'opponent']
numerical_columns = ['time', 'shot_outcome', 'location', 'is_goal', 'time_since_last_event']
text_column = 'text'

# Handle missing values in numerical columns
shots_data[numerical_columns] = shots_data[numerical_columns].fillna(shots_data[numerical_columns].mean())
print("Data after filling missing values:")
print(shots_data.head())

# Balance the dataset using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(shots_data, shots_data['shot_place'])
shots_data = pd.DataFrame(X_balanced, columns=shots_data.columns)
print("Balanced dataset shape:", shots_data.shape)

# Preprocessing: One-hot encode categorical, scale numerical, and extract text features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns),
        ('text', TfidfVectorizer(max_features=50), text_column)
    ],
    remainder='drop'
)

# Apply preprocessing
processed_data = preprocessor.fit_transform(shots_data).toarray()
print("Processed data shape:", processed_data.shape)

# Hyperparameter tuning for DBSCAN
param_grid = {
    'eps': [0.5, 1.0, 1.5],
    'min_samples': [5, 10, 20],
    'metric': ['euclidean', 'cosine']
}

best_score = -1
best_params = None
best_labels = None

for params in ParameterGrid(param_grid):
    dbscan_model = DBSCAN(**params)
    labels = dbscan_model.fit_predict(processed_data)
    if len(set(labels)) > 1:
        score = silhouette_score(processed_data, labels, metric='euclidean')
        if score > best_score:
            best_score = score
            best_params = params
            best_labels = labels

print("Best parameters:", best_params)
print("Best silhouette score:", best_score)

# Use the best model
dbscan_model = DBSCAN(**best_params)
labels = dbscan_model.fit_predict(processed_data)
shots_data['cluster'] = labels
print("Cluster labels:", labels[:10])

# Evaluate clustering
unique_labels = set(labels)
print(f"Clusters found: {unique_labels}")
if len(unique_labels) > 1:
    silhouette = silhouette_score(processed_data, labels, metric='euclidean')
    davies_bouldin = davies_bouldin_score(processed_data, labels)
    calinski_harabasz = calinski_harabasz_score(processed_data, labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
else:
    print("Clustering metrics cannot be computed with fewer than 2 clusters.")
    
# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(processed_data)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    pca_data[:, 0], 
    pca_data[:, 1], 
    c=labels, 
    cmap='Spectral', 
    s=40, 
    edgecolor='k', 
    alpha=0.8
)
plt.title('PCA Visualization of Clusters', fontsize=14)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.colorbar(scatter, label="Cluster Index")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Analyze shot_place proportions within each cluster
cluster_proportions = shots_data.groupby('cluster')['shot_place'].value_counts(normalize=True).unstack(fill_value=0)
print("Proportion of shot_place in each cluster:")
print(cluster_proportions)

# Assign shot_place label to each cluster based on majority vote
cluster_to_label = cluster_proportions.idxmax(axis=1)
print("Cluster-to-shot_place mapping:")
print(cluster_to_label)

# Map the predicted labels back to the dataset
shots_data['predicted_shot_place'] = shots_data['cluster'].map(cluster_to_label)

# Compute accuracy
valid_rows = shots_data['predicted_shot_place'].notna()
accuracy = accuracy_score(shots_data.loc[valid_rows, 'shot_place'], shots_data.loc[valid_rows, 'predicted_shot_place'])
print(f"Accuracy of clustering-based shot_place prediction: {accuracy:.4f}")
# Save the final dataset
shots_data.to_csv('dbscan_clustered_shots_with_predictions.csv', index=False)
print("Clustered data with predictions saved to 'dbscan_clustered_shots_with_predictions.csv'.")

# Mapping of shot places to goal coordinates
shot_place_map = {
    1: "Bit too high", 2: "Blocked", 3: "Bottom left corner", 4: "Bottom right corner", 5: "Centre of the goal",
    6: "High and wide", 7: "Hits the bar", 8: "Misses to the left", 9: "Misses to the right", 10: "Too high",
    11: "Top centre of the goal", 12: "Top left corner", 13: "Top right corner"
}

# Define goal regions (intervals for plotting shots)
goal_regions = {
    "Bottom left corner": ((0, 2.44 / 3), (0, 7.32 / 3)),
    "Bottom right corner": ((0, 2.44 / 3), (7.32 * 2 / 3, 7.32)),
    "Centre of the goal": ((2.44 / 3, 2.44 * 2 / 3), (7.32 / 3, 7.32 * 2 / 3)),
    "Top left corner": ((2.44 * 2 / 3, 2.44), (0, 7.32 / 3)),
    "Top right corner": ((2.44 * 2 / 3, 2.44), (7.32 * 2 / 3, 7.32)),
    "Top centre of the goal": ((2.44 * 2 / 3, 2.44), (7.32 / 3, 7.32 * 2 / 3)),
    "Misses to the left": ((0, 2.44), (-1, 0)),
    "Misses to the right": ((0, 2.44), (7.32, 7.32 + 1)),
    "Too high": ((2.44, 2.44 + 1), (0, 7.32)),
    "Bit too high": ((2.44, 2.44 + 0.5), (7.32 / 3, 7.32 * 2 / 3)),
    "High and wide": ((2.44, 2.44 + 1), (-1, 0)),
    "Hits the bar": ((2.44 - 0.1, 2.44 + 0.1), (7.32 / 3, 7.32 * 2 / 3)),
    "Blocked": ((0, 2.44 / 2), (7.32 / 3, 7.32 * 2 / 3)),
}

# Function to draw a soccer goal and plot the shots
def plot_goal_with_clusters(data, shot_place_column, title, ax):
    # Draw the goal
    goal_width = 7.32  # Goal width in meters
    goal_height = 2.44  # Goal height in meters

    # Add a rectangle for the goal frame
    goal_frame = patches.Rectangle((0, 0), goal_width, goal_height, edgecolor='black', facecolor='none', lw=2)
    ax.add_patch(goal_frame)

    # Add the goal net (grid)
    for x in np.linspace(0, goal_width, 10):
        ax.plot([x, x], [0, goal_height], color='gray', linestyle='--', alpha=0.7)
    for y in np.linspace(0, goal_height, 5):
        ax.plot([0, goal_width], [y, y], color='gray', linestyle='--', alpha=0.7)

    # Plot the shots for each shot place
    for shot_place_value in data[shot_place_column].unique():
        # Convert numeric shot_place_value to corresponding string
        shot_place = shot_place_map.get(int(shot_place_value), "Unknown")  # Handle potential missing keys
        if shot_place in goal_regions:
            y_range, x_range = goal_regions[shot_place]
            print(f"Plotting shots for: {shot_place}")
            shot_data = data[data[shot_place_column] == shot_place_value]
            x_coords = np.random.uniform(x_range[0], x_range[1], len(shot_data))
            y_coords = np.random.uniform(y_range[0], y_range[1], len(shot_data))
            color = plt.cm.tab10(hash(shot_place) % 10)  # Ensure consistent color mapping
            ax.scatter(x_coords, y_coords, label=f"{shot_place}", alpha=0.7, edgecolor='k', s=50, color=color)

    ax.set_xlim(-1, goal_width + 1)
    ax.set_ylim(-1, goal_height + 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Width (meters)', fontsize=12)
    ax.set_ylabel('Height (meters)', fontsize=12)
    
# Load the dataset with predictions and real labels
data_path = 'dbscan_clustered_shots_with_predictions.csv'
shots_data = pd.read_csv(data_path)

# Create the figure and axes for the subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# Plot predicted shot labels
plot_goal_with_clusters(shots_data, shot_place_column='predicted_shot_place', title='Predicted Shot Places', ax=axes[0])

# Plot real shot labels
plot_goal_with_clusters(shots_data, shot_place_column='shot_place', title='Real Shot Places', ax=axes[1])

# Adjust the legend placement for better visualization
for ax in axes:
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), title='Shot Places', fontsize=10, ncol=3, frameon=True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()