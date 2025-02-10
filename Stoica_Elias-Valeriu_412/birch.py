import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from prettytable import PrettyTable
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score, davies_bouldin_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from tqdm import tqdm
from itertools import permutations
from sklearn.metrics import silhouette_samples

# Load data
events = pd.read_csv('../data/events.csv')
games_info = pd.read_csv('../data/ginf.csv')

# Ensure all fields are valid
valid_fields = [
    'id_odsp', 'id_event', 'sort_order', 'time', 'text', 'event_type', 'event_type2', 
    'side', 'event_team', 'opponent', 'player', 'player2', 'player_in', 'player_out', 
    'shot_place', 'shot_outcome', 'is_goal', 'location', 'bodypart', 'assist_method', 
    'situation', 'fast_break'
]

# Select a random line from the events dataframe
random_event = events.sample(n=1).iloc[0]

# Create a PrettyTable object
table = PrettyTable()
table.field_names = ["Field", "Value"]

# Add rows to the table, excluding NaN values
for field, value in random_event.items():
    if field in valid_fields and pd.notna(value):
        table.add_row([field, value])

# Print the table
print(table)

# Plot the table using matplotlib
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('tight')
ax.axis('off')
table_data = [[field, value] for field, value in random_event.items() if field in valid_fields and pd.notna(value)]
ax.table(cellText=table_data, colLabels=["Field", "Value"], cellLoc='center', loc='center')

plt.show()

# Ensure all fields are valid
valid_fields = [
    'id_odsp', 'link_odsp', 'adv_stats', 'date', 'league', 'season', 'country', 
    'ht', 'at', 'fthg', 'ftag', 'odd_h', 'odd_d', 'odd_a', 'odd_over', 'odd_under', 
    'odd_bts', 'odd_bts_n'
]

# Select a random line from the games_info dataframe
random_game = games_info.sample(n=1).iloc[0]

# Create a PrettyTable object
table = PrettyTable()
table.field_names = ["Field", "Value"]

# Add rows to the table, excluding NaN values
for field, value in random_game.items():
    if field in valid_fields and pd.notna(value):
        table.add_row([field, value])

# Print the table
print(table)

# Plot the table using matplotlib
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('tight')
ax.axis('off')
table_data = [[field, value] for field, value in random_game.items() if field in valid_fields and pd.notna(value)]
ax.table(cellText=table_data, colLabels=["Field", "Value"], cellLoc='center', loc='center')

plt.show()

# Filter relevant columns
relevant_columns = [
    'event_team',
    'event_type',
    'event_type2',
    'location',
    'assist_method',
    'situation',
    'bodypart',
    'shot_place',
    'shot_outcome'
]

events = events[relevant_columns]

# Drop rows with missing critical values
critical_columns = ['event_team', 'event_type']
events = events.dropna(subset=critical_columns)

# One-hot encode categorical columns
categorical_columns = [
    'event_type',
    'event_type2',
    'location',
    'assist_method',
    'situation',
    'bodypart',
    'shot_place',
    'shot_outcome'
]

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(events[categorical_columns])

# Create a DataFrame with the encoded features
encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
encoded_features = pd.DataFrame(encoded, columns=encoded_feature_names)

# Combine encoded features with the team column
data_encoded = pd.concat([events['event_team'].reset_index(drop=True), encoded_features], axis=1)

# Group by team
team_aggregated = data_encoded.groupby('event_team').sum().reset_index()

# Drop "Not recorded" or "nan" if it exists in new columns 
columns_to_drop = [
    col for col in team_aggregated.columns 
    if any(substr in col for substr in ['nan', 'Not recorded']) or col.endswith('19.0') 
]
team_aggregated.drop(columns=columns_to_drop, errors='ignore', inplace=True)

# Count games played for each team
# Combine home and away counts
games_played = games_info['ht'].value_counts() + games_info['at'].value_counts()
team_aggregated['games_played'] = team_aggregated['event_team'].map(games_played).fillna(1)  # Avoid division by zero

# Normalize event-related columns by games played
columns_to_normalize = [
    col for col in team_aggregated.columns 
    if col not in ['event_team', 'games_played']
]

for col in columns_to_normalize:
    team_aggregated[col] = team_aggregated[col] / team_aggregated['games_played']
    
# Create additional aggregate statistics / ratios
# Example: total attempts, total fouls, total red cards, etc.

event_type_cols       = [c for c in columns_to_normalize if c.startswith('event_type_')]
shot_outcome_cols     = [c for c in columns_to_normalize if c.startswith('shot_outcome_')]
shot_place_cols       = [c for c in columns_to_normalize if c.startswith('shot_place_')]
card_cols             = [c for c in event_type_cols if any(x in c for x in ['event_type_4', 'event_type_5', 'event_type_6'])]
foul_cols             = [c for c in event_type_cols if 'event_type_3' in c]
card_cols

# Total event_types sum
team_aggregated['total_events'] = team_aggregated[event_type_cols].sum(axis=1)

# Shots on target vs off target
team_aggregated['total_shots_on_target'] = team_aggregated.get('shot_outcome_1.0', 0.0)  # 1=On target
team_aggregated['total_shots_off_target'] = team_aggregated.get('shot_outcome_2.0', 0.0) # 2=Off target
team_aggregated['total_shots_blocked'] = team_aggregated.get('shot_outcome_3.0', 0.0)    # 3=Blocked
team_aggregated['total_shots_hit_the_bar'] = team_aggregated.get('shot_outcome_4.0', 0.0)  # 4=Hit the bar

# On-target shots vs total shots
total_shots = team_aggregated['total_shots_on_target'] + team_aggregated['total_shots_off_target'] + team_aggregated['total_shots_blocked'] + team_aggregated['total_shots_hit_the_bar']

team_aggregated['ratio_shots_on_target'] = np.where(
    total_shots > 0, 
    team_aggregated['total_shots_on_target'] / total_shots, 
    0
)

# Total cards (yellow, second yellow and red) vs total fouls
team_aggregated['total_cards'] = team_aggregated[card_cols].sum(axis=1) if card_cols else 0
team_aggregated['total_fouls'] = team_aggregated[foul_cols].sum(axis=1) if foul_cols else 0
team_aggregated['ratio_cards_per_foul'] = np.where(
    team_aggregated['total_fouls'] > 0,
    team_aggregated['total_cards'] / team_aggregated['total_fouls'],
    0
)

# Add selected interaction terms
event_type_columns = [col for col in team_aggregated.columns if col.startswith('event_type_')]
location_columns = [col for col in team_aggregated.columns if col.startswith('location_')]
assist_method_columns = [col for col in team_aggregated.columns if col.startswith('assist_method_')]
bodypart_columns = [col for col in team_aggregated.columns if col.startswith('bodypart_')]
shot_outcome_columns = [col for col in team_aggregated.columns if col.startswith('shot_outcome_')]
shot_place_columns = [col for col in team_aggregated.columns if col.startswith('shot_place_')]
situation_columns = [col for col in team_aggregated.columns if col.startswith('situation_')]

interaction_pairs = [
    (event_type_columns, location_columns),       # Event Type x Location
    (event_type_columns, assist_method_columns),  # Event Type x Assist Method
    (location_columns, bodypart_columns),         # Location x Body Part
    (location_columns, assist_method_columns),    # Location x Assist Method
    (event_type_columns, shot_outcome_columns),   # Event Type x Shot Outcome
    (shot_outcome_columns, shot_place_columns),   # Shot Outcome x Shot Place
    (event_type_columns, situation_columns),      # Event Type x Situation
]

for category1, category2 in interaction_pairs:
    for col1, col2 in product(category1, category2):
        team_aggregated[f'{col1}_x_{col2}'] = team_aggregated[col1] * team_aggregated[col2]
        
# Drop games_played (no longer needed for clustering features)
team_aggregated.drop(columns=['games_played'], inplace=True)

# Removing unnecessary features
variances = team_aggregated.drop(columns=['event_team']).var(axis=0)
low_variance_threshold = 0.001  # Define a threshold for low variance
low_variance_features = variances[variances < low_variance_threshold].index.tolist()

print(f"Low Variance Features (threshold={low_variance_threshold}): {len(low_variance_features)}")
team_aggregated.drop(columns=low_variance_features, inplace=True, errors='ignore')

# Correlation Analysis
# Compute pairwise correlations
correlation_matrix = team_aggregated.drop(columns=['event_team']).corr()
correlation_threshold = 0.9  # High correlation threshold

# Plot the correlation heatmap for visual analysis
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Convert correlation matrix to distance matrix (1 - abs(correlation))
distance_matrix = 1 - abs(correlation_matrix)

# Perform hierarchical clustering on the distance matrix
linkage_matrix = linkage(squareform(distance_matrix), method='average')

# Form clusters based on the correlation threshold
cluster_labels = fcluster(linkage_matrix, t=1 - correlation_threshold, criterion='distance')

# Group features into clusters
clusters = {}
for feature, cluster in zip(correlation_matrix.columns, cluster_labels):
    clusters.setdefault(cluster, []).append(feature)
    
# Retain the feature with the highest variance within each cluster
features_to_keep = []
for cluster, features in clusters.items():
    if len(features) > 1:
        # Retain the feature with the highest variance
        variances = team_aggregated[features].var()
        best_feature = variances.idxmax()
        features_to_keep.append(best_feature)
    else:
        # If only one feature in the cluster, retain it
        features_to_keep.append(features[0])
        
# Drop all other features not in `features_to_keep`
features_to_drop = [col for col in team_aggregated.columns if col not in features_to_keep and col != 'event_team']
print(f"Features retained after clustering: {len(features_to_keep)} (from {len(correlation_matrix.columns)})")
print(f"Features dropped: {len(features_to_drop)}")
team_aggregated = team_aggregated[['event_team'] + features_to_keep]

# StandardScaler for feature scaling
scaler = StandardScaler()
features_to_scale = [col for col in team_aggregated.columns if col != 'event_team']
scaled_features = scaler.fit_transform(team_aggregated[features_to_scale])

# Create final scaled DataFrame
final_team_cluster_data = pd.DataFrame(scaled_features, columns=features_to_scale)
final_team_cluster_data['event_team'] = team_aggregated['event_team']

# Save processed data for clustering
output_file = 'data/final_team_cluster_data2.csv'
final_team_cluster_data.to_csv(output_file, index=False)
print(f"Processed and normalized dataset saved to {output_file}")

# Clustering teams based on their playstyle and league

# Load the processed data
file_path = '../data/final_team_cluster_data2.csv'
final_team_cluster_data = pd.read_csv(file_path)

# Add league information from the matches dataset
matches_file_path = '../data/ginf.csv'
matches_data = pd.read_csv(matches_file_path)
team_league_map = pd.concat([matches_data[['ht', 'league']], matches_data[['at', 'league']].rename(columns={'at': 'ht'})])
team_league_map = team_league_map.drop_duplicates().set_index('ht')['league'].to_dict()
final_team_cluster_data['league'] = final_team_cluster_data['event_team'].map(team_league_map)

# Verify league column
print("\nNumber of Teams from Each League:")
print(final_team_cluster_data['league'].value_counts())

# Prepare data for clustering
team_names = final_team_cluster_data['event_team']
league_labels = final_team_cluster_data['league']  # Actual league labels
clustering_features = final_team_cluster_data.drop(columns=['event_team', 'league'])

# Apply RandomOverSampler
print("\nBefore Oversampling:")
print(final_team_cluster_data['league'].value_counts())

# Use oversampling to balance the dataset based on league labels
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(clustering_features, league_labels)
print("\nAfter Oversampling:")
print(pd.Series(y_resampled).value_counts())

# Create a new DataFrame with oversampled data
oversampled_data = pd.DataFrame(X_resampled, columns=clustering_features.columns)
oversampled_data['league'] = y_resampled

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(oversampled_data.drop(columns=['league']))
oversampled_data_scaled = pd.DataFrame(scaled_features, columns=clustering_features.columns)
oversampled_data_scaled['league'] = y_resampled

# Apply BIRCH clustering
best_model = None
best_metrics = {'score': -np.inf}
threshold_values = np.linspace(0.1, 0.9, 9)
n_clusters = 5  # Fixed to 5 for league clustering
weights = {
    'silhouette': 0.4,
    'davies_bouldin': -0.3,
    'ari': 0.15,
    'homogeneity': 0.1,
    'completeness': 0.05
}

for threshold in tqdm(threshold_values, desc="Tuning BIRCH"):
    birch_model = Birch(n_clusters=n_clusters, threshold=threshold)
    birch_model.fit(oversampled_data_scaled.drop(columns=['league']))
    labels = birch_model.labels_

    if len(set(labels)) > 1:
        silhouette = silhouette_score(oversampled_data_scaled.drop(columns=['league']), labels)
        dbi = davies_bouldin_score(oversampled_data_scaled.drop(columns=['league']), labels)
        ari = adjusted_rand_score(y_resampled, labels)
        homogeneity = homogeneity_score(y_resampled, labels)
        completeness = completeness_score(y_resampled, labels)

        weighted_score = (
            weights['silhouette'] * silhouette +
            weights['davies_bouldin'] * (1 / (1 + dbi)) +
            weights['ari'] * ari +
            weights['homogeneity'] * homogeneity +
            weights['completeness'] * completeness
        )

        if weighted_score > best_metrics['score']:
            best_metrics = {
                'score': weighted_score,
                'silhouette': silhouette,
                'davies_bouldin': dbi,
                'ari': ari,
                'homogeneity': homogeneity,
                'completeness': completeness,
                'threshold': threshold
            }
            best_model = birch_model
            
# Print the best hyperparameters and metrics
print("\nBest BIRCH Model:")
for metric, value in best_metrics.items():
    if metric != 'score':
        print(f"{metric.capitalize()}: {value:.2f}")
print(f"Weighted Score: {best_metrics['score']:.2f}")

# Assign clusters
oversampled_data_scaled['cluster'] = best_model.labels_

# Predict leagues for each cluster
cluster_league_df = oversampled_data_scaled[['cluster', 'league']]

# Calculate league proportions within each cluster
league_distribution = cluster_league_df.groupby(['cluster', 'league']).size().unstack(fill_value=0)
league_distribution_normalized = league_distribution.div(league_distribution.sum(axis=1), axis=0)

# Actual league proportions
real_league_counts = oversampled_data_scaled['league'].value_counts()
real_league_proportions = real_league_counts / real_league_counts.sum()

# Generate all possible mappings between clusters and leagues
possible_mappings = list(permutations(real_league_counts.index, len(league_distribution)))

def calculate_fit_score(mapping, league_distribution, real_league_proportions):
    score = 0
    for cluster, league in enumerate(mapping):
        cluster_league_proportion = league_distribution.iloc[cluster].get(league, 0)
        league_proportion = real_league_proportions.get(league, 0)
        score += cluster_league_proportion * league_proportion
    return score

# Find the best mapping
best_mapping = None
best_score = -np.inf
for mapping in possible_mappings:
    score = calculate_fit_score(mapping, league_distribution, real_league_proportions)
    if score > best_score:
        best_score = score
        best_mapping = mapping
        
# Print the best mapping
cluster_to_league_mapping = {cluster: league for cluster, league in enumerate(best_mapping)}
print("\nBest Cluster-to-League Mapping:")
for cluster, league in cluster_to_league_mapping.items():
    print(f"Cluster {cluster}: {league}")
    
# Assign predicted leagues
oversampled_data_scaled['predicted_league'] = oversampled_data_scaled['cluster'].map(cluster_to_league_mapping)

# Compute Accuracy
accuracy = (oversampled_data_scaled['predicted_league'] == oversampled_data_scaled['league']).mean()
print(f"\nClustering Accuracy: {accuracy:.2f}")

# Visualize Clusters with PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(oversampled_data_scaled.drop(columns=['league', 'cluster', 'predicted_league']))
oversampled_data_scaled['pca_x'] = pca_features[:, 0]
oversampled_data_scaled['pca_y'] = pca_features[:, 1]

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    oversampled_data_scaled['pca_x'], 
    oversampled_data_scaled['pca_y'], 
    c=oversampled_data_scaled['cluster'], 
    cmap='rainbow', 
    alpha=0.7
)
plt.colorbar(scatter, label='Cluster')
plt.title(f'BIRCH Clustering with Oversampling (Threshold={best_metrics["threshold"]:.2f})', fontsize=16)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)

# Annotate team names
for i, team in enumerate(team_names):
    plt.annotate(
        team,
        (oversampled_data_scaled['pca_x'][i], oversampled_data_scaled['pca_y'][i]),
        fontsize=8,
        alpha=0.7,
        textcoords="offset points",
        xytext=(5, 5),
        ha='center'
    )
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# For curiosity, print final league distribution
print("\nNumber of Teams from Each League:")
print(oversampled_data_scaled['league'].value_counts())

# Plot heatmap for cluster-to-league proportions
plt.figure(figsize=(10, 6))
sns.heatmap(
    league_distribution_normalized,
    annot=True,
    cmap='YlGnBu',
    fmt=".2f",
    cbar=True,
    linewidths=0.5
)
plt.title('Cluster-to-League Proportions Heatmap', fontsize=16)
plt.xlabel('League', fontsize=12)
plt.ylabel('Cluster', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate silhouette scores for each point
silhouette_vals = silhouette_samples(oversampled_data_scaled.drop(columns=['league', 'cluster', 'predicted_league', 'pca_x', 'pca_y']), oversampled_data_scaled['cluster'])

# Plot silhouette scores
plt.figure(figsize=(10, 6))
y_lower = 10
for i in range(n_clusters):
    ith_cluster_silhouette_vals = silhouette_vals[oversampled_data_scaled['cluster'] == i]
    ith_cluster_silhouette_vals.sort()
    size_cluster_i = ith_cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_vals, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Cluster {i}")
    y_lower = y_upper + 10

plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
plt.title('Silhouette Scores for Clusters', fontsize=16)
plt.xlabel('Silhouette Coefficient', fontsize=12)
plt.ylabel('Samples', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()