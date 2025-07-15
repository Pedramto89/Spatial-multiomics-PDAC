import scanpy as sc
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load Scanpy AnnData object
adata = sc.read_h5ad("your_spatial_data.h5ad")

# Extract spatial coordinates and gene expression
spatial_data = adata.obsm["spatial"]  # Shape: (n_spots, 2)
expression_data = adata.X  # Shape: (n_spots, n_genes)

# Preprocess gene expression data
# Convert to dense if sparse
if hasattr(expression_data, "toarray"):
    expression_data = expression_data.toarray()

# Apply PCA to reduce dimensionality of expression data
n_pca_components = 50  # Adjust based on your data (e.g., 10–100)
pca = PCA(n_components=n_pca_components)
expression_pca = pca.fit_transform(expression_data)  # Shape: (n_spots, 50)
print(f"Explained variance ratio by PCA: {np.sum(pca.explained_variance_ratio_):.3f}")

# Normalize spatial coordinates to [0, 1]
scaler = MinMaxScaler()
spatial_data_normalized = scaler.fit_transform(spatial_data)

# Combine spatial and expression features per spot
n_channels = 2 + n_pca_components  # 2 (spatial) + 50 (expression) = 52
combined_features = np.hstack([spatial_data_normalized, expression_pca])  # Shape: (n_spots, 52)

# Reshape for CNN input
# Treat each spot as a 1x1 "patch" with C channels
spot_patches = combined_features[:, :, np.newaxis, np.newaxis]  # Shape: (n_spots, 52, 1, 1)
spot_patches = np.transpose(spot_patches, (0, 2, 3, 1))  # Shape: (n_spots, 1, 1, 52)

# Define CNN Encoder model
def create_spot_encoder(input_shape, latent_dim=64):
    inputs = layers.Input(shape=input_shape)  # (1, 1, 52)
    # Use Conv2D to process the "patch" (even if 1x1, for consistency with CNN framework)
    x = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation='relu', name='latent')(x)
    return models.Model(inputs, latent, name='spot_encoder')

# Define input shape
input_shape = (1, 1, n_channels)  # Each spot is a 1x1 patch with 52 channels

# Create encoder
encoder = create_spot_encoder(input_shape, latent_dim=64)

# Pretrain with a minimal autoencoder to learn latent representations
def create_spot_autoencoder(encoder):
    inputs = encoder.input
    latent = encoder.output
    x = layers.Dense(128, activation='relu')(latent)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(n_channels, activation='linear')(x)  # Reconstruct the input features
    x = layers.Reshape((1, 1, n_channels))(x)
    return models.Model(inputs, x, name='spot_autoencoder')

# Create and compile autoencoder
autoencoder = create_spot_autoencoder(encoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(
    spot_patches, spot_patches,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ],
    verbose=1
)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.show()

# Extract latent features for each spot
latent_per_spot = encoder.predict(spot_patches)  # Shape: (n_spots, 64)

# Cluster the latent features
num_clusters = 5
clusters = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(latent_per_spot)

# Add clusters to AnnData object
adata.obs["Spatial_Niches"] = clusters.astype(str)  # Ensure categorical for plotting

# Visualize spatial clustering
sc.pl.spatial(adata, color="Spatial_Niches", size=1.2, title="Spatial Niches")

# Save results
encoder.save('spot_encoder.h5')
np.save('latent_features.npy', latent_per_spot)
adata.write('adata_with_clusters.h5ad')
