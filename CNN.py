import scanpy as sc
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Load Scanpy AnnData object
adata = sc.read_h5ad("your_spatial_data.h5ad")

# Extract gene expression (normalized, log-transformed)
gene_data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
gene_data = StandardScaler().fit_transform(gene_data)  # Normalize

# Extract spatial coordinates
spatial_data = adata.obsm["spatial"]
spatial_data = StandardScaler().fit_transform(spatial_data)  # Normalize

# Load histology image and resize
image = Image.open("your_tissue_image.png").resize((224, 224))
image_array = np.array(image) / 255.0  # Normalize pixel values

# Split into training and validation sets
from sklearn.model_selection import train_test_split

X_gene_train, X_gene_val, X_spatial_train, X_spatial_val = train_test_split(
    gene_data, spatial_data, test_size=0.2, random_state=42
)

# Expand dimensions for CNN input (batch, height, width, channels)
X_image_train = np.expand_dims(image_array, axis=0)
X_image_val = np.expand_dims(image_array, axis=0)

# Define labels (optional: niche clusters if available)
y_train = np.random.randint(0, 3, size=(X_gene_train.shape[0],))  # Dummy labels
y_val = np.random.randint(0, 3, size=(X_gene_val.shape[0],))

# Convert labels to categorical (if classification)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# Compile & Train Model
fusion_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
fusion_model.fit(
    [X_image_train, X_gene_train, X_spatial_train],
    y_train,
    validation_data=([X_image_val, X_gene_val, X_spatial_val], y_val),
    epochs=20,
    batch_size=16
)

# Extract latent features
latent_features = fusion_model.predict([X_image_train, X_gene_train, X_spatial_train])

# Apply clustering (e.g., KMeans, Leiden)
from sklearn.cluster import KMeans

num_clusters = 5
clusters = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(latent_features)

# Add clusters to AnnData
adata.obs["CNN_Niches"] = clusters
sc.pl.spatial(adata, color="CNN_Niches", size=1.2)
