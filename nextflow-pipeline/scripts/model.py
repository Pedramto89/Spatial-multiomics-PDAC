#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers, Model

class SimpleGraphConvLayer(tf.keras.layers.Layer):
    """
    Simple graph convolutional layer for spatial transcriptomics.
    Performs basic message passing on spatial graphs.
    """
    def __init__(self, units, dropout_rate=0.2, **kwargs):
        super(SimpleGraphConvLayer, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # input_shape[0] is feature shape, input_shape[1] is adjacency shape
        input_dim = input_shape[0][-1]
        
        self.dense = tf.keras.layers.Dense(self.units, activation='relu')
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.built = True
        
    def call(self, inputs, training=None):
        x, adj = inputs
        
        # Apply dense transformation
        h = self.dense(x)
        
        # Graph convolution: multiply by adjacency matrix
        # Note: This is a simplified version - real GCN would normalize adjacency
        output = tf.matmul(adj, h)
        
        return self.dropout(output, training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate
        })
        return config

def create_simple_gnn(gene_dim, img_dim, num_classes):
    """
    Create a simple GNN model for spatial transcriptomics classification.
    This version matches the architecture used in gnn_training.py.
    
    Args:
        gene_dim (int): Dimension of gene expression features
        img_dim (int): Dimension of image features  
        num_classes (int): Number of classification classes
        
    Returns:
        tf.keras.Model: Compiled GNN model
    """
    # Input layers
    gene_input = layers.Input(shape=(gene_dim,), name='gene_input')
    img_input = layers.Input(shape=(img_dim,), name='img_input')
    
    # Gene expression processing
    gene_dense = layers.Dense(256, activation='relu')(gene_input)
    gene_dense = layers.Dropout(0.3)(gene_dense)
    gene_dense = layers.Dense(128, activation='relu')(gene_dense)
    
    # Image feature processing
    img_dense = layers.Dense(128, activation='relu')(img_input)
    img_dense = layers.Dropout(0.3)(img_dense)
    
    # Combine features
    combined = layers.Concatenate()([gene_dense, img_dense])
    combined = layers.Dense(256, activation='relu')(combined)
    combined = layers.Dropout(0.4)(combined)
    combined = layers.Dense(128, activation='relu')(combined)
    
    # Output layer
    output = layers.Dense(num_classes, activation='softmax', name='output')(combined)
    
    model = Model(inputs=[gene_input, img_input], outputs=output)
    return model

def create_graph_aware_gnn(gene_dim, img_dim, num_classes):
    """
    Create a graph-aware GNN model that uses adjacency matrices.
    This version includes the graph convolutional layers.
    
    Args:
        gene_dim (int): Dimension of gene expression features
        img_dim (int): Dimension of image features
        num_classes (int): Number of classification classes
        
    Returns:
        tf.keras.Model: Compiled graph-aware GNN model
    """
    # Input layers
    gene_input = layers.Input(shape=(gene_dim,), name="gene_features")
    img_input = layers.Input(shape=(img_dim,), name="image_features")
    adj_input = layers.Input(shape=(None, None), name="adjacency_matrix", dtype=tf.float32)
    
    # Feature processing
    gene_features = layers.Dense(128, activation='relu')(gene_input)
    gene_features = layers.Dropout(0.3)(gene_features)
    
    img_features = layers.Dense(128, activation='relu')(img_input)
    img_features = layers.Dropout(0.3)(img_features)
    
    # Combine features
    combined = layers.Concatenate()([gene_features, img_features])
    
    # Graph convolutional layers
    x = SimpleGraphConvLayer(256, name='simple_graph_conv_layer_1')([combined, adj_input])
    x = SimpleGraphConvLayer(128, name='simple_graph_conv_layer_2')([x, adj_input])
    latent = SimpleGraphConvLayer(64, name='simple_graph_conv_layer_3')([x, adj_input])
    
    # Classification output
    class_output = layers.Dense(num_classes, activation='softmax', name='classification')(latent)
    
    model = Model(inputs=[gene_input, img_input, adj_input], outputs=class_output)
    
    return model

def compile_model(model, learning_rate=1e-4):
    """
    Compile the model with appropriate optimizer and loss function.
    
    Args:
        model: TensorFlow/Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Example usage:
if __name__ == "__main__":
    # Example of creating and compiling models
    
    # Simple model (currently used in pipeline)
    simple_model = create_simple_gnn(gene_dim=1000, img_dim=128, num_classes=4)
    simple_model = compile_model(simple_model)
    
    print("Simple GNN Model:")
    simple_model.summary()
    
    print("\n" + "="*50 + "\n")
    
    # Graph-aware model (for future use)
    graph_model = create_graph_aware_gnn(gene_dim=1000, img_dim=128, num_classes=4)
    graph_model = compile_model(graph_model)
    
    print("Graph-aware GNN Model:")
    graph_model.summary()
