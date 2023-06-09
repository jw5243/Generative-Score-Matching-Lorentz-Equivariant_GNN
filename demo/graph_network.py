from tensorflow.keras.layers import BatchNormalization, Layer, TimeDistributed
from tensorflow.keras.layers import Dense, Input, ReLU, Masking, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def generate_edges(num_nodes, batch_size = 1):
    # Returns an array of edge links corresponding to a fully-connected graph
    rows, cols = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = np.stack([rows, cols])
    edges = np.repeat(np.expand_dims(edges, axis = 0), batch_size, axis = 0)
    edges = np.transpose(edges, axes = [0, 2, 1])
    edge_attr = np.ones(len(edges[0]) * batch_size)  # Create 1D-tensor of 1s for each edge for a batch of graphs
    return edges, edge_attr


def create_ffn(hidden_units, dropout_rate, activation = tf.nn.gelu, name = None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation = activation))

    return tf.keras.Sequential(fnn_layers, name = name)


class GraphConvLayer(layers.Layer):
    def __init__(self,
                 hidden_units,
                 dropout_rate = 0.2,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        self.update_fn = create_ffn(hidden_units, dropout_rate)

    def reshape_time_embedding(self, time_embed, tensor):
        return tf.repeat(tf.expand_dims(time_embed, 1), tensor.shape[1], axis = 1)

    def prepare(self, node_representations, time_embed, weights = None):
        # node_repesentations shape is [num_edges, embedding_dim].
        time_embedding = self.reshape_time_embedding(time_embed, node_representations)
        messages = self.ffn_prepare(tf.concat([node_representations, time_embedding], axis = 2))
        # if weights is not None:
        #    messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbor_messages, node_representations):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        # node_repesentations shape is [num_nodes, representation_dim]
        num_nodes = node_representations.shape[1]

        # print(neighbor_messages.shape)

        transposed_messages = tf.transpose(neighbor_messages, (1, 2, 0))

        aggregated_message = tf.math.unsorted_segment_mean(transposed_messages, node_indices[0], num_segments = num_nodes)

        # print(aggregated_message.shape)

        return tf.transpose(aggregated_message, (2, 0, 1))
        # return aggregated_message

    def update(self, node_representations, aggregated_messages, time_embed):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        # Concatenate the node_repesentations and aggregated_messages.
        time_embedding = self.reshape_time_embedding(time_embed, node_representations)
        h = tf.concat([node_representations, aggregated_messages, time_embedding], axis = 2)
        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        node_representations, edges, edge_weights, time_embed = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbor_indices = edges[:, :, 0], edges[:, :, 1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbor_representations = tf.gather(node_representations, neighbor_indices, batch_dims = 1)

        # Prepare the messages of the neighbours.
        neighbor_messages = self.prepare(neighbor_representations, time_embed, edge_weights)
        # Aggregate the neighbour messages.

        aggregated_messages = self.aggregate(node_indices, neighbor_messages, node_representations)

        # Update the node embedding with the neighbour messages.
        return self.update(node_representations, aggregated_messages, time_embed)


class GNN(tf.keras.Model):
    def __init__(
            self,
            feature_dim,
            hidden_units,
            dropout_rate = 0.2,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.time_embed_mlp = create_ffn(hidden_units, dropout_rate, name = "time")
        # Create a process layer.
        self.preprocess = create_ffn(hidden_units, dropout_rate, name = "preprocess")
        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            name = "graph_conv1",
        )
        # Create the second GraphConv layer.
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            name = "graph_conv2",
        )
        # Create a postprocess layer.
        self.postprocess = create_ffn(hidden_units, dropout_rate, name = "postprocess")
        # Create a compute logits layer.
        self.compute_logits = layers.Dense(units = feature_dim, name = "logits")

    def call(self, input):
        p, time = input

        # Unpack graph_info to three elements: node_features, edges, and edge_weight.
        edges, edge_weights = generate_edges(p.shape[1], p.shape[0])

        time_embed = self.time_embed_mlp(time)
        #time_embed = tf.repeat(tf.expand_dims(time_embed, 1), p.shape[1], axis = 1)

        # Preprocess the node_features to produce node representations.
        x = self.preprocess(p)
        # Apply the first graph conv layer.
        x1 = self.conv1((x, edges, edge_weights, time_embed))
        # Skip connection.
        x = x1 + x
        # Apply the second graph conv layer.
        x2 = self.conv2((x, edges, edge_weights, time_embed))
        # Skip connection.
        x = x2 + x
        # Postprocess node embedding.
        x = self.postprocess(x)

        # Compute logits
        return self.compute_logits(x)


if __name__ == "__main__":
    batch = 100
    particle_count = 200
    particle_feature_dim = 4

    model = GNN(particle_feature_dim, [16, 16])

    permutation = np.random.permutation(particle_count)

    tf.random.set_seed(1111)
    p1 = tf.random.normal([batch, particle_count, particle_feature_dim])
    t1 = tf.random.uniform([batch, 1])
    model_output1 = model([p1, t1]).numpy()[:, permutation]

    tf.random.set_seed(1111)
    p2 = tf.random.normal([batch, particle_count, particle_feature_dim]).numpy()[:, permutation]
    t2 = tf.random.uniform([batch, 1])
    model_output2 = model([p2, t2])

    print(np.sqrt(tf.reduce_mean((model_output1 - model_output2) ** 2).numpy()))

