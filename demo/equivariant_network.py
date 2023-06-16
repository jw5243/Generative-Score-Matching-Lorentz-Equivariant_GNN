from tensorflow.keras.layers import BatchNormalization, Layer, TimeDistributed
from tensorflow.keras.layers import Dense, Input, ReLU, Masking, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


"""
Generates an edge adjacency matrix (fully connected) of shape [batch, num_nodes, 2] and corresponding edge attribute array with just ones
"""
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


"""
Computes the (special) relativistic norm x * x = -t^2 + x^2 + y^2 + z^2 or inner product x * y
"""
def minkowski_norm_squared(x, y = None):
    c = tf.constant([-1, 1, 1, 1], dtype = tf.float32)  # 64)
    h = c * (x * x) if y is None else c * (x * y)
    h = tf.expand_dims(tf.reduce_sum(h, axis = 2), axis = 2)
    return h


"""
Returns a keras network with layers in the order of batch_norm, dropout, and a dense layer with given activation
"""
def create_ffn(hidden_units, dropout_rate, activation = tf.nn.gelu, name = ""):
    fnn_layers = []

    for i in range(len(hidden_units)):
        fnn_layers.append(layers.BatchNormalization(name = name + "batch_norm_" + str(i)))
        fnn_layers.append(layers.Dropout(dropout_rate, name = name + "_dropout_" + str(i)))
        fnn_layers.append(layers.Dense(hidden_units[i], activation = activation, name = name + "_dense_" + str(i)))

    return tf.keras.Sequential(fnn_layers, name = name)


"""
Renormalizes input to network to avoid divergences in untrained network
"""
def normalize(x):
    return tf.math.sign(x) * tf.math.log(x + 1)


"""
Single graph convolution layer with Lorentz equivariance implemented via LorentzNet type structure
"""
class GraphConvLayer(layers.Layer):
    def __init__(self,
                 num_particles,
                 coordinate_hidden_units,
                 invariant_hidden_units,
                 dropout_rate = 0.2,
                 name = "",
                 final_layer = False,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.num_particles = num_particles
        self.edge_count = num_particles * (num_particles - 1.)
        self.message_mlp = create_ffn(invariant_hidden_units, dropout_rate, name = name + "_message")
        self.coordinate_mlp = create_ffn(coordinate_hidden_units, dropout_rate, name = name + "_coordinate")
        self.final_layer = final_layer
        if not final_layer:
            self.invariant_feature_mlp = create_ffn(invariant_hidden_units, dropout_rate,
                                                    name = name + "_invariant_feature")

        # Returns multiples of node and neighbor contributions to coordinate update (removes spacetime translation equivariance)
        self.a_mlp = layers.Dense(1, activation = tf.nn.gelu, use_bias = True)
        self.b_mlp = layers.Dense(1, activation = tf.nn.gelu, use_bias = True)

    # Adjusts time_embed shape to have the same shape as tensor
    def reshape_time_embedding(self, time_embed, tensor):
        return tf.repeat(tf.expand_dims(time_embed, 1), tensor.shape[1], axis = 1)

    # Computes aggregation of messages and coordinates via unsorted_segment_mean or sum
    def sum_neighbors(self, neighbor_values, segment_ids, num_particles, mean = False):
        # Must set first index to be the node index since unsorted_segment_* only applies to first index of tensor
        transposed_values = tf.transpose(neighbor_values, (1, 2, 0))
        if mean:
            aggregate_values = tf.math.unsorted_segment_mean(transposed_values, segment_ids,
                                                            num_segments = num_particles)
        else:
            aggregate_values = tf.math.unsorted_segment_sum(transposed_values, segment_ids,
                                                            num_segments = num_particles)
            
        # Apply inverse of permutation of the [batch, node, feature] -> [node, feature, batch] group element
        return tf.transpose(aggregate_values, (2, 0, 1))

    # Computes the messages of neighbors using the message_mlp
    def compute_messages(self, node_invariants, neighbor_invariants, invariants, time_embed, weights = None):
        # node_repesentations shape is [num_edges, embedding_dim].
        time_embedding = self.reshape_time_embedding(time_embed, neighbor_invariants)
        messages = self.message_mlp(
            tf.concat([node_invariants, neighbor_invariants, invariants, time_embedding], axis = 2))
        # if weights is not None:
        #    messages = messages * tf.expand_dims(weights, -1)
        return messages

    # Sums over the neighbor messages to yield an aggregated message which is used to update the invariant features
    def aggregate_messages(self, node_indices, neighbor_messages, node_representations):
        # neighbour_messages shape: [batch, num_edges, representation_dim].
        # node_repesentations shape is [batch, num_nodes, representation_dim]
        num_nodes = node_representations.shape[1]
        return self.sum_neighbors(neighbor_messages, node_indices[0], num_nodes)

    # Update the invariant features using only relativistic invariant quantities like aggregated messages and time_embedding
    def update_invariant_features(self, invariant_features, aggregated_messages, time_embed):
        # invariant_features shape is [batch, num_nodes, representation_dim].
        # aggregated_messages shape is [batch, num_nodes, representation_dim].
        time_embedding = self.reshape_time_embedding(time_embed, invariant_features)
        input = tf.concat([invariant_features, aggregated_messages, time_embedding], axis = 2)
        invariant_features_updated = self.invariant_feature_mlp(input)
        return invariant_features_updated

    # Update the coordinate features in a way that preserves the relativistic structure, 
    # by only taking linear combinations of four-vectors (which is still a four-vector
    def update_coordinates(self, coordinates, node_coordinates, neighbor_coordinates, messages, node_indices, time_embed):
        num_nodes = coordinates.shape[1]
        time_embedding = self.reshape_time_embedding(time_embed, messages)
        a = self.a_mlp(tf.concat([messages, time_embedding], axis = 2))
        b = self.b_mlp(tf.concat([messages, time_embedding], axis = 2))
        input = tf.concat([messages, time_embedding], axis = 2)
        coordinate_update = self.coordinate_mlp(input) * (a * node_coordinates + b * neighbor_coordinates)
        updated_sum = self.sum_neighbors(coordinate_update, node_indices[0], num_nodes, mean = True)
        return coordinates + updated_sum

    # Calculate relativistic invariant quantities using the four-vectors of the node and neighbor (x*x and x*y)
    # We apply normalization after so that the network does not diverge
    def compute_invariants(self, coordinates, neighbor_coordinates, node_indices):
        node_coordinates = tf.gather(coordinates, node_indices, batch_dims = 1)
        inner_product = normalize(tf.math.abs(minkowski_norm_squared(node_coordinates, neighbor_coordinates)))
        squared_difference = normalize(tf.math.abs(minkowski_norm_squared(node_coordinates - neighbor_coordinates)))
        return tf.concat([inner_product, squared_difference], axis = 2)
        #return tf.concat([node_coordinates, neighbor_coordinates], axis = 2)

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: h, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        x, h, edges, edge_weights, time_embed = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbor_indices = edges[:, :, 0], edges[:, :, 1]
        # node_invariants shape is [batch, num_edges, representation_dim]
        # neighbour_invariants shape is [batch, num_edges, representation_dim].
        node_invariants = tf.gather(h, node_indices, batch_dims = 1)
        neighbor_invariants = tf.gather(h, neighbor_indices, batch_dims = 1)
        
        # node_coordinates shape is [batch, num_edges, coordinate_dim = 4]
        # neighbor_coordinates shape is [batch, num_eddges, coordinate_dim = 4]
        node_coordinates = tf.gather(x, node_indices, batch_dims = 1)
        neighbor_coordinates = tf.gather(x, neighbor_indices, batch_dims = 1)

        # Compute invariants from the node and neighbor coordinates
        invariants = self.compute_invariants(x, neighbor_coordinates, node_indices)

        # Prepare the messages of the neighbours.
        neighbor_messages = self.compute_messages(node_invariants, neighbor_invariants, invariants, time_embed,
                                                  edge_weights)
        # Aggregate the neighbour messages.

        aggregated_messages = self.aggregate_messages(node_indices, neighbor_messages, h)

        x_updated = self.update_coordinates(x, node_coordinates, neighbor_coordinates, neighbor_messages, node_indices, time_embed)
        # Do not update invariant features h in the last layer since the graph output is based on coordinates, 
        # not invariants, so the last layer invariants are not included in cost function
        if not self.final_layer:
            h_updated = self.update_invariant_features(h, aggregated_messages, time_embed)
            return x_updated, h_updated

        # Update the node embedding with the neighbour messages.
        return x_updated, None


"""
Combines multiple Lorentz equivariant convolution layers together to form the LEGNN network used for generative score-matching
"""
class LEGNN(tf.keras.Model):
    def __init__(
            self,
            num_particles,
            feature_dim,
            coordinate_hidden_units,
            invariant_hidden_units,
            num_layers = 4,
            dropout_rate = 0.2,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.time_embed_mlp = create_ffn(invariant_hidden_units, dropout_rate, name = "time")
        # Create a process layer.
        self.preprocess = create_ffn(invariant_hidden_units, dropout_rate, name = "preprocess")
        self.network_layers = []
        for i in range(num_layers):
            self.network_layers.append(GraphConvLayer(
                num_particles,
                coordinate_hidden_units,
                invariant_hidden_units,
                dropout_rate,
                name = "graph_conv" + str(i),
                final_layer = (i == num_layers - 1)
            ))

    def call(self, input):
        x, time = input

        # Unpack graph_info to three elements: node_features, edges, and edge_weight.
        edges, edge_weights = generate_edges(x.shape[1], x.shape[0])
        time_embed = self.time_embed_mlp(time)

        # Initialize h with invariants so that h maintains Lorentz invariance throughout network
        h = normalize(tf.math.abs(minkowski_norm_squared(x)))
        # Can apply h to any network since h only contains invariant quantities
        h = self.preprocess(h)
        for layer in self.network_layers:
            x, h = layer((x, h, edges, edge_weights, time_embed))

        return x


if __name__ == "__main__":
    batch = 1
    particle_count = 3
    particle_feature_dim = 4

    model = LEGNN(particle_feature_dim, [1, 1], [16, 16], num_layers = 1)

    p = tf.random.normal([1, particle_count, particle_feature_dim])
    p = tf.repeat(p, batch, axis = 0)
    t = tf.random.uniform([1, 1])
    t = tf.repeat(t, batch, axis = 0)
    # print(model([p, t]).numpy())
    print(tf.math.reduce_std(model([p, t]), axis = 0))

    """permutation = np.random.permutation(particle_count)

    tf.random.set_seed(1111)
    p1 = tf.random.normal([batch, particle_count, particle_feature_dim])
    t1 = tf.random.uniform([batch, 1])
    model_output1 = model([p1, t1]).numpy()[:, permutation]

    tf.random.set_seed(1111)
    p2 = tf.random.normal([batch, particle_count, particle_feature_dim]).numpy()[:, permutation]
    t2 = tf.random.uniform([batch, 1])
    model_output2 = model([p2, t2])

    print(np.sqrt(tf.reduce_mean((model_output1 - model_output2) ** 2).numpy()))"""

