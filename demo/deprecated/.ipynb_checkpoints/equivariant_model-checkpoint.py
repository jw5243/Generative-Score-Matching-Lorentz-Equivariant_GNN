from tensorflow.keras.layers import BatchNormalization, Layer, TimeDistributed
from tensorflow.keras.layers import Dense, Input, ReLU, Masking,Conv1D
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
            if True:#i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    edges = [np.array(edges[0], dtype = int), np.array(edges[1], dtype = int)]  # Convert 2D array of edge links to a 2D-tensor
    rows, cols = [], []
    for i in range(batch_size):
        rows.append(edges[0]) #+ num_nodes * i)  # Offset rows for each graph in the batch
        cols.append(edges[1]) #+ num_nodes * i)

    edges = np.stack([rows, cols])
    
    adj_matrix = np.ones([batch_size, num_nodes, num_nodes]) - np.array([np.eye(num_nodes) for _ in range(batch_size)])
    edge_attr = np.ones(len(edges[0]) * batch_size)
    return edges, edge_attr, adj_matrix


class L_GCL(Model):
    """
    SO+(1, 3) Equivariant Convolution Layer
    """

    def __init__(self, input_feature_dim, message_dim, output_feature_dim, edge_feature_dim, projection_dim = 32, activation = layers.LeakyReLU(alpha=0.01), name = "", final_layer = False):
        """
        Sets up the MLPs needed to compute the layer update of the equivariant network.
        :param input_feature_dim: The amount of numbers needed to specify a feature inputted into the GCL
        :param message_dim: The amount of numbers needed to specify a message passed through the GCL
        :param output_feature_dim: The amount of numbers needed to specify the updated feature after passing through the GCL
        :param edge_feature_dim: The amount of numbers needed to specify an edge attribute a_{ij}
        :param activation: The activation function used as the main non-linearity throughout the GCL
        """

        super(L_GCL, self).__init__()
        self.final_layer = final_layer
        radial_dim = 1  # Only one number is needed to specify Minkowski distance
        coordinate_dim = 4
        
        """self.time_mlp = tf.keras.Sequential([
            layers.Dense(2 * projection_dim, activation = activation, name = name + "_time_1"),
            layers.Dense(projection_dim, name = name + "_time_2")
        ], name = "time")"""

        # The MLP used to calculate messages
        self.edge_mlp = tf.keras.Sequential([
            layers.Dense(message_dim, activation = activation, name = name + "_edge_1"),
            #layers.BatchNormalization(epsilon=1e-6),
            layers.Dense(message_dim, activation = activation, name = name + "_edge_2")
        ], name = "edge")

        # The MLP used to update the feature vectors h_i
        if not final_layer:
            self.feature_mlp = tf.keras.Sequential([
                layers.Dense(message_dim, activation = activation, name = name + "_feature_1"),
                #layers.BatchNormalization(epsilon=1e-6),
                layers.Dense(output_feature_dim, activation = activation, name = name + "_feature_2")
            ], name = "feature")

        # The MLP used to update coordinates (node embeddings) x_i
        self.coordinate_mlp = tf.keras.Sequential([
            layers.Dense(message_dim, activation = activation, name = name + "_coordinate_1"),
            #layers.BatchNormalization(epsilon=1e-6),
            layers.Dense(1, activation = activation, name = name + "_coordinate_2")
        ], name = "coordinate")

        self.self_multiplier = tf.Variable(1, trainable = True, dtype = tf.float32, name = "self")
        self.other_multiplier = tf.Variable(1, trainable = True, dtype = tf.float32, name = "other")
    
    def get_sources_and_targets(self, adj_matrix, x):
        adj_extended = tf.expand_dims(adj_matrix, axis = 3)
        
        x_extended = tf.expand_dims(x, axis = 2)
        x_extended = tf.repeat(x_extended, repeats = x.shape[1], axis = 2)
        
        x_extended_flipped = tf.transpose(x_extended, perm = [0, 2, 1, 3])
        
        sources = tf.reshape(tf.multiply(adj_extended, x_extended), shape = (x.shape[0], x.shape[1] ** 2, x.shape[2]))
        targets = tf.reshape(tf.multiply(adj_extended, x_extended_flipped), shape = (x.shape[0], x.shape[1] ** 2, x.shape[2]))
        return sources, targets

    def compute_messages(self, source, target, radial, edge_attribute = None):
        """
        Calculates the messages to send between two nodes 'target' and 'source' to be passed through the network.
        The message is computed via an MLP of Lorentz invariants.
        :param source: The source node's feature vector h_i
        :param target: The target node's feature vector h_j
        :param radial: The Minkowski distance between the source and target's coordinates
        :param edge_attribute: Features at the edge connecting the source and target nodes
        :return: The message m_{ij}
        """
        if edge_attribute is None:
            message_inputs = tf.concat([source, target, radial], axis = 2)
        else:
            message_inputs = tf.concat([source, target, radial], axis = 2)
            #message_inputs = tf.concat([source, target, radial, time_embed, edge_attribute], axis = 2)
            
        out = self.edge_mlp(message_inputs)  # Apply \phi_e to calculate the messages
        return out

    def update_feature_vectors(self, h, edge_index, messages, time_embed):
        """
        Updates the feature vectors via an MLP of Lorentz invariants, specifically the feature vector itself and
        aggregated messages.
        :param h: The feature vectors outputted from the previous layer
        :param edge_index: Array containing the connection between nodes
        :param messages: List of messages m_{ij} used to calculated an aggregated message for h
        :return: The updated feature vectors h_i^{l+1}
        """
        
        row = edge_index[0]
        col = edge_index[1]
        
        message_aggregate = tf.math.unsorted_segment_sum(messages, row, num_segments = h.shape[0])
        message_aggregate = tf.repeat(tf.expand_dims(message_aggregate, 1), h.shape[1], axis = 1)
        time_embed = tf.repeat(tf.expand_dims(time_embed, 1), h.shape[1], axis = 1)
        feature_inputs = tf.concat([h, message_aggregate, time_embed], axis = 2)
        out = self.feature_mlp(feature_inputs)
        return out, message_aggregate

    def update_coordinates(self, x, edge_index, adj_matrix, coordinate_difference, messages, time_embed):
        """
        Updates the coordinates (node embeddings) through the update rule
            x_i^{l+1} = x_i^l + Σ(x_i^l - x_j^l)\phi_x(m_{ij})
        :param x: The coordinates (node embeddings) outputted from the previous layer
        :param edge_index: Array containing the connection between nodes
        :param coordinate_difference: The differences between two coordinates x_i and x_j
        :param messages: List of messages m_{ij} to be passed through the coordinate MLP \phi_x
        :return: The updated coordinates (node embeddings) x_i^{l+1}
        """

        row = edge_index[0]
        col = edge_index[1]
        
        x_sources, x_targets = self.get_sources_and_targets(adj_matrix, x)
        coordinate_linear_combination = self.self_multiplier * x_sources + self.other_multiplier * x_targets
        
        weighted_linear_combination = coordinate_linear_combination * self.coordinate_mlp(messages, time_embed)  # Latter part of the update rule
        relative_updated_coordinates = tf.math.unsorted_segment_mean(weighted_linear_combination, row, num_segments = x.shape[1])
        x += relative_updated_coordinates  # Finishes the update rule
        return x

    def compute_radials(self, adj_matrix, x, relative_differences = True):
        """
        Calculates the Minkowski distance (squared) between coordinates (node embeddings) x_i and x_j
        :param edge_index: Array containing the connection between nodes
        :param x: The coordinates (node embeddings)
        :return: Minkowski distances (squared) and coordinate differences x_i - x_j
        """
        
        x_sources, x_targets = self.get_sources_and_targets(adj_matrix, x)
        
        coordinate_differences = x_sources - x_targets
        minkowski_distance_squared = coordinate_differences ** 2
        c = tf.constant([-1, 1, 1, 1], dtype = tf.float32)
        minkowski_distance_squared = c * minkowski_distance_squared
        radial = tf.expand_dims(tf.reduce_sum(minkowski_distance_squared, axis = 2), axis = 2)
        return radial, coordinate_differences

    def call(self, input):
        h, x, edge_index, time_embed, edge_attribute, adj_matrix = input
        
        radial, coordinate_differences = self.compute_radials(adj_matrix, x)
        #time_embed = self.time_mlp(time_embed)
        h_sources, h_targets = self.get_sources_and_targets(adj_matrix, h)
        
        messages = self.compute_messages(h_sources, h_targets, radial, edge_attribute)
        x_updated = self.update_coordinates(x, edge_index, adj_matrix, coordinate_differences, messages, time_embed)
        if not self.final_layer:
            h_updated, _ = self.update_feature_vectors(h, edge_index, messages, time_embed)
            return h_updated, x_updated

        return None, x_updated
    
    
class LEGNN(Model):
    """
    The main network used for Lorentz group equivariance consisting of several layers of L_GCLs
    """

    def __init__(self, message_dim, output_feature_dim, edge_feature_dim, projection_dim = 32,
                 activation = layers.LeakyReLU(alpha=0.01), n_layers = 4):
        """
        Sets up the equivariant network and creates the necessary L_GCL layers
        :param input_feature_dim: The amount of numbers needed to specify a feature inputted into the LEGNN
        :param message_dim: The amount of numbers needed to specify a message passed through the LEGNN
        :param output_feature_dim: The amount of numbers needed to specify the updated feature after passing through the LEGNN
        :param edge_feature_dim: The amount of numbers needed to specify an edge attribute a_{ij}
        :param device: Specification on whether the cpu or gpu is to be used
        :param activation: The activation function used as the main non-linearity throughout the LEGNN
        :param n_layers: The number of layers the LEGNN network has
        """

        super(LEGNN, self).__init__()
        self.message_dim = message_dim
        self.projection_dim = projection_dim
        self.n_layers = n_layers
        self.feature_in = layers.Dense(message_dim, name = "in")  # Initial mixing of features
        #self.feature_out = layers.Dense(output_feature_dim, name = "out")  # Final mixing of features to yield desired output
        
        self.time_mlp = tf.keras.Sequential([
            layers.Dense(2 * projection_dim, activation = activation, name = "time_1"),
            layers.Dense(projection_dim, name = "time_2")
        ], name = "time")
        
        self.edges = None
        self.edge_attribute = None
        self.adj_matrix = None
        
        self.model_layers = []
        for i in range(0, n_layers):
            
            self.model_layers.append(L_GCL(self.message_dim, self.message_dim, self.message_dim,
                                     edge_feature_dim, projection_dim, activation = activation, 
                                     name = "Layer_" + str(i), final_layer = (i == n_layers - 1)))
            
    def set_edges(self, edges, edge_attribute, adj_matrix):
        self.edges = edges
        self.edge_attribute = edge_attribute
        self.adj_matrix = adj_matrix

    def call(self, input):
        x, time_embed = input
        edges, edge_attribute, adj_matrix = generate_edges(x.shape[1], x.shape[0])
        self.set_edges(edges, edge_attribute, adj_matrix)
        time_embed = self.time_mlp(time_embed)
        h = x * x
        c = tf.constant([-1, 1, 1, 1], dtype = tf.float32)
        h = c * h
        h = tf.expand_dims(tf.reduce_sum(h, axis = 2), axis = 2)
        h = self.feature_in(h)
        for i in range(self.n_layers):
            h, x = self.model_layers[i]([h, x, self.edges, time_embed, self.edge_attribute, self.adj_matrix])
        #h = self.feature_out(h)
        return x#h, x
    
