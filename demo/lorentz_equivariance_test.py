from tensorflow.keras.layers import BatchNormalization, Layer, TimeDistributed
from tensorflow.keras.layers import Dense, Input, ReLU, Masking, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt
from graph_network import GNN
from equivariant_network import LEGNN, minkowski_norm_squared
from copy import deepcopy


def rotate_x(four_vector, angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return four_vector @ np.array([[1,    0,   0, 0],
                                   [0,  cos, sin, 0],
                                   [0, -sin, cos, 0],
                                   [0,    0,   0, 1]])


def rotate_y(four_vector, angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return four_vector @ np.array([[1,   0, 0,    0],
                                   [0, cos, 0, -sin],
                                   [0,   0, 1,    0],
                                   [0, sin, 0,  cos]])


def rotate_z(four_vector, angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return four_vector @ np.array([[1,    0,   0, 0],
                                   [0,  cos, sin, 0],
                                   [0, -sin, cos, 0],
                                   [0,    0,   0, 1]])


def boost_x(four_vector, rapidity):
    cosh = np.cosh(rapidity)
    sinh = np.sinh(rapidity)
    return four_vector @ np.array([[cosh, sinh, 0, 0],
                                   [sinh, cosh, 0, 0],
                                   [   0,    0, 1, 0],
                                   [   0,    0, 0, 1]])


def boost_y(four_vector, rapidity):
    cosh = np.cosh(rapidity)
    sinh = np.sinh(rapidity)
    return four_vector @ np.array([[cosh, 0, sinh, 0],
                                   [   0, 1,    0, 0],
                                   [sinh, 0, cosh, 0],
                                   [   0, 0,    0, 1]])


def boost_z(four_vector, rapidity):
    cosh = np.cosh(rapidity)
    sinh = np.sinh(rapidity)
    return four_vector @ np.array([[cosh, 0, 0, sinh],
                                   [   0, 1, 0,    0],
                                   [   0, 0, 1,    0],
                                   [sinh, 0, 0, cosh]])


def generate_jets(batch_size, jet_feature_dim):
    return tf.random.normal([batch_size, jet_feature_dim], mean = 5., stddev = 0.3)


def test_equivariance(model):
    boosts = []
    errors = []
    for i in range(0, 25, 1):
        boost = 10 ** ((i - 10.) / 2.) + 1.
        rapidity = np.arccosh(boost)

        p = tf.random.normal([event_count, 10, 4], dtype = tf.float64)
        t = tf.random.uniform([event_count, 1], dtype = tf.float64)

        p1 = deepcopy(p)
        t1 = deepcopy(t)
        output1 = model([p1, t1])
        output1_particle = boost_y(output1, rapidity)

        p2 = deepcopy(p)
        t2 = deepcopy(t)
        output2 = model([boost_y(p2, rapidity), t2])
        output2_particle = output2

        boosts.append(boost)
        errors.append(np.sqrt(np.mean((output2_particle - output1_particle) ** 2)))

    return boosts, errors


if __name__ == "__main__":
    event_count = 100

    model = LEGNN(4, [4, 1], [32, 32], num_layers = 4)
    boosts, errors = test_equivariance(model)

    model2 = GNN(4, [32, 32])
    boosts2, errors2 = test_equivariance(model2)

    axes = plt.axes()
    axes.set_xscale('log')
    axes.set_yscale('log')
    plt.title('Lorentz Equivariance Error vs. Boost Parameter')
    plt.ylabel('RMS Error')
    plt.xlabel('Y-axis Boost')
    plt.plot(boosts, errors, label = "Lorentz Equivariant Network")
    plt.plot(boosts2, errors2, label = "Graph Neural Network")
    plt.legend()
    plt.show()