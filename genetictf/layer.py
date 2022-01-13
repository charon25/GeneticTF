import random
from typing import List, Union

import tensorflow as tf


ACTIVATION_FUNCTIONS = ['linear', 'sigmoid', 'tanh', 'relu', 'lrelu', 'prelu', 'elu']


class Layer:
    def __init__(self, neuron_count, activation, has_dropout, dropout_rate) -> None:
        self.neuron_count: int = neuron_count
        self.activation: str = activation
        self.has_dropout: bool = has_dropout
        self.dropout_rate: float = dropout_rate

    def uniform(self, probability):
        return random.random() < probability
    
    def get_tf_activation(self):
        assert self.activation in ACTIVATION_FUNCTIONS

        if self.activation in ('linear', 'sigmoid', 'tanh', 'relu', 'elu'):
            return self.activation
        if self.activation == 'lrelu':
            return tf.keras.layers.LeakyReLU(alpha=0.01)
        if self.activation == 'prelu':
            return tf.keras.layers.PReLU()

    def get_keras_layers(self) -> List[Union[tf.keras.layers.Dense, tf.keras.layers.Dropout]]:
        if not self.has_dropout:
            return [
                tf.keras.layers.Dense(self.neuron_count, activation=self.get_tf_activation())
            ]
        else:
            return [
                tf.keras.layers.Dense(self.neuron_count, activation=self.get_tf_activation()),
                tf.keras.layers.Dropout(self.dropout_rate)
            ]

    def create_new_layer_from_mutation(self, new_neurons_mean: float, new_neurons_sd: float, activation_mutation_p: float, dropout_switch_p: float, dropout_rate_variation: float) -> 'Layer':
        assert 0 < new_neurons_sd
        assert 0.0 <= activation_mutation_p <= 1.0
        assert 0 <= dropout_switch_p

        neuron_count = max(1, self.neuron_count + int(0.5 + random.gauss(new_neurons_mean, new_neurons_sd)))

        activation = self.activation
        if self.uniform(activation_mutation_p):
            activation = random.choice(ACTIVATION_FUNCTIONS)
        
        has_dropout = self.has_dropout
        if self.uniform(dropout_rate_variation):
            has_dropout = not has_dropout
        
        dropout_rate = min(1, max(0, self.dropout_rate + random.uniform(-dropout_rate_variation, dropout_rate_variation)))
        
        return Layer(neuron_count, activation, has_dropout, dropout_rate)

    def __repr__(self) -> str:
        return f'{self.neuron_count}-{self.activation}'


    @classmethod
    def create_random(cls, min_neuron_count: int, max_neuron_count: int, has_dropout_p: float, max_dropout_rate: float) -> 'Layer':
        assert 0 < min_neuron_count <= max_neuron_count
        assert 0 <= max_dropout_rate

        neuron_count: int = random.randint(min_neuron_count, max_neuron_count)
        activation: str = random.choice(ACTIVATION_FUNCTIONS)
        has_dropout = (random.random() < has_dropout_p)
        dropout_rate = random.uniform(0, max_dropout_rate)

        return cls(neuron_count, activation, has_dropout, dropout_rate)


    @staticmethod
    def get_output_layer(output_layer_size, output_activation) -> tf.keras.layers.Dense:
        return Layer(output_layer_size, output_activation, 0, 0).get_keras_layers()[0]
