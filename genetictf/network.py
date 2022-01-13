import os
import random
from typing import Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')

from .layer import Layer, ACTIVATION_FUNCTIONS


LOSS_FUNCTIONS = ['bce', 'cce', 'poi', 'kld', 'mse', 'mae', 'cs', 'hub', 'lc', 'hin', 'shin', 'chin', 'scclogits']


class Network:
    def __init__(self, layers: List[Layer], loss_function: str, output_layer_size: int, output_activation: str, output_type: str, history: Union[Tuple[int], int]) -> None:
        assert len(layers) > 0
        assert output_type in ('classification_one_value', 'classification_multiple_values', 'values')

        self.layer_count: int = len(layers)

        self.layers: List[Layer] = layers

        self.loss_function: str = loss_function

        self.output_layer_size: int = output_layer_size
        self.output_activation: str = output_activation

        self.output_type: str = output_type

        self.history = [history] if type(history) == int else history

        self.fitness = None

        self.model = None
    
    def uniform(self, probability):
        return random.random() < probability

    def get_accuracy(self, test_data: List, test_labels: List, verbose: int) -> float:
        if self.output_type == 'classification_one_value':
            test_results = self.model(test_data).numpy()
            return np.sum(np.argmax(test_results, axis=1) == test_labels) / len(test_data)

        elif self.output_type == 'classification_multiple_values':
            test_results = self.model(test_data).numpy()
            test_results_max = np.zeros_like(test_results)[np.arange(len(test_results)), test_results.argmax(axis=1)] = 1
            return np.sum(np.all(test_results_max == test_labels, axis=1)) / len(test_data)

        elif self.output_type == 'values':
            _, mean_squared_error = self.model.evaluate(test_data, test_labels, verbose=verbose)
            return mean_squared_error


    def get_tf_loss_function(self):
        assert self.loss_function in LOSS_FUNCTIONS

        if self.loss_function == 'bce':
            return tf.keras.losses.BinaryCrossentropy()
        if self.loss_function == 'cce':
            return tf.keras.losses.CategoricalCrossentropy()
        if self.loss_function == 'poi':
            return tf.keras.losses.Poisson()
        if self.loss_function == 'kld':
            return tf.keras.losses.KLDivergence()
        if self.loss_function == 'mse':
            return tf.keras.losses.MeanSquaredError()
        if self.loss_function == 'mae':
            return tf.keras.losses.MeanAbsoluteError()
        if self.loss_function == 'cs':
            return tf.keras.losses.CosineSimilarity(axis=1)
        if self.loss_function == 'hub':
            return tf.keras.losses.Huber()
        if self.loss_function == 'lc':
            return tf.keras.losses.LogCosh()
        if self.loss_function == 'hin':
            return tf.keras.losses.Hinge()
        if self.loss_function == 'shin':
            return tf.keras.losses.SquaredHinge()
        if self.loss_function == 'chin':
            return tf.keras.losses.CategoricalHinge()
        if self.loss_function == 'scclogits':
            return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


    def construct_model(self) -> None:
        self.model = tf.keras.Sequential()
        for layer in self.layers:
            for keras_layer in layer.get_keras_layers():
                self.model.add(keras_layer)
        
        self.model.add(Layer.get_output_layer(self.output_layer_size, self.output_activation))

        metrics = ['MeanSquaredError'] if self.output_type == 'values' else []

        self.model.compile(loss=self.get_tf_loss_function(), optimizer='adam', metrics=metrics)


    def split_train_test(self, data: List, labels: List, split: float) -> Tuple:
        data = np.array(data)
        labels = np.array(labels)

        p = np.random.permutation(len(data))
        data = data[p]
        labels = labels[p]

        train_limit = int(split * len(data))

        if len(labels.shape) == 1:
            return (data[:train_limit, :], data[train_limit:, :], labels[:train_limit], labels[train_limit:])
        else:
            return (data[:train_limit, :], data[train_limit:, :], labels[:train_limit, :], labels[train_limit:, :])

    # Return the average accuracy over multiple trainings
    def get_fitness(self, data: List, labels: List, split: float = 0.8, fit_parameters: Tuple[int] = (30, 20, 32, 0)) -> float:
        assert len(data) == len(labels)
        assert 0.0 < split < 1.0
        assert len(fit_parameters) == 4
        
        repetitions, epochs, batch_size, verbose = fit_parameters
        assert 0 < repetitions
        assert 0 < epochs
        assert 0 < batch_size
        assert 0 <= verbose <= 2

        accuracies = []

        for _ in tqdm(range(repetitions), total=repetitions, leave=False, desc="Network progression"):
            train_data, test_data, train_labels, test_labels = self.split_train_test(data, labels, split)

            self.construct_model()
            self.model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, verbose=verbose)

            accuracy = self.get_accuracy(test_data, test_labels, verbose)
            accuracies.append(accuracy)

        self.fitness = sum(accuracies) / len(accuracies)
        return self.fitness
        

    def create_new_network_from_mutation(self, network_mutation_parameters: Tuple[float], layer_mutation_parameters: Tuple[float], layer_creation_parameters: Tuple[float], index: int) -> 'Network':
        new_layer_p, remove_layer_p, output_activation_mutation_p, loss_function_mutation_p = network_mutation_parameters
        assert 0.0 <= new_layer_p <= 1.0
        assert 0.0 <= remove_layer_p <= 1.0
        assert 0.0 <= output_activation_mutation_p <= 1.0
        assert 0.0 <= loss_function_mutation_p <= 1.0
        
        layers: List[Layer] = []
        for layer in self.layers:
            layers.append(layer.create_new_layer_from_mutation(*layer_mutation_parameters))
        
        if len(layers) > 1 and self.uniform(remove_layer_p):
            layers.pop(random.randrange(len(layers)))

        if self.uniform(new_layer_p):
            index = random.randint(0, len(layers))
            layers.insert(index, Layer.create_random(*layer_creation_parameters))

        output_activation = self.output_activation
        if self.uniform(output_activation_mutation_p):
            output_activation = random.choice(ACTIVATION_FUNCTIONS)
        
        loss_function = self.loss_function
        if self.uniform(loss_function_mutation_p):
            loss_function = random.choice(LOSS_FUNCTIONS)
        
        return Network(layers, loss_function, self.output_layer_size, output_activation, self.output_type, self.history + [index])
    
    def to_object(self) -> Dict:
        layers_object = []
        for layer in self.layers:
            layers_object.extend([layer.neuron_count, layer.activation, layer.has_dropout, layer.dropout_rate])

        return {
            'history': self.history,
            'layers': layers_object,
            'output_activation': self.output_activation,
            'loss_function': self.loss_function
        }

    def __repr__(self) -> str:
        return f'{self.history}:{"/".join(repr(layer) for layer in self.layers)}->{self.output_layer_size}-{self.output_activation}:{self.loss_function}'

    @classmethod
    def create_from_object(cls: 'Network', network_object: Dict, output_layer_size: int, output_type: str) -> 'Network':
        layers = []
        for k in range(0, len(network_object['layers']), 4):
            layers.append(Layer(network_object['layers'][k], network_object['layers'][k + 1], network_object['layers'][k + 2], network_object['layers'][k + 3]))

        return cls(layers, network_object['loss_function'], output_layer_size, network_object['output_activation'], output_type, network_object['history'])


    @classmethod
    def create_random(cls, network_creation_parameters: Tuple[Union[int, float]], layer_creation_parameters: Tuple[float], index) -> 'Network':
        min_layer_count, max_layer_count, output_layer_size, output_type, only_loss_function = network_creation_parameters
        assert 0 < min_layer_count <= max_layer_count
        assert output_layer_size > 0
        assert only_loss_function is None or type(only_loss_function) == str

        layer_count = random.randint(min_layer_count, max_layer_count)
        layers = [Layer.create_random(*layer_creation_parameters) for _ in range(layer_count)]

        loss_function = only_loss_function if only_loss_function is not None else random.choice(LOSS_FUNCTIONS)

        output_activation = random.choice(ACTIVATION_FUNCTIONS)

        return cls(layers, loss_function, output_layer_size, output_activation, output_type, index)
