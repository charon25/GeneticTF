import concurrent.futures
import itertools
import json
import os
from typing import Dict, List, Literal, Tuple, Union

import matplotlib.pyplot as plt
from tqdm import tqdm

from .network import Network


class EvolutionManager:
    def __init__(self, parameters: Dict, **kwargs) -> None:
        assert 'network_mutation_parameters' in parameters
        assert 'network_creation_parameters' in parameters
        assert 'layer_mutation_parameters' in parameters
        assert 'layer_creation_parameters' in parameters
        assert 'population_size' in parameters
        assert 'survivors_by_generation' in parameters
        assert 'mutated_by_generation' in parameters
        assert 'random_by_generation' in parameters
        assert 'fit_parameters' in parameters

        self.parameters = parameters

        self.network_mutation_parameters = parameters['network_mutation_parameters']
        self.network_creation_parameters = parameters['network_creation_parameters']
        self.layer_mutation_parameters = parameters['layer_mutation_parameters']
        self.layer_creation_parameters = parameters['layer_creation_parameters']
        
        self.population_size = parameters['population_size']
        self.survivors_by_generation = parameters['survivors_by_generation']
        self.mutated_by_generation = parameters['mutated_by_generation']
        self.random_by_generation = parameters['random_by_generation']

        self.fit_parameters = parameters['fit_parameters']

        self.networks: List[Network] = []
        self.network_index = kwargs.get('network_index', 0)

        self.fitnesses = {}
        self.best_fitnesses = kwargs.get('best_fitnesses', [])
        self.average_fitnesses = kwargs.get('average_fitnesses', [])
        
        self.generation_count = kwargs.get('generation_count', 0)

    def initialize_population(self):
        self.networks = [Network.create_random(self.network_creation_parameters, self.layer_creation_parameters, index) for index in range(self.population_size)]
        self.network_index = self.population_size


    def get_fitnesses(self, data: List, labels: List, split: float = 0.8, worker_count:int = 3) -> Dict[Network, float]:
        fitnesses = {}
        if worker_count == 1:
            for _, network in tqdm(enumerate(self.networks), total=self.population_size, leave=False, desc=f"Generation {1+self.generation_count}"):
                fitnesses[network] = network.get_fitness(data, labels, split, self.fit_parameters)
        else:
            with tqdm(total=self.population_size, leave=False, desc=f"Generation {1+self.generation_count}") as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures_to_networks = {executor.submit(network.get_fitness, data, labels, split, self.fit_parameters): network for network in self.networks}
                    for future in concurrent.futures.as_completed(futures_to_networks):
                        pbar.update(1)
                        fitnesses[futures_to_networks[future]] = future.result()
        return fitnesses


    # Returns the generation index, best, average and worst fitness of the generation
    def pass_one_generation(self, data: List, labels: List, split: float = 0.8, worker_count:int = 3) -> Tuple[float, float, float]:
        assert 0 < worker_count

        self.fitnesses = self.get_fitnesses(data, labels, split, worker_count)
        self.fitnesses = {network: fitness for network, fitness in sorted(self.fitnesses.items(), key=lambda item:item[1], reverse=True)}

        mutated = [network for network in list(self.fitnesses.keys())[:self.mutated_by_generation]]

        next_generation: List[Network] = [network for network in list(self.fitnesses.keys())[:min(self.survivors_by_generation, self.population_size - self.random_by_generation)]]

        index = 0
        while len(next_generation) < self.population_size - self.random_by_generation:
            next_generation.append(mutated[index].create_new_network_from_mutation(self.network_mutation_parameters, self.layer_mutation_parameters, self.layer_creation_parameters, self.network_index))
            self.network_index += 1
            index += 1
            if index >= self.mutated_by_generation:
                index = 0

        for _ in range(self.random_by_generation):
            next_generation.append(Network.create_random(self.network_creation_parameters, self.layer_creation_parameters, self.network_index))
            self.network_index += 1
        
        self.networks = next_generation

        self.generation_count += 1

        fitness_values = list(self.fitnesses.values())
        self.best_fitnesses.append(fitness_values[0])
        self.average_fitnesses.append(sum(fitness_values) / self.population_size)
        return (self.generation_count, fitness_values[0], self.average_fitnesses[-1], fitness_values[-1])
    
    def pass_n_generations(self, n_generations: int, data: List, labels: List, split: float = 0.8, worker_count:int = 3) -> List[Tuple[float, float]]:
        assert 0 < n_generations

        fitnesses = []
        for _ in tqdm(range(n_generations), total=n_generations, desc="Generations"):
            fitnesses.append(self.pass_one_generation(data, labels, split, worker_count))

        return fitnesses

    def pass_generations(self, savepath: Union[Literal[None], str], data: List, labels: List, split: float = 0.8, worker_count:int = 3, save_each_gen: bool = False) -> List[Tuple[float, float]]:
        assert savepath is None or os.path.exists(savepath) or os.access(os.path.dirname(('.\\' if os.path.dirname(savepath) == '' else '') + savepath), os.W_OK)

        fitnesses = []
        for _ in tqdm(itertools.count(), desc="Generations"):
            try:
                fitnesses.append(self.pass_one_generation(data, labels, split, worker_count))

                if save_each_gen and savepath is not None:
                    self.save(savepath)
            except KeyboardInterrupt:
                break
        
        if savepath is not None:
            self.save(savepath)
        
        return fitnesses

    def save(self, savepath):
        assert type(savepath) == str
        if os.path.dirname(savepath) == '':
            savepath = '.\\' + savepath
        assert os.path.exists(savepath) or os.access(os.path.dirname(savepath), os.W_OK)

        manager_object = {
            'parameters': self.parameters,
            'networks': [network.to_object() for network in self.networks],
            'network_index': self.network_index,
            'best_fitnesses': self.best_fitnesses,
            'average_fitnesses': self.average_fitnesses,
            'generation_count': self.generation_count
        }

        try:
            json.dump(manager_object, open(savepath, 'w', encoding='utf-8'))
            return None
        except Exception:
            return manager_object


    def print(self):
        print(f'===== Evolution manager (generation {self.generation_count}) =====\nPopulation size : {self.population_size}\nSurvivors each generation : {self.survivors_by_generation}\nNew random networks each generation : {self.random_by_generation}')
        print('=== Networks')
        print(' - ' + '\n - '.join(f'{network} => {100 * self.fitnesses[network] if network in self.fitnesses else -1:.2f}' for network in self.networks))
        print('=== Data')
        print(f'Best fitness of last generation : {100 * self.best_fitnesses[-1] if len(self.best_fitnesses) > 0 else -1:.2f}')
        print(f'Average fitness of last generation : {100 * self.average_fitnesses[-1] if len(self.average_fitnesses) > 0 else -1:.2f}')
        print('==========')

    def graph(self, show=True, savepath=None):
        assert savepath is None or os.path.exists(savepath) or os.access(os.path.dirname(('.\\' if os.path.dirname(savepath) == '' else '') + savepath), os.W_OK)

        plt.clf()
        plt.plot(range(1, 1 + len(self.best_fitnesses)), self.best_fitnesses, 'o', label='Best fitness')
        plt.plot(range(1, 1 + len(self.average_fitnesses)), self.average_fitnesses, 'o', label='Average fitness')
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.ylim([0, 1])
        plt.grid(True, 'both', 'both')
        plt.legend(loc='lower right')
        
        if savepath is not None:
            plt.savefig(savepath)

        if show:
            plt.show()
        

    @classmethod
    def from_file(cls, filepath):
        assert type(filepath) == str
        if os.path.dirname(filepath) == '':
            filepath = '.\\' + filepath
        assert os.path.exists(filepath)

        man_object = json.load(open(filepath, 'r', encoding='utf-8'))

        manager = cls(
            man_object['parameters'],
            network_index=man_object['network_index'],
            best_fitnesses=man_object['best_fitnesses'],
            average_fitnesses=man_object['average_fitnesses'],
            generation_count=man_object['generation_count']
        )

        output_layer_size, output_type = manager.network_creation_parameters[2], manager.network_creation_parameters[3]
        for network in man_object['networks']:
            manager.networks.append(Network.create_from_object(network, output_layer_size, output_type))

        return manager