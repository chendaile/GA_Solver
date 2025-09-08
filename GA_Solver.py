import numpy as np
import json


class GA_Solver:
    DEFAULT_PARAMETERS = {"POP_SIZE": 100,
                          "MAX_GEN": 100,
                          "STANDARD_DEVIATION": 10,
                          "N_CROSSOVER": 3
                          }

    def __init__(self, init_matrix: np.ndarray):
        self.CONFIG = GA_Solver.DEFAULT_PARAMETERS
        self.solve_shape = init_matrix.shape
        self.init_matrix = init_matrix

    def UpdateConfig(self, JSON_PATH: str):
        try:
            with open(JSON_PATH, 'r') as f:
                tmp_read = json.load(f)
        except Exception as e:
            print(f"Wrong reading json: {e}")

        for key, value in tmp_read.items():
            if key not in GA_Solver.DEFAULT_PARAMETERS.keys():
                print(f"Wrong parameter {key}")
            self.CONFIG[key] = value
            print(f"Change {key}'s value to {value}")

    def get_MutateIndividual(self, individual: np.ndarray):
        tmp_individual = individual.copy()
        tmp_individual += np.random.normal(0,
                                           GA_Solver.DEFAULT_PARAMETERS["STANDARD_DEVIATION"])
        return tmp_individual

    def get_CrossoverIndividuals(self, individuals):
        weights = np.random.rand(len(individuals))
        weights = weights / weights.sum()
        return sum(w * ind for w, ind in zip(weights, individuals))
