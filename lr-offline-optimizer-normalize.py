"""
    linear regression (offline) optimizer constructs a linear regression model
    for design space exploration.
    it trains the model with many selected samples and freezes the model to search.
    a command to test "lr-offline-optimizer.py":
    ``` 
        python example_optimizer/lr-offline-optimizer.py \
            -o [your experiment outputs directory] \
            -q [the number of your queries]
    ```
    set the '--num-of-queries' to 2 to avoid more time cost.
    you can specify more options to test your optimizer. please use
    ```
        python example_optimizer/lr-offline-optimizer.py -h
    ```
    to check.
"""


import random
import torch
import sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from iccad_contest.abstract_optimizer import AbstractOptimizer
from iccad_contest.design_space_exploration import experiment
from iccad_contest.functions.problem import get_pareto_frontier


class OfflineLinearRegressionOptimizer(AbstractOptimizer):
    primary_import = "iccad_contest"

    def __init__(self, design_space):
        """
            build a wrapper class for an optimizer.

            parameters
            ----------
            design_space: <class "MicroarchitectureDesignSpace">
        """
        AbstractOptimizer.__init__(self, design_space)
        self.model = LinearRegression()
        # NOTICE: you can put `self.initial_size`, `self.training_size`, etc.,
        # to a JSON file for better coding and tuning experience.
        self.initial_size = 50
        self.training_size = round(0.8 * self.initial_size)
        self.n_suggestions = 10
        self.fit = True
        self.microarchitecture_embedding_set = self.construct_microarchitecture_embedding_set()
        self.compute_cand()

    def construct_microarchitecture_embedding_set(self):
        microarchitecture_embedding_set = []
        for i in range(1, self.design_space.size + 1):
            microarchitecture_embedding_set.append(
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(i)
                )
            )
        return np.array(microarchitecture_embedding_set)

    def suggest(self):
        """
            get a suggestion from the optimizer.

            returns
            -------
            next_guess: <list> of <list>
                list of suggestions.
                each suggestion is a microarchitecture embedding.
        """
        try:
            # we only use a very naive way to pick up the design just for demonstration only.
            normalize_set = [
                (np.array(_set) / self.design_space_cand_max) for _set in self.microarchitecture_embedding_set
            ]
            ppa = torch.Tensor(self.model.predict(np.array(normalize_set)))
            print("here is ppa")
            print(ppa)
            potential_parteo_frontier = get_pareto_frontier(ppa)
            potential_suggest = []
            for point in potential_parteo_frontier:
                potential_suggest.append(
                    self.microarchitecture_embedding_set[
                        torch.all(ppa == point.unsqueeze(0), axis=1)
                    ].tolist()[0]
                )
            
            return potential_suggest
        except sklearn.exceptions.NotFittedError:
            x_guess = random.sample(
                range(1, self.design_space.size + 1), k=self.initial_size
            )
            potential_suggest =  [
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(_x_guess)
                ) for _x_guess in x_guess
            ]
            return potential_suggest

    def observe(self, x, y):
        """
            send an observation of a suggestion back to the optimizer.

            parameters
            ----------
            x: <list> of <list>
                the output of `suggest`.
            y: <list> of <list>
                corresponding objective values where each `x` is mapped to.
        """
        if self.fit:
            # NOTICE: we can check the model's accuracy and verify if it is
            # suitable for exploration.
            total_x = np.array(x)
            normalize_x = np.array([
                (np.array(_x) / self.design_space_cand_max) for _x in x
            ])
            total_y = np.array(y)
            training_x = normalize_x[:self.training_size + 1]
            training_y = total_y[:self.training_size + 1]
            testing_x = normalize_x[self.training_size:]
            testing_y = total_y[self.training_size:]
            self.model.fit(training_x, training_y)
            pred = self.model.predict(testing_x)
            mape = mean_absolute_percentage_error(testing_y, pred)
            print(
                "linear regression trained on data size: {}, " \
                "mean absolute percentage error: {}".format(
                    self.training_size,
                    mape
                )
            )
            self.fit = False
    def compute_cand(self):
        vector_size = self.microarchitecture_embedding_set[0].size # should be 29
        # initialize design_space_cand to a 2d array of size 29
        self.design_space_cand = []
        for _ in range(vector_size):
            self.design_space_cand.append([])
        for set in self.microarchitecture_embedding_set:
            for index in range(vector_size):
                if(set[index] not in self.design_space_cand[index]):
                    self.design_space_cand[index].append(set[index])
        self.design_space_cand_max = []
        for index in range(vector_size):
            self.design_space_cand_max.append(max(self.design_space_cand[index]))
        self.design_space_cand_max = np.array(self.design_space_cand_max)


if __name__ == "__main__":
    experiment(OfflineLinearRegressionOptimizer)
