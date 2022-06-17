"""
    gaussian process optimizer constructs a gaussian process model
    for design space exploration.
    it updates the model progressively and picks the most valuable design to evaluate,
    hoping to reduce the total running time.
    a command to test "gp-optimizer.py":
    ``` 
        python example_optimizer/gp-optimizer.py \
            -o [your experiment outputs directory] \
            -q [the number of your queries] \
            -s /path/to/example_optimizer/gp-configs.json
    ```
    `optimizer`, `random_state`, etc. are provided with `gp-configs.json`, making you
    develop your optimizer conveniently when you tune your solution.
    you can specify more options to test your optimizer. please use
    ```
        python example_optimizer/gp-optimizer.py -h
    ```
    to check.
    the code is only for example demonstration.
"""


import random
import torch
import sklearn
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from iccad_contest.abstract_optimizer import AbstractOptimizer
from iccad_contest.design_space_exploration import experiment
from iccad_contest.functions.problem import get_pareto_frontier


class GaussianProcessRegressorOptimizer(AbstractOptimizer):
    primary_import = "iccad_contest"

    def __init__(self, design_space, optimizer, random_state):
        """
            build a wrapper class for an optimizer.

            parameters
            ----------
            design_space: <class "MicroarchitectureDesignSpace">
        """
        AbstractOptimizer.__init__(self, design_space)
        kernel = DotProduct() + WhiteKernel()
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=optimizer,
            random_state=random_state
        )
        self.n_suggestions = 5
        self.x = []
        self.y = []
        self.first = True
        self.microarchitecture_embedding_set = self.construct_microarchitecture_embedding_set()
        self.compute_cand_specify()

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
                list of `self.n_suggestions` suggestion(s).
                each suggestion is a microarchitecture embedding.
        """
        x_guess = random.sample(
            range(1, self.design_space.size + 1), k=self.n_suggestions
        )
        potential_suggest =  [
            self.design_space.vec_to_microarchitecture_embedding(
                self.design_space.idx_to_vec(_x_guess)
            ) for _x_guess in x_guess
        ]
        if not self.first:
            # NOTICE: we can also use the model to sweep the design space if
            # the design space is not quite large.
            # NOTICE: we only use a very naive way to pick up the design just for demonstration only.
            ppa = torch.Tensor(self.model.predict(np.array(potential_suggest)))
            potential_parteo_frontier = get_pareto_frontier(ppa)
            _potential_suggest = []
            for point in potential_parteo_frontier:
                index = torch.all(ppa == point.unsqueeze(0), axis=1)
                _potential_suggest.append(
                    torch.Tensor(potential_suggest)[
                        torch.all(ppa == point.unsqueeze(0), axis=1)
                    ].tolist()[0]
                )
            print("-----------------not first suggest------------------")
            return _potential_suggest
        else :
            print("--------------------first suggest--------------------")
            self.first=False
            return potential_suggest

    def observe(self, x, y):
        """
            send an observation of a suggestion back to the optimizer.

            parameters
            ----------
            x: <list> of <list>
                the output of `suggest`.
            y: <list> of <list>
                corresponding values where each `x` is mapped to.
        """
        for _x in x:
            self.x.append(_x)
        for _y in y:
            self.y.append(_y)
        self.model.fit(np.array(self.x), np.array(self.y))
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
        print(self.design_space_cand)
    def compute_cand_specify(self):
        # fetch,decoder,isu,ifu,rob,prf,lsu,icache,dcache
        # fetch and decoder have only one candidate
        # design_space_cand is 2d array with 7 component
        fetch_cand = [[1]]
        decoder_cand = [[1]]

        self.design_space_cand = []



if __name__ == "__main__":
    experiment(GaussianProcessRegressorOptimizer)
