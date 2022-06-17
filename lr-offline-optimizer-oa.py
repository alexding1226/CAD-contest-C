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
        self.testing_size = 10
        self.training_size = 25
        self.n_suggestions = 10
        self.fit = True
        self.microarchitecture_embedding_set = self.construct_microarchitecture_embedding_set()
        self.first = True

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
        if not self.first:
            # we only use a very naive way to pick up the design just for demonstration only.
            print("------------------not first suggest -------------------------------")
            ppa = torch.Tensor(self.model.predict(self.microarchitecture_embedding_set))
            print("here is ppa")
            print(ppa)
            potential_parteo_frontier = get_pareto_frontier(ppa)
            potential_suggest = []
            print("point amount")
            print(len(potential_parteo_frontier))
            for point in potential_parteo_frontier:
                potential_suggest.append(
                    self.microarchitecture_embedding_set[
                        torch.all(ppa == point.unsqueeze(0), axis=1)
                    ].tolist()[0]
                )
            return potential_suggest
        else:
            print("------------------first suggest -------------------------------")
            x_guess = random.sample(
                range(1, self.design_space.size + 1), k=self.testing_size
            )
            initial_suggest = self.initial_sample()
            #print("initial suggest")
            testing_suggest = [
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(_x_guess)
                ) for _x_guess in x_guess
            ]
            print("testing suggest\n",testing_suggest)

            initial_suggest.extend(testing_suggest)
            self.first = False
            
            return initial_suggest

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
            total_y = np.array(y)
            training_x = total_x[:self.training_size + 1]
            training_y = total_y[:self.training_size + 1]
            testing_x = total_x[self.training_size:]
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
        pass
    def initial_sample(self):
        factor_levels = [3, 3, 4, 3, 3, 4, 4]
        oa = [
            [1,1,1,1,1,1],
            [1,2,2,2,2,2],
            [1,3,3,3,3,3],
            [1,4,4,4,4,4],
            [1,5,5,5,5,5],
            [2,1,2,3,4,5],
            [2,2,3,4,5,1],
            [2,3,4,5,1,2],
            [2,4,5,1,2,3],
            [2,5,1,2,3,4],
            [3,1,3,5,2,4],
            [3,2,4,1,3,5],
            [3,3,5,2,4,1],
            [3,4,1,3,5,2],
            [3,5,2,4,1,3],
            [4,1,4,2,5,3],
            [4,2,5,3,1,4],
            [4,3,1,4,2,5],
            [4,4,2,5,3,1],
            [4,5,3,1,4,2],
            [5,1,5,4,3,2],
            [5,2,1,5,4,3],
            [5,3,2,1,5,4],
            [5,4,3,2,1,5],
            [5,5,4,3,2,1]
            ]
        for lis in oa:
            for ind in range(6):
                if lis[ind] > factor_levels[ind]:
                    lis[ind] = random.randint(1,factor_levels[ind])
            lis.append(random.randint(1,4))
        ISU =   [
                [1,1,8,1,1,8,1,1,8],
                [1,1,6,1,1,6,1,1,6],
                [1,1,10,1,1,12,1,1,12]
                ]
        IFU =   [
                [8,8,16],
                [6,6,14],
                [10,12,20]
                ]
        ROB =   [
                [32],
                [30],
                [34],
                [36]
                ]
        PRF =   [
                [52,48],
                [42,38],
                [62,58]]
        LSU =   [
                [8,8],
                [6,6],
                [12,12]
                ]
        ICACHE= [
                [64,4,1,32],
                [32,8,1,32],
                [32,4,1,16],
                [64,1,1,16]
                ]
        DCACHE = [
                [64,4,0,2,1,8],
                [64,4,0,2,1,32],
                [64,4,1,2,1,8],
                [64,4,1,2,1,32]]
        all_param = [ISU,IFU,ROB,PRF,LSU,ICACHE,DCACHE]
        result = []
        for vec_ind in range(len(oa)):
            vec_params = oa[vec_ind]
            vec = [4,1]
            for param_ind in range(7):
                param = vec_params[param_ind] -1
                vec.extend(all_param[param_ind][param])
            result.append(vec)
            
        return result

if __name__ == "__main__":
    experiment(OfflineLinearRegressionOptimizer)
