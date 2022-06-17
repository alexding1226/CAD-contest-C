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
            ppa = torch.Tensor(self.model.predict(self.microarchitecture_embedding_set))
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
            total_x_lower = np.array(
                [self.settolowerd(_x) for _x in x]
            )
            total_y = np.array(y)
            #training_x = total_x[:self.training_size + 1]
            training_x = total_x_lower[:self.training_size + 1]
            training_y = total_y[:self.training_size + 1]
            #testing_x = total_x[self.training_size:]
            testing_x = total_x_lower[self.training_size:]
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
    def settolowerd(self,set):
        result = [set[4],set[11],set[14],set[15],set[17],set[19],set[20],set[25],set[28]]
        return result
        

    def lowertoset(self,lower):
        result = [4,1]
        ISU =   [
                [1,1,8,1,1,8,1,1,8],
                [1,1,6,1,1,6,1,1,6],
                [1,1,10,1,1,12,1,1,12]
                ]
        if lower[0] == 8:
            result.extend(ISU[0])
        elif lower[0] == 6:
            result.extend(ISU[1])
        else :
            result.extend(ISU[2])

        IFU =   [
                [8,8,16],
                [6,6,14],
                [10,12,20]
                ]
        if lower[1] == 8:
            result.extend(IFU[0])
        elif lower[1] == 6:
            result.extend(IFU[1])
        else :
            result.extend(IFU[2])
        ROB =   [32,30,34,36]
        result.append(lower[2])
        PRF =   [
                [52,48],
                [42,38],
                [62,58]]
        if lower[3] == 52:
            result.extend(PRF[0])
        elif lower[3] == 42:
            result.extend(PRF[1])
        else :
            result.extend(PRF[2])
        LSU =   [
                [8,8],
                [6,6],
                [12,12]
                ]
        if lower[4] == 8:
            result.extend(LSU[0])
        elif lower[4] == 6:
            result.extend(LSU[1])
        else :
            result.extend(LSU[2])
        ICACHE= [
                [64,4,1,32],
                [32,8,1,32],
                [32,4,1,16],
                [64,1,1,16]
                ]
        if (lower[5],lower[6]) == (64,4):
            result.extend(ICACHE[0])
        elif (lower[5],lower[6]) == (32,8):
            result.extend(ICACHE[1])
        elif (lower[5],lower[6]) == (32,4):
            result.extend(ICACHE[2])
        else :
            result.extend(ICACHE[3])
        DCACHE = [
                [64,4,0,2,1,8],
                [64,4,0,2,1,32],
                [64,4,1,2,1,8],
                [64,4,1,2,1,32]]
        if (lower[7],lower[8]) == (0,8):
            result.extend(DCACHE[0])
        elif (lower[7],lower[8]) == (0,32):
            result.extend(DCACHE[1])
        elif (lower[7],lower[8]) == (1,8):
            result.extend(DCACHE[2])
        else :
            result.extend(DCACHE[3])
        return result
if __name__ == "__main__":
    experiment(OfflineLinearRegressionOptimizer)
