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
import math


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
        self.training_size = 40
        self.n_suggestions = 10
        self.fit = True
        self.microarchitecture_embedding_set = self.construct_microarchitecture_embedding_set()
        self.first = True
        self.u = 0.1
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
        self.set_normalize = [
                (np.array(_set) / self.design_space_cand_max) for _set in self.microarchitecture_embedding_set
            ]

    def initial_sample(self):
        Kmatrix = np.array([[self.distance_vec(v1, v2) for v2 in self.set_normalize] for v1 in self.set_normalize])
        print("matrix initialize complete")
        train_ind = []
        for i in range(self.training_size):
            print("data number %i generating" %i)
            largest_ind = self.largest_col(Kmatrix,train_ind)
            train_ind.append(largest_ind+1)
            Kmatrix = self.update_mat(Kmatrix,largest_ind)
            print("data number %i is %i"%(i,largest_ind))
        print("train index",train_ind)
        train_set = [
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(_x_guess)
                ) for _x_guess in train_ind
            ]
        return train_set

    def largest_col(self,mat,chose):
        norms = np.linalg.norm(mat,axis=0)
        norms_square = norms*norms

        # norms_div = np.array([
        #     norms_square[i]/(mat[i][i]+self.u) for i in range(len(mat))
        # ])
        norms_div = []
        for i in range(len(mat)):
            norms_div.append(norms_square[i]/(mat[i][i]+self.u))
        #print("norms div : ",norms_div)
        # for ind in chose:
        #     norms_div[ind] = 0
        max_ind = np.argmax(norms_div)
        print("max index is %i" %max_ind)
        print("max value is %.9g" %norms_div[max_ind])
        return max_ind

    def update_mat(self,mat,i):
        denum = mat[i][i]+self.u
        col = mat[:,[i]]
        sub_mat = np.matmul(col , np.transpose(col))/denum
        new_mat = mat - sub_mat
        # new_mat[:,i] = 0
        # new_mat[i,:] = 0
        #new_mat = np.delete(new_mat,i,0)
        #new_mat = np.delete(new_mat,i,1)
        # for j in range(size):
        #     for k in range(size):
        #         new_mat[j][k] = mat[j][k] - mat[j][i]*mat[k][i]/denum
        return new_mat

    def distance_vec(self,vec1,vec2):
        dist = np.linalg.norm(vec1-vec2)
        rbf = math.exp(-(dist**2/0.02)) #0.02 : 2*sigma^2
        return rbf



if __name__ == "__main__":
    experiment(OfflineLinearRegressionOptimizer)
