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


from operator import itemgetter
import random
import torch
import sklearn
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from iccad_contest.abstract_optimizer import AbstractOptimizer
from iccad_contest.design_space_exploration import experiment
from iccad_contest.functions.problem import get_pareto_frontier
from sklearn.linear_model import LinearRegression
import copy


class GaussianProcessRegressorOptimizer(AbstractOptimizer):
    primary_import = "iccad_contest"

    def __init__(self, design_space):
        """
            build a wrapper class for an optimizer.

            parameters
            ----------
            design_space: <class "MicroarchitectureDesignSpace">
        """
        AbstractOptimizer.__init__(self, design_space)
        kernel = DotProduct() + WhiteKernel()
        self.model = LinearRegression(
        )
        self.initial_suggestions = 10
        self.x = []
        self.y = []
        self.indexlist = []
        self.paretolist = []
        self.ppalist = []
        self.highestpower = 0
        self.first = True
        self.powerboundary = 1 # suggest it when power > self.highestpower - self.powerboundary
        self.lowpowertime = 0
        self.construct_microarchitecture_embedding_set()
        self.oa = True


    def construct_microarchitecture_embedding_set(self):
        microarchitecture_embedding_set = []
        for i in range(1, self.design_space.size + 1):
            microarchitecture_embedding_set.append(
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(i)
                )
            )
        self.set_vec = []
        for i in range(1, self.design_space.size + 1):
            self.set_vec.append(
                    self.design_space.idx_to_vec(i)
            )
        self.set_vec = np.array(self.set_vec)
        self.microarchitecture_embedding_set = np.array(microarchitecture_embedding_set)

    def suggest(self):

        """
            get a suggestion from the optimizer.

            returns
            -------
            next_guess: <list> of <list>
                list of `self.n_suggestions` suggestion(s).
                each suggestion is a microarchitecture embedding.
        """
        print("-------------------------start suggest -------------------------------")
        x_guess = random.sample(
            range(1, self.design_space.size + 1), k=self.initial_suggestions
        )
        oa = self.orthogonal_array()
        x_guess = [3300, 3163, 4642, 5012, 212, 1861, 4687, 2404, 565, 1114]
        if self.oa:
            initial_suggest =  oa
        else:
            initial_suggest =  [
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(_x_guess)
                ) for _x_guess in x_guess
            ]
        if not self.first:
            # NOTICE: we can also use the model to sweep the design space if
            # the design space is not quite large.
            # NOTICE: we only use a very naive way to pick up the design just for demonstration only.
            # ppa = torch.Tensor(self.model.predict(np.array(initial_suggest)))
            # potential_parteo_frontier = get_pareto_frontier(ppa)
            # _potential_suggest = []
            # for point in potential_parteo_frontier:
            #     index = torch.all(ppa == point.unsqueeze(0), axis=1)
            #     _potential_suggest.append(
            #         torch.Tensor(initial_suggest)[
            #             torch.all(ppa == point.unsqueeze(0), axis=1)
            #         ].tolist()[0]
            #     )
            ppa_all = torch.Tensor(self.model.predict(self.microarchitecture_embedding_set))
            potential_parteo_frontier = get_pareto_frontier(ppa_all)
            potential_suggest = []
            print("point amount")
            print(len(potential_parteo_frontier))
            for point in potential_parteo_frontier:
                potential_suggest.append(
                    self.microarchitecture_embedding_set[
                        torch.all(ppa_all == point.unsqueeze(0), axis=1)
                    ].tolist()[0]
                )
            ppa_list = ppa_all.tolist()
            max_area = 0
            max_ind = 0
            max_power = -100
            for ppa_ind in range(len(ppa_list)):
                ppa = ppa_list[ppa_ind]
                if ppa[1]>max_power:
                    max_power = ppa[1]
                    max_ind = ppa_ind
                if ((ppa[1] > self.highestpower - self.powerboundary) and (ppa[1] > 0)):
                    print("high power,predict power = %f, highest power = %f"%(ppa[1],self.highestpower))
                    return [torch.Tensor(self.microarchitecture_embedding_set)[
                               torch.all(ppa_all == ppa_all[ppa_ind].unsqueeze(0), axis=1)
                           ].tolist()[0]]
            if self.lowpowertime >= 3:
                self.lowpowertime = 0
                print("lower power time >=3,max power = %f, predict power = %f, highest power = %f"%(max_power,ppa_all[max_ind][1],self.highestpower))
                return[torch.Tensor(self.microarchitecture_embedding_set)[
                               torch.all(ppa_all == ppa_all[max_ind].unsqueeze(0), axis=1)
                           ].tolist()[0]]
            print("max power is",max_power)
                # else :
                #     pa = ppa[:]
                #     pa.pop(1)
                #     area = self.checkVolumn(pa)
                #     if area > max_area:
                #         max_area = area
                #         max_ind = ppa_ind
            potential_ppa_list = potential_parteo_frontier.tolist()
            max_perf = 0
            max_ar = 0
            max_power = 0
            max_area = 0
            max_ind = 0
            for ppa_ind in range(len(potential_parteo_frontier)):
                pa = potential_ppa_list[ppa_ind]
                perf = pa[0]
                power = pa[1]
                ar = pa[2]
                pa.pop(1)
                area = self.checkVolumn(pa)
                if area > max_area:
                    max_area = area
                    max_ind = ppa_ind
                    max_perf = perf
                    max_ar = ar
                    max_power = power
            pred = [max_perf,max_power,max_ar]
            print("low power, predict [p,p,a] is",pred,"area is ",max_area)
            self.lowpowertime += 1

            return [potential_suggest[max_ind]]
            

            

            print("-----------------not first suggest------------------")
            #return _potential_suggest
        else :
            print("--------------------first suggest--------------------")
            return initial_suggest

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
        print("--------------------start observe ---------------------------")
        for _x in x:
            self.x.append(_x)
        for _y in y:
            self.y.append(_y)
        if self.first:
            self.model.fit(np.array(self.x), np.array(self.y))
        pred = self.model.predict(np.array(x))
        print("prediction before fit: ",pred)
        self.model.fit(np.array(self.x), np.array(self.y))
        pred = self.model.predict(np.array(x))
        print("prediction after fit: ",pred)
        if self.first:
            print("--------------------construct-------------------------")
            self.first = False
            self.constructVolumn(y)
            self.ppalist = y
            self.paretolist = get_pareto_frontier(torch.Tensor(np.array(y))).tolist()
            print("self.pa = ",self.pa)
            print("self.paretolist",self.paretolist)
            print("--------------------construct-------------------------")
        else:
            self.ppalist.append(y[0])
            self.paretolist = get_pareto_frontier(torch.Tensor(np.array(self.ppalist))).tolist()
            if -y[0][1] > self.highestpower:
                self.highestpower = -y[0][1]
                self.powerboundary -= 0.1
            pa = copy.deepcopy(y[0])
            pa.pop(1)
            print("--------------------update-------------------------")
            print("new point of pa is",pa)
            print("self.pareto list = ",self.paretolist)
            self.updateVolumn(pa)
            print("--------------------update-------------------------")
            print("self.pa = ",self.pa)
            print("--------------------update-------------------------")


    # 

    def constructVolumn(self,ppa_list): # only construct power < 0
        ppa_tenser = torch.Tensor((np.array(ppa_list)))
        pareto_tenser = get_pareto_frontier(ppa_tenser)
        for pareto in pareto_tenser:
            print(pareto)
        pareto_list = ppa_list[:]
        for ppa in pareto_list:
            if -ppa[1] > self.highestpower:
                self.highestpower = -ppa[1]
                ppa[1] = 0
        print("p0a list",pareto_list)
        pareto_list = get_pareto_frontier(torch.Tensor((np.array(pareto_list)))).tolist() # set power = 0, the pareto frontier on 2d space
        print("pareto list",pareto_list)
        for ppa in pareto_list:
            ppa.pop(1) #remove power
            ppa[0] = ppa[0] + 5 #set bias of performence
            ppa[1] = ppa[1] + 5 #set bias of area
        sorted_pareto = sorted(pareto_list,key=itemgetter(0))
        for pa_ind in range(1,len(sorted_pareto)):
            if ((sorted_pareto[pa_ind][0] > sorted_pareto[pa_ind-1][0]) and (sorted_pareto[pa_ind][1] > sorted_pareto[pa_ind-1][1])):
                pareto_list.pop(pa_ind-1)
        area = 0
        oldpower = 0
        for pa in sorted_pareto:
            area += (pa[0]-oldpower)*pa[1]
            oldpower = pa[0]
        self.area = area
        self.pa = sorted_pareto
        print("self.area is %f"%self.area)
        print("self.pa is",type(self.pa),self.pa)

    def checkVolumn(self,in_pa): #return the point's influence to area
        in_pa[1] = -in_pa[1]
        in_pa[0] = in_pa[0] + 5
        in_pa[1] = in_pa[1] + 5
        copy_pa = copy.deepcopy(self.pa)
        for pa in copy_pa:
            if (pa[0] == in_pa[0]) and (pa[1]>=in_pa[1]):
                return 0
            elif pa[0] > in_pa[0]:
                if pa[1] >= in_pa[1]: # dominate by another point
                    return 0
                else:
                    break
        # print("----------------1---------------------")
        # print("self.pa is",self.pa)
        # print("copy_pa is",copy_pa)
        # print("in_pa is",in_pa)
        # print("----------------1---------------------")
        copy_pa.append(in_pa)
        for pa_ind in range(len(copy_pa)):
            pa = copy_pa[pa_ind]
            pa[1] = pa[1] - 5 #sub bias
            pa[0] = pa[0] - 5 #sub bias
            pa[1] = -pa[1] #inverse area again
            pa.insert(1,0)
            copy_pa[pa_ind] = pa
        p0a_tenser = torch.Tensor((np.array(copy_pa)))
        pareto = get_pareto_frontier(p0a_tenser).tolist()
        for ppa_ind in range(len(pareto)):
            ppa = pareto[ppa_ind]
            ppa.pop(1) #remove power
            ppa[0] = ppa[0] + 5 #set bias of performence
            ppa[1] = -ppa[1] + 5 #set bias of area
            pareto[ppa_ind] = ppa

        sort_pa = sorted(pareto,key=itemgetter(0))
        area = 0
        oldpower = 0
        for pa in sort_pa:
            area += (pa[0]-oldpower)*pa[1]
            oldpower = pa[0]
        return self.area - area

    def updateVolumn(self,in_pa): #modify self.area and self.pa by the added point
        in_pa[1] = -in_pa[1]
        in_pa[0] = in_pa[0] + 5
        in_pa[1] = in_pa[1] + 5
        copy_pa = copy.deepcopy(self.pa)
        for pa in copy_pa:
            if ((pa[0] == in_pa[0]) and (pa[1]>=in_pa[1])):
                return 
            elif pa[0] > in_pa[0]:
                if pa[1] >= in_pa[1]: # dominate by another point
                    return 
                else:
                    break
        print("in pa now is",in_pa)
        copy_pa.append(in_pa)
        for pa_ind in range(len(copy_pa)):
            pa = copy_pa[pa_ind]
            pa[1] = pa[1] - 5 #sub bias
            pa[0] = pa[0] - 5 #sub bias
            pa[1] = -pa[1] #inverse area again(get_pareto_frontier inverse area)
            pa.insert(1,0)
            copy_pa[pa_ind] = pa
        print("copy pa is",type(copy_pa),copy_pa)
        copy_pa = np.array(copy_pa)
        p0a_tenser = torch.Tensor((np.array(copy_pa)))
        pareto = get_pareto_frontier(p0a_tenser).tolist()
        print("pareto is (area neg, same as self.pa) = ",pareto)
        for ppa_ind in range(len(pareto)):
            ppa = pareto[ppa_ind]
            ppa.pop(1) #remove power
            ppa[0] = ppa[0] + 5 #set bias of performence
            ppa[1] = ppa[1] + 5 #set bias of area
            pareto[ppa_ind] = ppa
        print("pareto is (should be same as self.pa) = ",pareto)
        sort_pa = sorted(pareto,key=itemgetter(0))
        area = 0
        oldpower = 0
        for pa in sort_pa:
            area += (pa[0]-oldpower)*pa[1]
            oldpower = pa[0]
        self.area = area
        self.pa = sort_pa
    def orthogonal_array(self):
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
    experiment(GaussianProcessRegressorOptimizer)
