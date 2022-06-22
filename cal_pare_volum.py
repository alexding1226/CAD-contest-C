from operator import itemgetter
import random
from unittest.util import sorted_list_difference
import torch
import sklearn
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from iccad_contest.abstract_optimizer import AbstractOptimizer
from iccad_contest.design_space_exploration import experiment
from iccad_contest.functions.problem import get_pareto_frontier
import copy


def sweep2d_vol(li,hdiff):
    area=0
    sor2d=li[li[:, 0].argsort()[::-1]]
    prev=sor2d[0]
    first=1
    prevarea=0
    print("*****each turn point is*****\n")
    print(sor2d,'\n')
    for point in sor2d:
        if first:
            area=point[0]*point[1]
            # print("a加了%d area\n"%area)
            first=0
            prevarea=area
        else:
            if(point[0]==prev[0]):
                if (point[1]>prev[1]):
                    area+=point[0]*(point[1]-prev[1])
                    # print("b加了%d area\n"%(point[0]*(point[1]-prev[1])))
                    prev=point
                    prevarea=point[0]*point[1]
                    
                else:
                    # area+=point[0]*(prev[1]-point[1])
                    # print("c加了%d area\n"%(point[0]*(prev[1]-point[1])))
                    pass
            else:
                if(point[1]<=prev[1]):
                    continue
                else:
                    area+=((point[1]-prev[1])*point[0])
                    # print("d加了%d area\n"%((point[1]-prev[1])*point[0]))
                    prev=point

    
    return hdiff*area
    
    
    pass




def cal_vol(ppa_list):
    sorted_list=ppa_list[ppa_list[:, 2].argsort()[::-1]]
    cnt=0
    vol=0
    flag=0
    point_num=len(sorted_list)
    for point in sorted_list:
        if (cnt!=point_num-1):
            if(point[2]==sorted_list[cnt+1][2]):
                flag=1
                cnt+=1 
                continue
            
                

            else:
                hdiff=point[2]-sorted_list[cnt+1][2]
                if(flag==1):
                    vol+=sweep2d_vol(np.delete(sorted_list[:cnt+1,:],2,1),hdiff)
                    # print("now vol is %d\n"% vol)
                    flag=0
                else:
                    # print(sorted_list[:,:cnt+1],'\n')
                    vol+=sweep2d_vol(np.delete(sorted_list[:cnt+1,:],2,1),hdiff)  
                    # print("now vol is %d\n"% vol)  
            
        else:
            vol+=sweep2d_vol(np.delete(sorted_list,2,1),point[2])
            # print("now vol is %d\n"% vol)
            pass    
        cnt+=1    
        
    return vol



def constructVolumn(ppa_list,vref):
    
    
    
    
    
    
    pass
    
def checkVolumn(self,in_pa): #return the point's influence to area
    pass


def updateVolumn(self,in_pa):
    pass

if __name__ == "__main__" :
    test_case= np.array([[1,1,2],[2,1,1],[1,2,1]])
    print(test_case[test_case[:, 2].argsort()])
    print(np.delete(test_case,2,1))
    print("******分隔**********\n")
    print(test_case)
    print(cal_vol(test_case))
    # print("volume is\t",constructVolumn(test_case))
    
    pass
    