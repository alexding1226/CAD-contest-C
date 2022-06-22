from operator import itemgetter
import random
from termios import VEOL
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




#計算predict point 貢獻的pvol
def cal_eipv(non_dominated_cell_list,predict_point):
    vol=0
    for cell in non_dominated_cell_list:
        if ((predict_point[0]>cell[0]) and (predict_point[1]>cell[1]) and (predict_point[2]>cell[2])):
            vol+=(max(predict_point[0],cell[3])-cell[0])*(max(predict_point[1],cell[4])-cell[1])*(max(predict_point[2],cell[5])-cell[2])
            
            
    return vol
            





# pareto_list is n points represented by np.array with shape(n,3) ，h_bnd is the max_bound

# cell的表示法為  有6個element 的list ex: [0,0,0,1,1,1]  表示一個體積為1的cell
def gen_non_dominated_cell(pareto_list,h_bnd):
    x_list=[0,h_bnd]
    y_list=[0,h_bnd]
    z_list=[0,h_bnd]
    for pt in pareto_list:
        if (pt[0] not in x_list):
            x_list.append(pt[0])
        if (pt[1] not in y_list):
            y_list.append(pt[1])
            
        if (pt[2] not in z_list):
            z_list.append(pt[2])
            
    # x.sort(reverse = True)
    x_list.sort()
    y_list.sort()
    z_list.sort()
    all_cell=[];
    # gen all cell
    for x_idx,x in enumerate(x_list) :
        if(x!=h_bnd):
            for y_idx,y in enumerate(y_list):
                if(y!=h_bnd):
                    for z_idx,z in enumerate(z_list):
                        if((z!=h_bnd)):
                            all_cell.append([x,y,z,x_list[x_idx+1],y_list[y_idx+1],z_list[z_idx+1]])
                            pass
                            
                        else:
                            continue
                else:
                    continue
        else:
            continue
    non_dominated_cell_list=[]
    for cell in all_cell:
        flag=0
        for pt in pareto_list:
            if((cell[3]<=pt[0]) and (cell[4]<=pt[1]) and (cell[5]<=pt[2])):
                flag=1
                break
            else:
                continue
            
        if(flag):
            continue
            
        else:
            non_dominated_cell_list.append(cell)
    return non_dominated_cell_list






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
    test_case= np.array([[1,1,4],[1,5,3],[3,3,2]])
    # print(test_case[test_case[:, 2].argsort()])
    # print(np.delete(test_case,2,1))
    print("******分隔**********\n")
    # print(test_case)
    # print(cal_vol(test_case))
    # print("volume is\t",constructVolumn(test_case))
    print(gen_non_dominated_cell(np.array([[1,2,2],[2,1,1]]),5))
    
    pass
    