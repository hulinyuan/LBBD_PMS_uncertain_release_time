import copy
from telnetlib import STATUS
import time
import random
import numpy as np
import gurobipy as gp
from gurobipy import GRB
    
def Fun_directly_solve( subset_job_index, subset_job_num, release_time_mu, release_time_delta, jobs_num, process_time, current_m_index, Gamma, DT, Scenario_Set, infeasible_flag):
    '''
    直接求解SP
    '''
    global start_time, MAX_time
    # print(f'机器：{current_m_index}, 分配到工件数：{subset_job_num}')    

    Big_value = max(release_time_mu + release_time_delta) - release_time_mu
    notform = [] 
    for j in range(jobs_num):
        if j not in subset_job_index:
            notform.append(j)
            
    # print(f'不可加工工件数：{len(notform)}')
    sp_Model = gp.Model("SP")
    t = sp_Model.addVar( lb=0, vtype=GRB.CONTINUOUS, name="t")
    v = sp_Model.addVars(jobs_num, lb=0, ub=1, vtype=GRB.BINARY, name="v")
    u = sp_Model.addVars(jobs_num, lb=0, vtype=GRB.CONTINUOUS, name="u")

    # set objective
    sp_Model.setObjective(t + gp.quicksum(process_time[current_m_index, j] * v[j] for j in subset_job_index), GRB.MAXIMIZE)
    # Add constraint
    cons1 = sp_Model.addConstrs( t <= release_time_mu[j] + u[j] + Big_value[j]*(1 - v[j]) for j in subset_job_index)
    # Add constraint
    cons2 = sp_Model.addConstrs( v[j] == 0 for j in notform )
    # 至少要选一个工件 （新增，缺少这个约束，目标函数值不对）
    cons3 = sp_Model.addConstr( gp.quicksum(v[j] for j in subset_job_index) >=1 )
    # Add constraint
    cons4 = sp_Model.addConstr( gp.quicksum(u[j] for j in range(jobs_num)) <= Gamma )
    # Add constraint
    cons5 = sp_Model.addConstrs( u[j] <= release_time_delta[j] for j in range(jobs_num))

    # set  parameters
    sp_Model.Params.OutputFlag = 0
    already_used = time.time() -  start_time
    timelimit = max(180, MAX_time - already_used)
    sp_Model.Params.TimeLimit = timelimit
    sp_Model.optimize()
    if sp_Model.status != GRB.Status.INFEASIBLE:
        obj = sp_Model.getObjective()
        obj_val = obj.getValue()
        cur_rt_u = np.zeros(jobs_num)
        stand_time = t.getAttr(GRB.Attr.X)
        for j in range(jobs_num):
            if u[j].getAttr(GRB.Attr.X) - 0 >= 0.00001:
                cur_rt_u[j] = u[j].getAttr(GRB.Attr.X)
    
        if obj_val > DT:
            infeasible_flag  = 1
            u_s_index = len(Scenario_Set)
            Scenario_Set.update({u_s_index:cur_rt_u})
            
    return Scenario_Set, infeasible_flag

'''
主模块
'''
def CCG_main(jobs_num, machines_num, jobtabu, cost, process_time, release_time_mu, release_time_delta, Gamma, DT, max_iter, max_time):
    status = 0   # 算例是否可行
    
    obj_val =  0   # 最优目标函数值
    opt_x = 0      # 最优解
    opt_y = 0      # 最优解
    gap = 0        # 精度
    
    total_tm  = 0  
    mp_tm = 0
    sp_tm = 0
    mip_tm = 0
    rel_tm = 0
    gre_tm = 0
    
    itr_num = 0 # 外层迭代计数
    sp_num = 0
    int_num = 0
    fra_num = 0 
    mip_num = 0
    rel_num = 0
    gre_num = 0
    
    bk_cuts = 0
    bc_cuts = 0
    
    MP_LB = 0
    MP_UB = 10**8    
 # ===========================================================
    global start_time, MAX_time
    MAX_time = max_time
    start_time = time.time()  # 算法计时
    # 初始化场景集合 
    Scenario_Set = dict()
    scenario_index = 0
    current_scenario = np.zeros(jobs_num)
    Scenario_Set.update({scenario_index:current_scenario})

    # 决策变量下标
    ijm_index = dict()
    im_index = dict()
    for i in range(jobs_num):
        for j in range(jobs_num):
            for m in range(machines_num):
                ijm_index.update({(i,j,m):0})
                im_index.update({(i,m):0})
    z = {}
    C = {}
    # 松弛主问题   
    RMP_Model = gp.Model("RMP")
    x = RMP_Model.addVars(machines_num, lb=0, ub=1, vtype=GRB.BINARY,name="x")
    y = RMP_Model.addVars(machines_num, jobs_num, lb=0, ub=1, vtype=GRB.BINARY,name="y")
    z[scenario_index] = RMP_Model.addVars(ijm_index, lb=0, ub=1, vtype=GRB.BINARY, name=f"z_{scenario_index}")
    C[scenario_index] = RMP_Model.addVars(im_index, lb=0, vtype=GRB.CONTINUOUS, name=f"C_{scenario_index}")

    # 最小化开机成本
    obj_expr = gp.quicksum(cost[m]*x[m]for m in range(machines_num))

    RMP_Model.setObjective(obj_expr, GRB.MINIMIZE)

    # 每个工件都要有一台机器加工
    cons1 = RMP_Model.addConstrs(y.sum('*',j) == 1 for j in range(jobs_num))

    # 机器不可加工的工件不能分配给机器
    cons2 = RMP_Model.addConstrs( gp.quicksum(y[m,j] for j in jobtabu[m]) == 0 for m in range(machines_num))

    # 开机的机器才能够分配到工件
    cons3 = RMP_Model.addConstrs( y[m,j] <= x[m] for m in range(machines_num) for j in range(jobs_num))

    # 第一个场景相关的约束
    RMP_Model.addConstrs(gp.quicksum(z[scenario_index][i,j,m]for i in range(jobs_num)) == y[m,j] for j in range(jobs_num) for m in range(machines_num))

    RMP_Model.addConstrs(gp.quicksum(z[scenario_index][i,j,m]for j in range(jobs_num))  <= x[m] for i in range(jobs_num) for m in range(machines_num))

    RMP_Model.addConstrs(C[scenario_index][i,m] >= C[scenario_index][i-1,m] + gp.quicksum(z[scenario_index][i,j,m]*process_time[m,j] for j in range(jobs_num))
                        for i in range(1, jobs_num) for m in range(machines_num))

    RMP_Model.addConstrs(C[scenario_index][i,m] >=  gp.quicksum(z[scenario_index][i,j,m]*(release_time_mu[j] + current_scenario[j] + process_time[m,j]) for j in range(jobs_num)) 
                        for i in range(jobs_num) for m in range(machines_num))

    RMP_Model.addConstrs(DT >=  C[scenario_index][jobs_num -1, m] for m in range(machines_num))

    RMP_Model.setParam(GRB.Param.OutputFlag, 1)
    
    print(f'主问题初始化用时：{time.time() - start_time}')
   
    already_used = time.time() -  start_time
    timelimit = max(0, max_time - already_used)
    
    RMP_Model.setParam(GRB.Param.TimeLimit, timelimit)
    RMP_Model.optimize()  

    STOP  = 0 # 外层迭代停止标记

    # check and return result
    if RMP_Model.status == GRB.Status.INFEASIBLE:
        # print('1 算例不可行，算法停止')
        status = -1 
        STOP  = 1
    else:  
        MP_LB = max(MP_LB, obj_expr.getValue())
        STOP  = 0 
        # print(f'初始下界:{MP_LB}')

   

    while STOP == 0: # 如果外层迭代停止条件不满足，就进入内层迭代
        bar_obj = obj_expr.getValue()
        bar_x = RMP_Model.getAttr("X", x)
        bar_y = RMP_Model.getAttr("X", y)
        # ================================   SP ============================================================     
        
        print(f'第{itr_num}次外层迭代')
   
        # 把每台机器的工件分配记录下来
        machine_job_set = dict()
        for j in range(jobs_num):
            for m in range(machines_num):
                if m not in machine_job_set.keys():
                    machine_job_set.update({m:[]})
                if bar_y[m,j]> 0:
                    machine_job_set[m].append(j)
      
    
        # 初始化， 记录是否存在子问题不可行，有一个子问题不可行说明 当前 xy下需要增加新场景
        infeasible_flag_for_sp = 0 
        print('sub',end=',')
        # 对每个机器m进行子问题求解
        for m in range(machines_num): 
            if bar_x[m] > 0:
                sp_start = time.time()
                subset_job_index = machine_job_set[m]
                subset_job_num = len(subset_job_index)
                infeasible_flag  = 0 
                Scenario_Set, infeasible_flag = Fun_directly_solve(subset_job_index, subset_job_num, release_time_mu, release_time_delta, jobs_num, process_time, m, Gamma, DT, Scenario_Set, infeasible_flag)
                sp_tm = sp_tm + (time.time() - sp_start) 
                sp_num = sp_num + 1 
                if infeasible_flag == 1: 
                    infeasible_flag_for_sp = 1

        if infeasible_flag_for_sp == 0: #如果子问题都可行
            if bar_obj  < MP_UB:
                opt_x = bar_x   
                opt_y = bar_y
                opt_obj = bar_obj
                MP_UB = bar_obj

        if MP_LB == MP_UB:
            STOP = 1 # 如果上界和下界相等
        else:
            # 增加新的场景
            RMP_Model.reset()
            for new_s in range(scenario_index + 1, len(Scenario_Set)):   
                
                cur_scenario = Scenario_Set[new_s]
                z[new_s] = RMP_Model.addVars(ijm_index, lb=0, ub=1, vtype=GRB.BINARY, name=f"z_{new_s}")
                C[new_s] = RMP_Model.addVars(im_index, lb=0, vtype=GRB.CONTINUOUS, name=f"C_{new_s}")
                # 第 s 场景相关的约束
                RMP_Model.addConstrs(gp.quicksum(z[new_s][i,j,m]for i in range(jobs_num)) == y[m,j] for j in range(jobs_num) for m in range(machines_num))

                RMP_Model.addConstrs(gp.quicksum(z[new_s][i,j,m]for j in range(jobs_num))  <= x[m] for i in range(jobs_num) for m in range(machines_num))

                RMP_Model.addConstrs(C[new_s][i, m] >= C[new_s][i-1,m] + gp.quicksum(z[new_s][i,j,m]*process_time[m,j] for j in range(jobs_num))
                                        for i in range(1,jobs_num) for m in range(machines_num))

                RMP_Model.addConstrs(C[new_s][i, m] >=  gp.quicksum(z[new_s][i,j,m]*(release_time_mu[j] + cur_scenario[j] +  process_time[m,j]) for j in range(jobs_num)) 
                                        for i in range(jobs_num)for m in range(machines_num))

                RMP_Model.addConstrs(DT >=  C[new_s][jobs_num-1, m] for m in range(machines_num))

            scenario_index = len(Scenario_Set) - 1
            already_used = time.time() - start_time
            timelimit = max(0, max_time - already_used)
            RMP_Model.setParam(GRB.Param.TimeLimit, timelimit)
            RMP_Model.setParam(GRB.Param.OutputFlag, 1)
            RMP_Model.optimize() 

            if RMP_Model.status == GRB.Status.INFEASIBLE:
                STOP  = 1
            else:  
                MP_LB = max(MP_LB, obj_expr.getValue()) # 更新

        if time.time() - start_time > max_time: # 如果超过时间限制
            STOP = 1

        if itr_num > max_iter:
            STOP = 1  

        itr_num  = itr_num   + 1 #记录外层迭代次数

    total_tm =  time.time() - start_time 
    mp_tm = total_tm - sp_tm
    if RMP_Model.status == GRB.Status.OPTIMAL:
        print(f'=============================')
        print(f'算法总用时：{total_tm}')
        print(f'算法迭代次数：{itr_num}')
        print(f'下界：{MP_LB}')
        print(f'下界：{MP_UB}')
        print(f'GAP:{RMP_Model.getAttr(GRB.Attr.MIPGap)}')
        gap = RMP_Model.getAttr(GRB.Attr.MIPGap)
        obj_val = MP_UB

    return status, obj_val, opt_x, opt_y, gap, total_tm, mp_tm, sp_tm, mip_tm, rel_tm, gre_tm, itr_num, sp_num, int_num, fra_num, mip_num, rel_num, gre_num, bk_cuts, bc_cuts
