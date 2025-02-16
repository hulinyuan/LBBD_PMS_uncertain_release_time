
import copy
import time
import random
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def relaxing_solution(subset_job_index, subset_job_num, release_time_mu, process_time, release_time_delta, jobs_num, current_m_index):
    '''
    松弛分配算法
    '''
    current_release_time = release_time_delta + release_time_mu
    # 释放时间从小到大排
    subset_job_release_time_dict = dict()     # 工件序号：释放时间        
    for j in subset_job_index:
        subset_job_release_time_dict.update({j:current_release_time[j]})
        subset_job_position_index_map = sorted(subset_job_release_time_dict.items(), key=lambda x: x[1])
    
    # 寻找关键工件
    job_position_cmax = np.zeros(subset_job_num) 
    
    for i in range(subset_job_num): # 计算每个位置对应的 rj + sum(k>=i) pj
        job_position_cmax[i] = subset_job_position_index_map[i][1]    # 第一个位置
        for k in range(i, subset_job_num):
            job_idx = subset_job_position_index_map[k][0] 
            job_position_cmax[i] = job_position_cmax[i] + process_time[current_m_index, job_idx] # 加工时间
            
    key_pos_index = np.argmax(job_position_cmax) # 只会返回最大值中下标最小的那个
    h_t_u = job_position_cmax[key_pos_index]     # 记录当前最大完工时间

    # 得到最优排序下释放时间最小的关键工件序号
    key_job_idx = subset_job_position_index_map[key_pos_index][0] # 关键工件的定义
    # 万一有多个关键工件，就要找释放时间均值最小的工件
    mini_rt_key_job = key_job_idx
    mini_rt = current_release_time[key_job_idx]
    for i in range(subset_job_num):
        if job_position_cmax[i] == job_position_cmax[key_pos_index]:
            key_job_idx = subset_job_position_index_map[i][0]
            cur_rt = current_release_time[key_job_idx]
            if cur_rt <= mini_rt:
                mini_rt_key_job = key_job_idx

    stand_time = current_release_time[mini_rt_key_job] 
    # 划分集合
    J_plus = list()
    J_equal = list()
    J_minus = list()
    cur_rt_u = np.zeros(jobs_num) # 对所有工件来进行设置的
    for j in subset_job_index: 
        if release_time_mu[j] >= stand_time: # 均值大于标准时间
            J_plus.append(j)
        if (release_time_mu[j] < stand_time) and (current_release_time[j] >=  stand_time):
            cur_rt_u[j] = stand_time - release_time_mu[j]  #只对这些工件赋值
            J_equal.append(j)
        if current_release_time[j] < stand_time: 
            J_minus.append(j)
    # 这个代码和文档一致
    return cur_rt_u, stand_time, J_plus, J_equal, J_minus, h_t_u

def Fun_iter_update_method(subset_job_index, subset_job_num, release_time_mu, release_time_delta, jobs_num, process_time, m_idx, Gamma, DT):
    '''
    迭代改进算法
    '''
    stop_flag  = 0
    # 最差情景下对应的最小最大完工时间
    h_t_u  = 0 
    job_hat = subset_job_index
    subset_job_num = len(job_hat)
    cur_rt_u  = release_time_delta 
    sp_status = 0  # 子问题求解状态
    # 松弛分配算法，得到工件集合
    cur_rt_u, stand_time, J_plus, J_equal, J_minus, h_t_u = relaxing_solution(subset_job_index, subset_job_num, release_time_mu, process_time, release_time_delta, jobs_num, m_idx)
    J_geq = J_equal + J_plus  # 合并一个新的集合，后面会要用，每次都记得更新一下 
    while stop_flag == 0:
        # 计算赤字程度
        defici_delta = sum(cur_rt_u) - Gamma # 赤字程度
        if  defici_delta <= 0:
            # 无赤字，停止循环

            # print(f'找到可行候选解')
            h_t_u = stand_time
            for j in J_geq:
                    h_t_u =  h_t_u + process_time[m_idx, j]
                    
            # 返回当前的释放时间增量，和对应的最小最大完工时间
            if h_t_u > DT: 
                sp_status  = 1 #子问题不可行
            
            stop_flag  = 1 
            return sp_status, h_t_u, J_equal, J_plus, stand_time, cur_rt_u
        else:  
            # ===== 参数 0 ===================
            redu_job_num = len(J_equal)
            unit_reduction = defici_delta/redu_job_num
            # print(f'epsilon_0:{unit_reduction}')

            # ===== 参数 1 ====================
            j_value_for_e1 = np.zeros(jobs_num)
            for j in J_minus: # 外层j
                j_value_for_e1[j] = release_time_mu[j] 
                for s in J_minus: # 内层 s
                    if release_time_mu[s] >= release_time_mu[j]: # 找出
                        j_value_for_e1[j] = j_value_for_e1[j] + process_time[m_idx, s] 
            epsilon_1 = stand_time - max(j_value_for_e1)
            # print(f'epsilon_1:{epsilon_1}')

            # ===== 参数 2 ====================
            j_value_for_e2 = [cur_rt_u[j] for j in J_equal]
            epsilon_2 = min(j_value_for_e2)
            # print(f'epsilon_2:{epsilon_2}')

            # ===== 参数 3 ====================
            '''
            有可能存在J_plus
            '''
            if len(J_plus) == 0:
                # print(f'J_plus为空集')
                epsilon_3 = DT
            else:
                j_value_for_e3 = [stand_time - release_time_mu[j]  for j in J_plus]
                for idx in range(len(J_plus)):
                    j = J_plus[idx]
                    for s in J_geq:
                        s_end_time = release_time_mu[s] + process_time[m_idx, s] 
                        if s_end_time < release_time_mu[j]:
                            j_value_for_e3[idx] = j_value_for_e3[idx] + process_time[m_idx, s] 

                epsilon_3 = min(j_value_for_e3)
            # print(f'epsilon_3:{epsilon_3}')

        case_index =  np.argmin([unit_reduction, epsilon_1, epsilon_2, epsilon_3])

        if case_index == 0:
            # print(f'进入情形0，返回可行候选解')
            stand_time = stand_time - unit_reduction
            for j in J_equal:
                cur_rt_u[j]  = cur_rt_u[j] - unit_reduction
                
            print(cur_rt_u)

        if case_index == 1:
            # print(f'进入情形1, 更新候选解, 下面检查是否可行')
            # 情形 1
            j_value_1 = np.zeros(jobs_num)
            for j in J_minus:
                j_value_1[j] = release_time_mu[j] 
                for s in J_minus:
                    if release_time_mu[s] >= release_time_mu[j]:
                        j_value_1[j] = j_value_1[j] + process_time[m_idx, s]

            max_value = max(j_value_1)

            min_j = np.argmin(j_value_1)
            for s in J_minus: # 要找到最小释放时间的最大值的下标
                if max_value == j_value_1[s]:
                    if  release_time_mu[s] < release_time_mu[min_j] :
                        min_j = s

            stand_time = release_time_mu[min_j] # mu_j1
            for j in J_equal:
                cur_rt_u[j] = max(stand_time - release_time_mu[j], 0)

            # 如果需要再更新集合
            J_plus = list()
            J_equal = list()
            J_minus = list()
            for j in subset_job_index:
                if cur_rt_u[j] > 0:
                    J_equal.append(j)
                else: # 等于 0
                    if release_time_mu[j] < stand_time:
                        J_minus.append(j)
                    if  release_time_mu[j] >= stand_time:
                        J_plus.append(j) 

        if case_index == 2:
            # print(f'进入情形2, 更新候选解, 下面检查是否可行')
            stand_time = stand_time - epsilon_2
            # 如果需要再更新集合，这里要看看是不是所有的工件都不重复的分配了
            for j in J_equal:
                cur_rt_u[j] = cur_rt_u[j] - epsilon_2 
            # 更新 集合
            J_plus = list()
            J_equal = list()      
            for j in subset_job_index:
                if cur_rt_u[j] > 0:
                    J_equal.append(j)
                else: # 等于0 
                    if release_time_mu[j] >= stand_time:
                        J_plus.append(j) 

        if case_index == 3:
            # print(f'进入情形3, 重新进入松弛分配')
            j_value_1 = np.zeros(jobs_num)
            for j in J_minus:
                j_value_1[j] = release_time_mu[j] 
                for s in J_minus:
                    if release_time_mu[s] >= release_time_mu[j]:
                        j_value_1[j] = j_value_1[j] + process_time[m_idx, s]

            max_value = max(j_value_1)

            min_j = np.argmin(j_value_1)
            for s in J_minus: # 要找到最小释放时间的最大值的下标
                if max_value == j_value_1[s]:
                    if  release_time_mu[s] < release_time_mu[min_j] :
                        min_j = s

            job_hat = list() 
            for j in subset_job_index:
                if release_time_mu[min_j] >= release_time_mu[j]:
                    job_hat.append(j)  
            
            sp_status, h_t_u, J_equal, J_plus, stand_time, cur_rt_u = Fun_iter_update_method(subset_job_index, subset_job_num, release_time_mu, release_time_delta, jobs_num, process_time, m_idx, Gamma, DT)
            
    return sp_status, h_t_u, J_equal, J_plus, stand_time, cur_rt_u



def Fun_Feasibility_Check(y, bar_x, bar_y, DT, cb_cuts, release_time_delta, release_time_mu, process_time, Gamma, jobs_num, machines_num):
    '''
    检查当前分配下是否满足
    '''
    global sp_num, sp_tm

    status = 0 # 默认可行
    Valid_Inequality = list()

   
    machine_job_set = dict() # 把每台机器的工件分配记录下来
    for j in range(jobs_num):
        for m in range(machines_num):
            if m not in machine_job_set.keys():
                machine_job_set.update({m:[]})
            if bar_y[m,j]> 0:
                machine_job_set[m].append(j)  
    
    for m_idx in range(machines_num):
        subset_job_index = machine_job_set[m_idx] # 机器分配到工件的下标集合
        subset_job_num  = len(subset_job_index)
        if subset_job_num > 0: 
            sp_start = time.time()
            sp_num = sp_num + 1 # 直接求解MIP问题的版本
            sp_status, h_t_u, J_equal, J_plus, stand_time, cur_rt_u = Fun_iter_update_method(subset_job_index, subset_job_num, release_time_mu, release_time_delta, jobs_num, process_time, m_idx, Gamma, DT)
            J_geq =  J_equal + J_plus
            if sp_status == 1: # 子问题不可行
                status = 1
                #==========Cb cuts 1===================
                if cb_cuts[0] == 1:# print(f'添加割：{sp_status}')
                    vi = subset_job_num - 1 -  gp.quicksum(y[m_idx,j] for j in subset_job_index)  
                    Valid_Inequality.append(vi)

                #==========Cb cuts 2===================
                if cb_cuts[1] == 1: # print(f'添加割：{sp_status}')
                    vi = len(J_geq) - 1 -  gp.quicksum(y[m_idx,j] for j in J_geq)  
                    Valid_Inequality.append(vi)

                    #==========Cb cuts 3===================
                if cb_cuts[2] == 1: # print(f'添加割：{sp_status}')
                    J_hat_1 = []
                    max_pt = max([ process_time[m_idx, j] for j in J_geq])

                    for j in range(jobs_num):
                        if (j not in J_geq):
                            if (release_time_mu[j] >= stand_time) and (process_time[m_idx, j] >= max_pt):
                                J_hat_1.append(j)

                    J_sum = J_hat_1 + J_geq
                    vi = len(J_geq) - 1 -  gp.quicksum(y[m_idx, j] for j in J_sum)  
                    Valid_Inequality.append(vi)

                #==========Cb cuts 4===================
                if cb_cuts[3] == 1: # print(f'添加割：{sp_status}')
                    J_hat_1_2 = []
                    max_pt = max([ process_time[m_idx, j] for j in J_geq])
                    for j in range(jobs_num):
                        if (j not in J_geq):
                            if (release_time_mu[j] >= stand_time) and (process_time[m_idx, j] >= max_pt):
                                J_hat_1_2.append(j)
                            if (release_time_mu[j] < stand_time) and (process_time[m_idx, j] + release_time_mu[j] >= stand_time + max_pt):
                                J_hat_1_2.append(j)

                    J_sum = J_hat_1_2 + J_geq
                    vi = len(J_geq) - 1 -  gp.quicksum(y[m_idx,j] for j in J_sum)  
                    Valid_Inequality.append(vi)

                #==========Cb cuts 5===================
                if cb_cuts[4] == 1: # print(f'添加割：{sp_status}')
                    J_hat_1 = []
                    J_hat_2 = []
                    for j in range(jobs_num):
                        if (j not in J_geq):
                            if (release_time_mu[j] >= stand_time):
                                J_hat_1.append(j)
                        else:
                            J_hat_1.append(j)
                        if  (release_time_mu[j] + process_time[m_idx, j] >= stand_time )  and  (stand_time > release_time_mu[j] ):
                            J_hat_2.append(j)
                    for j_hat in J_equal:
                        vi = DT - stand_time * y[m_idx, j_hat] - gp.quicksum( process_time[m_idx, j] * y[m_idx,j] for j in J_hat_1) - gp.quicksum( (release_time_mu[j] + process_time[m_idx, j] - stand_time ) * y[m_idx,j] for j in J_hat_2 )
                        Valid_Inequality.append(vi)
                # else:
                    # print(f'子问题可行')

            sp_tm = sp_tm + time.time() - sp_start

    return status, Valid_Inequality

def Fun_Add_Cuts(model, where):  # callback函数
    '''
    自定义的callback 函数主问题求解过程中遇到整数解后执行check
    '''
    global  sp_tm, sp_num, int_num, rel_tm, rel_num,  bk_cuts

    global  x, y, DT1, cb_cuts1, release_time_delta1, release_time_mu1, process_time1, Gamma1, jobs_num1, machines_num1
    
    if where == GRB.Callback.MIPSOL:
        # MIP solution callback
        bar_x = model.cbGetSolution(x)  # 字典 {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0}
        bar_y = model.cbGetSolution(y) 
        int_num = int_num + 1 # 记录遇到整数节点的个数
        status, Valid_Inequality = Fun_Feasibility_Check(y, bar_x, bar_y, DT1, cb_cuts1, release_time_delta1, release_time_mu1, 
                                                                           process_time1, Gamma1, jobs_num1, machines_num1)
        if status == 1: # 指定1为添加cuts的变量
            for cut in Valid_Inequality:
                bk_cuts  =  bk_cuts + 1 # 记录添加bkcuts的数量
                model.cbLazy(cut >=0 ) # print({'cuts':linexpr}) 

def BCK_HER_main(jobs_num, machines_num, jobtabu, cost, process_time, release_time_mu, release_time_delta, Gamma, DT, cb_cuts, max_iter, max_time):
    
    global release_time_mu1, process_time1, Gamma1, jobs_num1, machines_num1, x, y, DT1, cb_cuts1, release_time_delta1
    
    cb_cuts1 = cb_cuts
    DT1 = DT
    release_time_delta1 = release_time_delta
    release_time_mu1 = release_time_mu
    process_time1 = process_time
    Gamma1 = Gamma
    jobs_num1 = jobs_num 
    machines_num1 = machines_num
    # ============================================================================
    global  sp_tm, sp_num, int_num, rel_tm, rel_num,  bk_cuts

    status = 0   # 算例是否可行
    
    obj_val =  0   # 目标函数值
    opt_x = 0      # 最优解
    opt_y = 0      # 最优解
    gap = 0        # 精度
    total_tm  = 0  
    mp_tm = 0

    sp_tm = 0 
    mip_tm = 0  # bk EAc有
    rel_tm = 0  # bk有
    gre_tm = 0  # bc 有
    
    itr_num = 0 # bk bc 算法没有
    sp_num = 0  
    int_num = 0 # bk有
    fra_num = 0 # bc 有
    mip_num = 0 # bk EAC有
    rel_num = 0 # bc bk 有
    gre_num = 0 # bc 有
    
    bk_cuts = 0 # bk 有
    bc_cuts = 0 # bc 有

    alg_start = time.time()
    m_index = dict() # 决策变量下标
    for m in range(machines_num):
        m_index.update({m:0})
    jm_index = dict()
    for m in range(machines_num):
        for j in range(jobs_num):
            jm_index.update({(m,j):0})
            
   
    mp_Model = gp.Model("MP") # 主问题   
    x = mp_Model.addVars(m_index, lb=0, ub=1, vtype=GRB.BINARY,name="x")
    y = mp_Model.addVars(jm_index, lb=0, ub=1, vtype=GRB.BINARY,name="y")

    # set objective
    mp_Model.setObjective(gp.quicksum(cost[m]*x[m]for m in range(machines_num)), GRB.MINIMIZE)
    # Add constraint
    cons1 = mp_Model.addConstrs(y.sum('*',j) == 1 for j in range(jobs_num))
    # Add constraint
    cons2 = mp_Model.addConstrs( gp.quicksum(y[m,j] for j in jobtabu[m]) == 0 for m in range(machines_num))
    # Add constraint
    cons3 = mp_Model.addConstrs( y[m,j] <= x[m] for m in range(machines_num)for j in range(jobs_num))

    # set  parameters
    mp_Model.Params.OutputFlag = 1
    # mp_Model.Params.Cuts = 0
    # mp_Model.Params.NodeMethod = 0
    # mp_Model.Params.MIPGap = 0.0001
    # mp_Model.Params.Heuristics = 0
    mp_Model.setParam(GRB.Param.TimeLimit, max_time)
    mp_Model.Params.LazyConstraints = 1
    mp_Model.optimize(Fun_Add_Cuts)

    if mp_Model.status == GRB.Status.INFEASIBLE:
        status = -1
        print('问题不可行')
    else:  
        if mp_Model.status == GRB.Status.OPTIMAL:
            status = 1
            for m in range(machines_num):
                print(f'机器：{m}', end=':')
                for j in range(jobs_num):
                    if y[m,j].getAttr(GRB.Attr.X) >  0:
                        print(j, end=',')
                print()

            total_tm = time.time() - alg_start
            mp_tm = total_tm - sp_tm    
            gap = mp_Model.getAttr(GRB.Attr.MIPGap)
            obj_expr = mp_Model.getObjective()
            obj_val = obj_expr.getValue()
            opt_x = mp_Model.getAttr("X", x)
            opt_y = mp_Model.getAttr("X", y)
            rel_tm = sp_tm     
            rel_num = sp_num
            itr_num = mp_Model.getAttr(GRB.Attr.NodeCount) 
    
    return status, obj_val, opt_x, opt_y, gap, total_tm, mp_tm, sp_tm, mip_tm, rel_tm, gre_tm, itr_num, sp_num, int_num, fra_num, mip_num, rel_num, gre_num, bk_cuts, bc_cuts