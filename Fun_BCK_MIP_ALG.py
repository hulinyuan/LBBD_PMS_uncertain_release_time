import copy
import time
import random
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def Fun_directly_solve( subset_job_index, subset_job_num, release_time_mu, release_time_delta, jobs_num, process_time, current_m_index, Gamma, DT):
    '''
    直接求解SP
    '''
    sp_status = 0
    r_bar = max(release_time_mu + release_time_delta)
    Big_value = r_bar - release_time_mu
    notform = [] 
    for j in range(jobs_num):
        if j not in subset_job_index:
            notform.append(j)
   

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
    cons3 = sp_Model.addConstr( gp.quicksum(v[j] for j in subset_job_index) >= 1 )

    # cons3 = sp_Model.addConstr( t <= r_bar)
    # Add constraint
    cons4 = sp_Model.addConstr( gp.quicksum(u[j] for j in range(jobs_num)) <= Gamma )
    # Add constraint
    cons5 = sp_Model.addConstrs( u[j] <= release_time_delta[j] for j in range(jobs_num))

    # set  parameters
    sp_Model.Params.OutputFlag = 0
    sp_Model.Params.TimeLimit = 600
    
    sp_Model.optimize()
    if sp_Model.status == GRB.Status.INFEASIBLE:
        print('问题不可行')
    else:  
        obj = sp_Model.getObjective()
        obj_val = obj.getValue()
        print(f'目标函数为 {obj_val}')
        cur_rt_u = np.zeros(jobs_num)
        stand_time = t.getAttr(GRB.Attr.X)
        for j in range(jobs_num):
            if u[j].getAttr(GRB.Attr.X) - 0 >= 0.00001:
                cur_rt_u[j] = u[j].getAttr(GRB.Attr.X)
        
        # 划分集合
        J_plus = list()
        J_equal = list()
        J_minus = list()
       
        current_release_time  = cur_rt_u + release_time_mu
            # 释放时间从小到大排
        subset_job_release_time_dict = dict()     # 工件序号：释放时间        
        for j in subset_job_index:
            subset_job_release_time_dict.update({j:current_release_time[j]})
            subset_job_position_index_map = sorted(subset_job_release_time_dict.items(), key=lambda x: x[1])

        # 寻找关键工件
        job_position_cmax = np.zeros(subset_job_num) 

        for i in range(subset_job_num):
            job_position_cmax[i] = subset_job_position_index_map[i][1] # 释放时间
            for k in range(i, subset_job_num):
                job_idx = subset_job_position_index_map[k][0] 
                job_position_cmax[i] = job_position_cmax[i] + process_time[current_m_index, job_idx] # 加工时间
        
          
        key_pos_index = np.argmax(job_position_cmax) # 只会返回最大值中下标最小的那个
        
        h_t_u = max(job_position_cmax)     # 记录当前最大完工时间
       
        # 得到最优排序下释放时间最小的关键工件序号
        key_job_idx =  subset_job_position_index_map[key_pos_index][0]
        mini_rt_key_job = key_job_idx
        mini_rt = current_release_time[key_job_idx]

        for i in range(key_pos_index + 1, subset_job_num):
            if job_position_cmax[i] == job_position_cmax[key_pos_index]:
                key_job_idx  = subset_job_position_index_map[i][0]
                cur_rt =  current_release_time[key_job_idx]
                if cur_rt <=  mini_rt:
                    mini_rt_key_job  = key_job_idx

        stand_time = current_release_time[mini_rt_key_job] 
        
        # 划分集合
        J_plus = list()
        J_equal = list()
        J_minus = list()
        cur_rt_u = np.zeros(jobs_num) # 对所有工件来进行设置的

        for j in subset_job_index: 
            if release_time_mu[j] >= stand_time:
                J_plus.append(j)
            if (release_time_mu[j] < stand_time) and (current_release_time[j] >=  stand_time):
                cur_rt_u[j] = stand_time - release_time_mu[j]  #只对这些工件赋值
                J_equal.append(j)
            if current_release_time[j] < stand_time:
                J_minus.append(j)
        
        if obj_val > DT:
            print(f'子问题不可行')
            sp_status  = 1
            
    return sp_status, obj_val, J_plus, J_equal, stand_time


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
            if bar_y[m,j]> 0.9:
                machine_job_set[m].append(j)  
    
    for m_idx in range(machines_num):
        subset_job_index = machine_job_set[m_idx] # 机器分配到工件的下标集合
        subset_job_num  = len(subset_job_index)
        if subset_job_num > 0: 
            sp_start = time.time()
            sp_num = sp_num + 1 # 直接求解MIP问题的版本
            sp_status, obj_val, J_plus, J_equal, stand_time = Fun_directly_solve(subset_job_index, subset_job_num, release_time_mu, release_time_delta, jobs_num, process_time, m_idx, Gamma, DT)
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
                if cb_cuts[4] == 1:  # print(f'添加割：{sp_status}')
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

        status, Valid_Inequality = Fun_Feasibility_Check(y, bar_x, bar_y, DT1, cb_cuts1, release_time_delta1, release_time_mu1,process_time1, Gamma1, jobs_num1, machines_num1)
        
        
        if status == 1: # 指定1为添加cuts的变量
            for cut in Valid_Inequality:
                bk_cuts  =  bk_cuts + 1 # 记录添加bkcuts的数量
                model.cbLazy(cut >=0 ) # print({'cuts':linexpr}) 
                    
                    
def BCK_MIP_main(jobs_num, machines_num, jobtabu, cost, process_time, release_time_mu,
                  release_time_delta, Gamma, DT, cb_cuts, max_iter, max_time):
    global  sp_time, sp_num, cut_num, x, y, DT1, cb_cuts1, release_time_delta1, release_time_mu1, process_time1, Gamma1, jobs_num1, machines_num1
    
    DT1 = DT
    cb_cuts1 = cb_cuts
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

    # 决策变量下标
    m_index = dict()
    for m in range(machines_num):
        m_index.update({m:0})
    jm_index = dict()
    for m in range(machines_num):
        for j in range(jobs_num):
            jm_index.update({(m,j):0})
            
    # 主问题   
    mp_Model = gp.Model("MP")
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
    mp_Model.Params.OutputFlag = 0
    # mp_Model.Params.Cuts = 0
    # mp_Model.Params.NodeMethod = 0
    # mp_Model.Params.MIPGap = 0.0001
    # mp_Model.Params.Heuristics = 0
    mp_Model.setParam(GRB.Param.TimeLimit, max_time)
    mp_Model.Params.LazyConstraints = 1
    mp_Model.optimize(Fun_Add_Cuts)

    status = mp_Model.status
    if mp_Model.status == GRB.Status.INFEASIBLE:
        print('原问题不可行')
    else:  
        '''

        for m in range(machines_num):
            print(f'机器：{m}',end=':')
            for j in range(jobs_num):
                if y[m,j].getAttr(GRB.Attr.X) >  0.5:
                    print(j, end=',')
            print()
        '''
        mp_tm = total_tm - sp_tm    
        gap = mp_Model.getAttr(GRB.Attr.MIPGap)
        obj_expr = mp_Model.getObjective()
        obj_val = obj_expr.getValue()
        opt_x = mp_Model.getAttr("X", x)
        opt_y = mp_Model.getAttr("X", y)
        rel_tm = sp_tm     
        rel_num = sp_num
        itr_num = mp_Model.getAttr(GRB.Attr.NodeCount) 


        print(f'========================')
        print(f'最优目标函数值：{obj_val}')
        print(f'添加割数量：{bk_cuts}')
        print(f'算法总用时：{total_tm}')
        print(f'整数节点个数：{int_num}')
        print(f'========================')

    return status, obj_val, opt_x, opt_y, gap, total_tm, mp_tm, sp_tm, mip_tm, rel_tm, gre_tm, itr_num, sp_num, int_num, fra_num, mip_num, rel_num, gre_num, bk_cuts, bc_cuts