# Project Title: Algorithm Source Code for Robust Parallel Machine Selection and Scheduling Problem with Uncertain Release Times
## Overview
This repository contains the source code for solving a **Parallel Machine Selection and Scheduling** problem with uncertain release times, as proposed in the manuscript *"Robust Parallel Machine Selection and Scheduling with Uncertain Release Times"*. The problem is modeled as a **two-stage robust PMSS model**, where the **release time deviation (RTD)** is characterized by a budget uncertainty set. The goal is to select machines and assign jobs in the first stage to minimize startup costs, and then optimize the job sequence in the second stage to minimize the makespan on each machine after the release times are revealed.

## Code

### Prerequisites

- Python 3.6 or higher
- Required libraries (list any necessary Python libraries):
    - `numpy`
    - `pandas`
    - `matplotlib`
    - `scipy`
    - `cvxpy` 
    - `gurobi`
      
### Code files

- `Fun_BWK_EAC_ALG2.py`: The proposed LBBD-BC algorithm/The variant with upper bound cuts/ The variant with the relaxation-and-correction procedure
- `Fun_CCG_ALG.py`:The CCG algorithm.
- `Fun_CBD_EAC_ALG2.py`:The LBBD-S algorithm.
- `Fun_BWK_MIP_ALG2.py`: The variant without the relaxation-and-correction procedure
- `Fun_BCK_EAC_ALG.py`: The variant without warm-start cuts.
- `Fun_BWK_EAC_ALG.py`: The variant without upper bound cuts.
- `Fun_CCG_ALG.py`:The CCG algorithm.

### Example Code Snippet
Here is a snippet of the algorithm execution for a given instance:
```python
import os
folder_path = "Path/Ins/T530"
file_names = os.listdir(folder_path)
file_name_list = []

for file_name in file_names:
    file_name = folder_path + "\\"+ file_name
    file_name_list.append(file_name)

# Initialize result storage
Result_list_by_ins = {}
result_save_path = 'results.spydata'

# Iterate over all input instances
for filename in file_name_list:
    filenamein = filename[8:].replace('.spydata','')
    print(filename)
    globals().update(load_dictionary(filename)[0])

    Result_list = []
    max_iter = 10000
    max_time = 3600   
    try:
        status, obj_val, opt_x, opt_y, gap, total_tm, mp_tm, sp_tm, mip_tm, rel_tm, warm_time, node_num, sp_num, int_num, fra_num, mip_num, rel_num, gre_num, bk_cuts, wm_cuts = BWK2_EAC_main(jobs_num, machines_num, jobtabu, cost, process_time, release_time_mu, release_time_delta, Gamma, DT, cb_bk_cuts, cbw_cuts, max_iter, max_time)
        Result_list.append(['R1', status, obj_val, gap, total_tm, mp_tm, sp_tm, mip_tm, rel_tm, warm_time, node_num, sp_num, int_num, fra_num, mip_num, rel_num, gre_num, bk_cuts, wm_cuts])
        Result_list_by_ins[filenamein] = Result_list
        save_dictionary({"Result_list_by_ins":Result_list_by_ins}, result_save_path)
    except Exception as e:
        print(e)
```
This will run the algorithm for each input instance and save the results.

### Instances
- `T530.zip`: $\theta = 30$
- `T560.zip`: $\theta = 60$
  
### Others
- `README.md`: This file.
