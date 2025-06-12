import ising_wrapper as ising
import numpy as np
import pickle as pkl
from multiprocessing import Pool
import os
import time

def run(args):
    """
    处理单个温度点的函数（在子进程中执行）
    
    Args:
        args: 包含(L, t, method)的元组
            L: 系统尺寸
            t: 温度值
            method: 模拟方法 ('cluster' 或 'local')
    
    Returns:
        tuple: (t, model) 处理后的模型数据
    """
    L, t, method = args
    # 创建模型
    model = ising.Ising((L, L), (t,-1.0,-1.0,0.2))
    
    # 根据方法运行模拟
    model.run(Nsample=10000, spacing=5, method=method)
    return model

def run_simulation_group(L, method, temperatures, base_dir="data"):
    """
    运行单个(L, method)组合的模拟（并行处理所有温度点）
    
    Args:
        L: 系统尺寸
        method: 模拟方法
        temperatures: 温度点列表
        base_dir: 数据存储目录
    """
    start_time = time.time()
    print(f"Starting {method}_L{L} with {len(temperatures)} temperatures...")
    
    # 创建任务列表
    tasks = [(L, t, method) for t in temperatures]
    
    # 使用进程池并行处理温度点
    with Pool(processes=14) as pool:
        # 使用imap按顺序获取结果
        results = []
        for i, result in enumerate(pool.imap(run, tasks)):
            model = result
            results.append(model)
    
    # 按温度排序结果
    results.sort(key=lambda x: x.get_couplings[0])  # 按温度排序
    
    # 保存结果到文件
    filename = f"{method}_L{L}.pkl"
    filepath = os.path.join(base_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    
    with open(filepath, "wb") as f:
        pkl.dump(results, f)
        f.close()
    
    print(f"Saved {len(results)} models to {filepath} "
          f"in {time.time()-start_time:.1f} seconds")
    
    # 释放内存
    del results

if __name__ == "__main__":
    # 配置参数
    L_values = [32,48]  # 系统尺寸

    large_search = np.linspace(2, 4, 21)
    fine_search = np.linspace(2.8, 3.1, 21)

    temperatures = np.concatenate((large_search, fine_search))
    temperatures = np.round(np.sort(np.unique(temperatures)), 4)  # 确保温度唯一且排序

    base_dir = "data"  # 输出目录
    
    # 记录总开始时间
    total_start = time.time()
    
    # 按方法和尺寸分组处理
    for l in L_values:
        for method in ["cluster", "local"]:
            run_simulation_group(l, method, temperatures, base_dir)


    # 总耗时报告
    total_time = time.time() - total_start
    print(f"All simulations completed in {total_time:.1f} seconds")
