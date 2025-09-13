import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ga_library.core.algorithm import GeneticAlgorithm
from ga_library.problems.benchmark import BenchmarkProblems


def run_optimization_example():
    print("遗传算法库使用示例")
    print("="*50)
    
    # 创建问题
    problem = BenchmarkProblems.get_problem('sphere', dimensions=5)
    
    # 配置参数
    config = {
        "POP_SIZE": 100,
        "MAX_GEN": 50,
        "CROSSOVER_RATE": 0.8,
        "MUTATION_RATE": 0.1,
        "N_ELITE": 10,
        "VERBOSE": True
    }
    
    # 创建算法实例
    ga = GeneticAlgorithm(problem, config)
    
    # 运行优化
    best_solution = ga.run()
    
    # 输出结果
    print("\n优化结果:")
    print(f"最优解: {best_solution.genes}")
    print(f"最优值: {best_solution.fitness:.6f}")
    print(f"理论最优: 0 (全零向量)")
    
    return ga


if __name__ == "__main__":
    ga_instance = run_optimization_example()