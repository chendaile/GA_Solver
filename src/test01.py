import numpy as np
import matplotlib.pyplot as plt
from GA_Solver import GA_Solver


def rosenbrock_function(matrix):
    """Rosenbrock函数 - 经典优化测试函数
    全局最优解: f(1,1,...,1) = 0
    """
    x = matrix.flatten()
    return -sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def ackley_function(matrix):
    """Ackley函数 - 多峰函数
    全局最优解: f(0,0,...,0) = 0
    """
    x = matrix.flatten()
    n = len(x)
    a, b, c = 20, 0.2, 2*np.pi
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(c * x)) / n)
    return -(term1 + term2 + a + np.e)


def sphere_function(matrix):
    """球面函数 - 简单凸优化
    全局最优解: f(0,0,...,0) = 0
    """
    return -np.sum(matrix**2)


def rastrigin_function(matrix):
    """Rastrigin函数 - 高度多峰
    全局最优解: f(0,0,...,0) = 0
    """
    x = matrix.flatten()
    A = 10
    n = len(x)
    return -(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def schwefel_function(matrix):
    """Schwefel函数 - 欺骗性函数
    全局最优解: f(420.969,...,420.969) ≈ 0
    """
    x = matrix.flatten()
    return -(-418.9829 * len(x) + np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def test_optimization_problem(name, fitness_func, init_range, optimal_value, dimensions=(10,)):
    print(f"\n{'='*50}")
    print(f"测试问题: {name}")
    print(f"维度: {dimensions}")
    print(f"理论最优值: {optimal_value}")
    print(f"{'='*50}")

    init_matrix = np.random.uniform(init_range[0], init_range[1], dimensions)
    solver = GA_Solver(init_matrix, fitness_func)

    solver.CONFIG.update({"POP_SIZE": 1500,
                          "STANDARD_DEVIATION": 0.5,
                          "N_CROSSOVER": 3,
                          "CROSSOVER_MUTATE_RATE": 0,
                          "ATTENTION_UPDATE_FREQ": 20,
                          "PERTURBATION_SIZE": 1.0,
                          "N_ELITE": 30,
                          "MAX_GEN": 4000,
                          "STAGNATION_THRESHOLD": 100
                          })
    solver.Optimize()

    return solver


def knapsack_problem():
    """0-1背包问题测试"""
    weights = np.array([2, 3, 4, 5, 9, 6, 7, 8])
    values = np.array([1, 4, 8, 10, 15, 4, 6, 3])
    capacity = 20

    def knapsack_fitness(matrix):
        items = (matrix > 0.5).astype(int).flatten()
        if np.sum(items * weights) > capacity:
            return -1000
        return np.sum(items * values)

    print(f"\n{'='*50}")
    print("0-1背包问题测试")
    print(f"物品数量: {len(weights)}")
    print(f"容量限制: {capacity}")
    print(f"{'='*50}")

    init_matrix = np.random.rand(len(weights), 1)
    solver = GA_Solver(init_matrix, knapsack_fitness)

    solver.CONFIG.update({
        "POP_SIZE": 50,
        "MAX_GEN": 50,
        "N_ELITE": 10,
        "ATTENTION_UPDATE_FREQ": 5,
        "STANDARD_DEVIATION": 0.3
    })

    solver.Optimize(verbose=True)

    best_solution = (solver.population[0] > 0.5).astype(int).flatten()
    total_weight = np.sum(best_solution * weights)
    total_value = np.sum(best_solution * values)

    print(f"\n最优解: {best_solution}")
    print(f"总重量: {total_weight}")
    print(f"总价值: {total_value}")

    return solver


def tsp_problem():
    """旅行商问题简化版"""
    n_cities = 8
    np.random.seed(42)
    cities = np.random.rand(n_cities, 2) * 100

    def distance_matrix():
        dist = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                dist[i, j] = np.sqrt(np.sum((cities[i] - cities[j])**2))
        return dist

    dist_matrix = distance_matrix()

    def tsp_fitness(matrix):
        route = np.argsort(matrix.flatten())
        total_dist = 0
        for i in range(len(route)):
            total_dist += dist_matrix[route[i], route[(i+1) % len(route)]]
        return -total_dist

    print(f"\n{'='*50}")
    print("旅行商问题测试")
    print(f"城市数量: {n_cities}")
    print(f"{'='*50}")

    init_matrix = np.random.rand(n_cities, 1)
    solver = GA_Solver(init_matrix, tsp_fitness)

    solver.CONFIG.update({
        "POP_SIZE": 80,
        "MAX_GEN": 100,
        "N_ELITE": 15,
        "ATTENTION_UPDATE_FREQ": 8,
        "STANDARD_DEVIATION": 0.8
    })

    solver.Optimize(verbose=True)

    best_route = np.argsort(solver.population[0].flatten())
    print(f"\n最优路径: {best_route}")
    print(f"路径长度: {-solver.scores[0]:.2f}")

    return solver


if __name__ == "__main__":
    print("遗传算法真实优化问题测试")

    # 1. Sphere函数 (简单)
    # test_optimization_problem(
    #     "Sphere函数", sphere_function, (-5, 5), 0, (5, 5), 30
    # )

    # # 2. Rosenbrock函数 (困难)
    # test_optimization_problem(
    #     "Rosenbrock函数", rosenbrock_function, (-2, 2), 0, (4, 4)
    # )

    # 3. Ackley函数 (多峰)
    test_optimization_problem(
        "Ackley函数", ackley_function, (-5, 5), 0, (3, 3)
    )

    # # 4. 背包问题
    # knapsack_problem()

    # # 5. TSP问题
    # tsp_problem()

    print(f"\n{'='*50}")
    print("所有测试完成!")
    print(f"{'='*50}")
