import numpy as np
import cvxpy as cp


def main():
    x = cp.Variable(3)

    objective = cp.Minimize(3 * x[0] + 4 * x[1] + 2 * x[2])
    constraints = [
        cp.sum(x) >= 10,
        2 * x[0] - x[1] + 3 * x[2] <= 15,
        x >= 0,
    ]

    problem = cp.Problem(objective, constraints)
    result = problem.solve()

    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    print("Optimal values for x, y, z:", x.value)


if __name__ == "__main__":
    main()
