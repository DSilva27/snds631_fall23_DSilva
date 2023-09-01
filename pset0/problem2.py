import numpy as np
import cvxpy as cp


def main():
    mu = np.array([0.1, 0.15, 0.12, 0.08, 0.2])

    sigma = np.array(
        [
            [0.02, 0.005, 0.01, 0.004, 0.007],
            [0.005, 0.03, 0.015, 0.003, 0.008],
            [0.01, 0.015, 0.025, 0.007, 0.01],
            [0.004, 0.003, 0.007, 0.015, 0.006],
            [0.007, 0.008, 0.01, 0.006, 0.04],
        ]
    )

    # Portfolio optimization using cvxpy
    w = cp.Variable(mu.shape[0])
    target_return = 0.12

    objective = cp.Minimize(cp.quad_form(w, sigma))

    constraints = [
        w.T @ mu >= target_return,
        cp.sum(w) == 1,
        w >= 0,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Optimal portfolio weights
    optimal_weights = w.value
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    print("Optimal Portfolio Weights:", optimal_weights)


if __name__ == "__main__":
    main()
