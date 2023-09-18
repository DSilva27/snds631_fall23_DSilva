from typing import Union, Optional
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax


def newton_rapson_1d(
    f: callable, x0: float, tol: Optional[float] = 1e-6, max_iter: Optional[int] = 100
):
    grad_f = jax.jit(jax.grad(f))
    x = x0
    for _ in range(max_iter):
        x = x - f(x) / grad_f(x)
        if jnp.abs(f(x)) < tol:
            return x, f(x)

    raise RuntimeError("Failed to converge, max iterations exceeded")


def newton_rapson_nd(
    f: callable,
    x0: ArrayLike,
    tol: Optional[float] = 1e-6,
    max_iter: Optional[int] = 100,
):
    jacob_f = jax.jit(jax.jacfwd(f))
    x = x0.copy()

    for _ in range(max_iter):
        x = x - jnp.linalg.solve(jacob_f(x), f(x))
        if jnp.linalg.norm(f(x)) < tol:
            return x, f(x)

    raise RuntimeError("Failed to converge, max iterations exceeded")


def newton_rapson(
    f: callable,
    x0: Union[float, ArrayLike],
    tol: Optional[float] = 1e-6,
    max_iter: Optional[int] = 100,
):
    if isinstance(x0, float):
        return newton_rapson_1d(f, x0, tol, max_iter)
    else:
        return newton_rapson_nd(f, x0, tol, max_iter)


@jax.jit
def f1(x):
    return x**3 - 6 * x**2 + 11 * x - 6


@jax.jit
def f2(x):
    return jnp.exp(x) - 3 * x


@jax.jit
def f3(x):
    return jnp.cos(x) - x


# Two-dimensional examples
@jax.jit
def f4(x):
    return jnp.array([x[0] ** 2 + x[1] ** 2 - 4, x[0] * x[1] - 1])


@jax.jit
def f5(x):
    return jnp.array([x[0] ** 2 - x[1], x[0] ** 2 + x[1] ** 2 - 5])


def main():
    root1 = newton_rapson(f1, 0.0)
    root2 = newton_rapson(f2, 0.0)
    root3 = newton_rapson(f3, 0.0)
    root4 = newton_rapson(f4, jnp.array([3.0, 1.0]))
    root5 = newton_rapson(f5, jnp.array([3.0, 1.0]))

    print(
        f"The root found by Newton-Rapson method is: {root1[0]}, the value of f1 is {root1[1]}"
    )
    print(
        f"The root found by Newton-Rapson method is: {root2[0]}, the value of f2 is {root2[1]}"
    )
    print(
        f"The root found by Newton-Rapson method is: {root3[0]}, the value of f3 is {root3[1]}"
    )
    print(
        f"The root found by Newton-Rapson method is: {root4[0]}, the value of f4 is {root4[1]}"
    )
    print(
        f"The root found by Newton-Rapson method is: {root5[0]}, the value of f5 is {root5[1]}"
    )


if __name__ == "__main__":
    main()
