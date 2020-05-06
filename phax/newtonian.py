import jax.numpy as np
from jax import grad, vmap


G_CONST_SI = 6.67430e-11
G_CONST_AIU = 4.30091e-3
ke_CONST_SI = 8.9875517923e9
ke_CONST_PP = 14.3996
ke_CONST_C1 = 1e-7


def kinetic(m):
    return lambda t, x, x_t: .5 * (m @ np.square(x_t).sum(-1))


def _dist_one_one(x, x_):
    return np.sqrt(x @ x + x_ @ x_ - 2 * x @ x_)


def gravitational_potential_pairwise(m, G=1):
    def gravitational_potential_pairwise_(t, x, x_t):
        n, *rest = x.shape
        x1 = x[None]
        x2 = x[:, None]
        x1, x2 = np.broadcast_arrays(x1, x2)
        x1 = x1.reshape(n * n, *rest)[:-1].reshape(n - 1, n + 1, *rest)[:, 1:]
        x2 = x2.reshape(n * n, *rest)[:-1].reshape(n - 1, n + 1, *rest)[:, 1:]
        r = vmap(vmap(_dist_one_one, (0, 0), 1), (0, 0), 0)(x1, x2)
        mm = m[None] * m[:, None]
        mm = mm.reshape(-1)[:-1].reshape(n - 1, n + 1)[:, 1:]
        p = mm / r
        return -.5 * G * p.sum()
    return gravitational_potential_pairwise_


def gravitational_potential_radial(M, m, G=1):
    def gravitational_potential_radial_(t, x, x_t):
        r = vmap(lambda x: np.sqrt(x @ x), (0, 0), 0)(x)
        return -G * M * (m / r).sum()
    return gravitational_potential_radial_


def lagrangian(T, V):
    return lambda t, x, x_t: T(t, x, x_t) - V(t, x, x_t)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings(
        'ignore', category=UserWarning,
        message='No GPU/TPU found, falling back to CPU.')
    t = 0
    x = np.linspace(0, 1, 3 * 9).reshape(9, 3)
    x_t = np.zeros_like(x)
    m = np.linspace(0, 1, 9)
    pot = gravitational_potential_pairwise(m)
    print(x)
    print(pot(t, x, x_t))
    print(grad(lambda xx: pot(t, xx, x_t))(x))

