import jax
import jax.numpy as np

import phax


if __name__ == '__main__':
    N = 1000
    m = np.linspace(1, 10, N)

    lagrangian = phax.newtonian.lagrangian(
        phax.newtonian.kinetic(m),
        phax.newtonian.gravitational_potential_pairwise(m)
    )
    accf = phax.lagrangian.acc_from_lagrangian(lagrangian)
    integrator = jax.jit(phax.integration.euler_method(accf))

    t = 0.
    x = np.linspace(0, 1, 3 * N).reshape(N, 3)
    x_t = np.zeros_like(x)

    for i in range(50):
        print(t)
        t, x, x_t = integrator(1., t, x, x_t)
