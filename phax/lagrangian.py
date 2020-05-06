from functools import partial

import jax.numpy as np
from jax import grad, jacfwd


def acc_from_lagrangian(lagrangian, newton_steps=1):
    # TODO: Make this more compact
    # TODO: add support for Jax trees
    def acc_from_lagrangian_(tt, x0, v0):
        lagrangiant = lambda x, x_t: lagrangian(tt, x, x_t)
        # ∂L/∂x as evaluated at (x0, v0)
        subx0v0_ddx_L = grad(lagrangiant, argnums=0)(x0, v0)
        # ∂L/∂v as a function of (x, v)
        ddv_L = grad(lagrangiant, argnums=1)

        # Recognize that x = x0 + t v0, v = v0 + t a0
        # Reparametrize: ∂L/∂v as a function of (t, a0).
        rep_ddv_L = lambda t, a0: ddv_L(x0 + t * v0, v0 + t * a0)
        ddt_rep_ddv_L = jacfwd(rep_ddv_L, argnums=0)
        # Set t = 0. This is a function of a0.
        subt_rep_ddt_ddv_L = partial(ddt_rep_ddv_L, 0.)

        # Find a0 with Newton's. We want to set
        #     subtx0v0_ddt_ddv_L(a0) = subx0v0_ddx_L.
        # In most cases subtx0v0_ddt_ddv_L is affine in a0, so one
        # Newton step is exact.
        diff = lambda a0: subt_rep_ddt_ddv_L(a0) - subx0v0_ddx_L
        a0 = np.zeros_like(v0)
        for _ in range(newton_steps):
            suba0_diff = diff(a0)
            suba0_dda0_diff = jacfwd(lambda a0: diff(a0).sum(-2))(a0)
            suba0_dda0_diff = np.moveaxis(suba0_dda0_diff, 1, 0)
            a0 = a0 - np.linalg.solve(suba0_dda0_diff, suba0_diff)
        
        return a0
    return acc_from_lagrangian_
