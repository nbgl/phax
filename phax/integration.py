def euler_method(acceleration_f):
    def euler_method_inner(dt, t, x, x_t):
        acc = acceleration_f(t, x, x_t)
        new_x_t = x_t + dt * acc
        new_x = x + dt * x_t
        new_t = t + dt
        return new_t, new_x, new_x_t
    return euler_method_inner


# TODO: add more ODE solvers, e.g. Runge-Kutta
# TODO: add support for Jax trees
