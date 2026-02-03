"""Loss functions for PINN training."""

import jax.numpy as jnp
from jax import grad, vmap

from .config import Config
from .model import forward


def data_loss(
    nn_params: list[tuple[jnp.ndarray, jnp.ndarray]], sensor_data, cfg: Config
) -> jnp.ndarray:
    """MSE loss between predictions and sensor measurements.

    Args:
        nn_params: Network parameters (list of (w, b) tuples)
        sensor_data: Array of [x, y, t, T_measured] (N, 4)
        cfg: Configuration

    Returns:
        Mean squared error
    """
    x, y, t, T_true = (
        sensor_data[:, 0],
        sensor_data[:, 1],
        sensor_data[:, 2],
        sensor_data[:, 3],
    )

    #######################################################################
    # Oppgave 4.2: Start
    #######################################################################

    T_f = forward(nn_params, x, y, t, cfg)
    loss = jnp.mean((T_f - T_true)**2)
    data_loss_val = jnp.array(loss)

    #######################################################################
    # Oppgave 4.2: Slutt (se også ic_loss)
    #######################################################################

    return data_loss_val


def ic_loss(
    nn_params: list[tuple[jnp.ndarray, jnp.ndarray]],
    ic_points: jnp.ndarray,
    cfg: Config,
) -> jnp.ndarray:
    """Initial condition loss: T(x, y, 0) = T_outside.

    Args:
        nn_params: Network parameters (list of (w, b) tuples)
        ic_points: Array of [x, y, 0] points (N, 3)
        cfg: Configuration

    Returns:
        Mean squared IC error
    """
    x, y = ic_points[:, 0], ic_points[:, 1]

    #######################################################################
    # Oppgave 4.2: Start
    #######################################################################

    T_f = forward(nn_params, x, y, cfg.T_outside, cfg)
    loss = jnp.mean((T_f - cfg.T_outside)**2)
    ic_loss_val = jnp.array(loss)
    
    #######################################################################
    # Oppgave 4.2: Slutt (se også data_loss)
    #######################################################################

    return ic_loss_val


def physics_loss(pinn_params, interior_points, cfg: Config):
    """PDE residual loss at collocation points.

    Args:
        pinn_params: Full pinn_params dict with 'nn', 'log_alpha', 'log_power'
        interior_points: Array of [x, y, t] points (N, 3)
        cfg: Configuration

    Returns:
        Mean squared PDE residual
    """
    x, y, t = interior_points[:, 0], interior_points[:, 1], interior_points[:, 2]

    #######################################################################
    # Oppgave 5.2: Start
    #######################################################################
    alpha = jnp.exp(pinn_params['log_alpha'])
    P = jnp.exp(pinn_params['log_power'])
    q = jnp.where(cfg.is_source(x, y), P, 0.0)

    def T_fn(x_, y_, t_):
            return forward(pinn_params["nn"], x_, y_, t_, cfg)
    
    def _pde_residual_scalar(pinn_params, xi, yi, ti, cfg):
        f_t = grad(T_fn, 2)(xi, yi, ti)  
        f_xx = grad(grad(T_fn, 0), 0)(xi, yi, ti)  
        f_yy = grad(grad(T_fn, 1), 1)(xi, yi, ti) 

        residual = f_t - alpha * (f_xx + f_yy) - q
        return residual
    
    residuals = vmap(
        lambda xi, yi, ti: _pde_residual_scalar(pinn_params, xi, yi, ti, cfg)
        )(x, y, t)

    physics_loss_val = jnp.array(jnp.mean(residuals**2))
    
    #######################################################################
    # Oppgave 5.2: Slutt
    #######################################################################

    return physics_loss_val


def bc_loss(pinn_params: dict, bc_points, cfg: Config) -> jnp.ndarray:
    """Robin boundary condition loss: -k * grad(T) . n = h * (T - T_out).

    Args:
        pinn_params: Full pinn_params dict with 'nn', 'log_k', 'log_h' keys
        bc_points: Array of [x, y, t, nx, ny] points (N, 5) where (nx, ny) is outward normal
        cfg: Configuration with k, h, T_outside

    Returns:
        Mean squared BC residual
    """
    x, y, t = bc_points[:, 0], bc_points[:, 1], bc_points[:, 2]
    nx, ny = bc_points[:, 3], bc_points[:, 4]

    def _bc_residual_scalar(pinn_params, x, y, t, nx, ny, cfg: Config):
        """Compute Robin BC residual: -k * grad(T) . n - h * (T - T_out) = 0.

        Args:
            pinn_params: Full pinn_params dict with 'nn', 'log_k', 'log_h' keys
            x, y, t: Point on boundary (scalars)
            nx, ny: Outward normal components (scalars)
            cfg: Configuration

        Returns:
            BC residual (scalar)
        """

        def T_fn(x, y, t):
            return forward(pinn_params["nn"], x, y, t, cfg)

        # Compute spatial gradients using automatic differentiation
        T_x = grad(T_fn, 0)(x, y, t)
        T_y = grad(T_fn, 1)(x, y, t)

        # Temperature at boundary point
        T = T_fn(x, y, t)

        # Robin BC: -k * (grad T . n) = h * (T - T_out)
        grad_T_dot_n = T_x * nx + T_y * ny
        k = jnp.exp(pinn_params["log_k"])
        h = jnp.exp(pinn_params["log_h"])
        residual = -k * grad_T_dot_n - h * (T - cfg.T_outside)

        return residual

    residuals = vmap(
        lambda xi, yi, ti, nxi, nyi: _bc_residual_scalar(
            pinn_params, xi, yi, ti, nxi, nyi, cfg
        )
    )(x, y, t, nx, ny)

    bc_loss_val = jnp.mean(residuals**2)

    return bc_loss_val
