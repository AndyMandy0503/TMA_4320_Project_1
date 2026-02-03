"""Training routines for NN and PINN models."""

import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

from .config import Config
from .loss import bc_loss, data_loss, ic_loss, physics_loss
from .model import init_nn_params, init_pinn_params
from .optim import adam_step, init_adam
from .sampling import sample_bc, sample_ic, sample_interior


def train_nn(
    sensor_data: jnp.ndarray, cfg: Config
) -> tuple[list[tuple[jnp.ndarray, jnp.ndarray]], dict]:
    """Train a standard neural network on sensor data only.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        params: Trained network parameters
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    nn_params = init_nn_params(cfg)
    adam_state = init_adam(nn_params)

    losses = {"total": [], "data": [], "ic": []}  # Fill with loss histories

    #######################################################################
    # Oppgave 4.3: Start
    #######################################################################

    ic_array = sensor_data[:, :3].at[:, 2].set(cfg.T_outside)

    def objective_fn(nn_params):
        return cfg.lambda_data * ic_loss(nn_params=nn_params, ic_points=ic_array, cfg=cfg) + cfg.lambda_ic * data_loss(nn_params=nn_params, sensor_data=sensor_data, cfg=cfg)
    
    
    from tqdm import tqdm
    for i in tqdm(range(cfg.num_epochs), desc="Training NN"):
        obj_val, obj_grad = jax.value_and_grad(objective_fn)(nn_params)
        loss_tuple = (ic_loss(nn_params=nn_params, ic_points=ic_array, cfg=cfg), data_loss(nn_params=nn_params, sensor_data=sensor_data, cfg=cfg))
        
        losses["ic"].append(loss_tuple[0])
        losses["data"].append(loss_tuple[1])
        losses["total"].append(obj_val)

        nn_params, adam_state = adam_step(nn_params, obj_grad, adam_state, lr=cfg.learning_rate)
  
    #######################################################################
    # Oppgave 4.3: Slutt
    #######################################################################

    return nn_params, {k: jnp.array(v) for k, v in losses.items()}





def train_pinn(sensor_data: jnp.ndarray, cfg: Config) -> tuple[dict, dict]:
    """Train a physics-informed neural network.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        pinn_params: Trained parameters (nn weights + alpha)
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    pinn_params = init_pinn_params(cfg)
    opt_state = init_adam(pinn_params)

    losses = {"total": [], "data": [], "physics": [], "ic": [], "bc": []}

    #######################################################################
    # Oppgave 5.3: Start
    #######################################################################

    ic_array = sensor_data[:, :3].at[:, 2].set(cfg.T_outside)
    x, y, t = sensor_data[:, 0], sensor_data[:, 1], sensor_data[:, 2]
    ph_array = []
    bc_array = []
    #splits into boundary and interior points
    for i in range(len(x)):
        if x[i] == cfg.x_min:
            bc_array.append(jnp.array([x[i], y[i], t[i], 0, -1]))

        elif x[i] == cfg.x_max:
            bc_array.append(jnp.array([x[i], y[i], t[i], 0, 1]))
        
        elif y[i] == cfg.y_min:
            bc_array.append(jnp.array([x[i], y[i], t[i], -1, 0]))
            
        elif y[i] == cfg.y_max:
            bc_array.append(jnp.array([x[i], y[i], t[i], 1, 0]))

        else:
            ph_array.append(jnp.array([x[i], y[i], t[i]]))
    
    ph_array = jnp.array(ph_array)
    bc_array = jnp.array(bc_array)

    def objective_fn(pinn_params):
        ret_val = cfg.lambda_data * ic_loss(nn_params=pinn_params['nn'], ic_points=ic_array, cfg=cfg)
        ret_val += cfg.lambda_ic * data_loss(nn_params=pinn_params['nn'], sensor_data=sensor_data, cfg=cfg)
        ret_val += cfg.lambda_physics * physics_loss(pinn_params=pinn_params, interior_points=sensor_data, cfg=cfg)
        ret_val += cfg.lambda_bc * bc_loss(pinn_params=pinn_params, bc_points=ic_array, cfg=cfg)
        return ret_val
    

    from tqdm import tqdm
    for i in tqdm(range(cfg.num_epochs), desc="Training PINN"):
        obj_val, obj_grad = jax.value_and_grad(objective_fn)(pinn_params)
        loss_ic = ic_loss(nn_params=pinn_params['nn'], ic_points=ic_array, cfg=cfg)
        loss_data = data_loss(nn_params=pinn_params['nn'], sensor_data=sensor_data, cfg=cfg)
        loss_physics = physics_loss(pinn_params=pinn_params, interior_points=sensor_data, cfg=cfg)
        loss_bc = bc_loss(pinn_params=pinn_params, bc_points=ic_array, cfg=cfg)
        
        losses["ic"].append(loss_ic)
        losses["data"].append(loss_data)
        losses["physics"].append(loss_physics)
        losses["bc"].append(loss_bc)
        losses["total"].append(obj_val)
        
        pinn_params, opt_state = adam_step(pinn_params, obj_grad, opt_state, lr=cfg.learning_rate)

    #######################################################################
    # Oppgave 5.3: Slutt
    #######################################################################

    return pinn_params, {k: jnp.array(v) for k, v in losses.items()}
