"""Data generation utilities."""

import jax.numpy as jnp
import numpy as np

from .config import Config
from .fdm import solve_heat_equation


def generate_training_data(
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, jnp.ndarray]:
    """Generate training data from FDM solver.

    Args:
        cfg: Configuration

    Returns:
        x, y, t: Coordinate arrays
        T_fdm: FDM solution (nt, nx, ny)
        sensor_data: Sensor measurements [x, y, t, T_noisy]
    """

    #######################################################################
    # Oppgave 3.3: Start
    #######################################################################

    import matplotlib.pyplot as plt

    x, y, t, T_fdm = solve_heat_equation(cfg=cfg)
    sensor_data = _generate_sensor_data(x=x, y=y, t=t, T=T_fdm, cfg=cfg)
    DECIMALS = 2

    sensor_x_data = np.array(sensor_data[:, 0])
    sensor_y_data = np.array(sensor_data[:, 1])
    t_data = np.array(sensor_data[:, 2])
    T_data = np.array(sensor_data[:, 3])

    sensor_x_datas = []
    sensor_y_datas = []
    t_datas = []
    T_datas = []
    prev_idx = 0
    for i in range(len(t_data)):
        if int(t_data[i])==24:
            sensor_x_datas.append(sensor_x_data[i])
            sensor_y_datas.append(sensor_y_data[i])
            t_datas.append(t_data[prev_idx:i+1])
            T_datas.append(T_data[prev_idx:i+1])
            prev_idx = i+1

    fig, ax = plt.subplots()
    
    for i in range(len(t_datas)):
        ax.plot(t_datas[i], T_datas[i], label=f'Sensor {i+1}:\nPos: {np.round(sensor_x_datas[i], DECIMALS)}, {np.round(sensor_y_datas[i], DECIMALS)}')
    
    ax.legend()
    plt.show()

    #######################################################################
    # Oppgave 3.3: Slutt
    #######################################################################
    return x, y, t, T_fdm, jnp.asarray(sensor_data)


def _generate_sensor_data(
    x: np.ndarray, y: np.ndarray, t: np.ndarray, T: np.ndarray, cfg: Config
) -> np.ndarray:
    """Generate noisy sensor measurements from FDM solution."""
    sensor_data = []

    for sx, sy in cfg.sensor_locations:
        # Find nearest grid point
        i = np.argmin(np.abs(x - sx))
        j = np.argmin(np.abs(y - sy))

        # Sample at specified rate
        dt = t[1] - t[0]
        for t_idx, time in enumerate(t):
            if time % cfg.sensor_rate < dt or t_idx == 0:
                temp = T[t_idx, i, j]
                temp_noisy = temp + np.random.normal(0, cfg.sensor_noise)
                sensor_data.append([x[i], y[j], time, temp_noisy])

    return np.array(sensor_data)
