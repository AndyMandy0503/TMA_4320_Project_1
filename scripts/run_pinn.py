"""Script for training and plotting the PINN model."""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_pinn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 5.4: Start
    #######################################################################

    sensor_data = generate_training_data(cfg=cfg)
    pinn_params, losses = train_pinn(sensor_data[4], cfg)
    x = sensor_data[0]
    y = sensor_data[1]
    t = sensor_data[2]
    T_pinn = predict_grid(nn_params=pinn_params['nn'], x=x, y=y, t=t, cfg=cfg)

    plot_snapshots(
        x, y, t, T_pinn, title='PINN', save_path='output/pinn/pinn_snapshots.png'
    )
    create_animation(
        x, y, t, T_pinn, title='PINN', save_path='output/pinn/pinn_animation.gif'
    )
    plt.figure()
    plt.plot(losses['total'], label='Total')
    plt.plot(losses['data'], label='Data')
    plt.plot(losses['ic'], label='IC')
    plt.plot(losses['physics'], label='Physics')
    plt.plot(losses['bc'], label='BC')

    plt.xlabel('Epoke')
    plt.ylabel('Tap')

    plt.legend()
    plt.savefig('output/pinn/pinn_losses.png')

    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
