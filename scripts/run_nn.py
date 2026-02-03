"""Script for training and plotting the NN model."""

import os

import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_nn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 4.4: Start
    #######################################################################
    def make_plot(sensor_data):
        import matplotlib 
        matplotlib.use("Agg", force=True) 
        import matplotlib.pyplot as plt 

        DECIMALS = 2

        sensor_x_data = np.array(sensor_data[0])
        sensor_y_data = np.array(sensor_data[1])
        t_data = np.array(sensor_data[2])
        T_data = np.array(sensor_data[3])

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
        ax.set_xlabel('t / h')
        ax.set_ylabel(r'$^o C$')
        
        for i in range(len(t_datas)):
            ax.plot(t_datas[i], T_datas[i], label=f'Sensor {i+1}:\nPos: {np.round(sensor_x_datas[i], DECIMALS)}, {np.round(sensor_y_datas[i], DECIMALS)}')
        
        ax.legend()
        plt.show()

    sensor_data = generate_training_data(cfg=cfg)
    nn_params, losses = train_nn(sensor_data[4], cfg)
    x = sensor_data[0]
    y = sensor_data[1]
    t = sensor_data[2]
    T_nn = predict_grid(nn_params=nn_params, x=x, y=y, t=t, cfg=cfg)

    plot_snapshots(
        x, y, t, T_nn, title='NN', save_path='output/nn/nn_snapshots.png'
    )
    create_animation(
        x, y, t, T_nn, title='NN', save_path='output/nn/nn_animation.gif'
    )
    plt.figure()
    plt.plot(losses['total'], label='Total')
    plt.plot(losses['data'], label='Data')
    plt.plot(losses['ic'], label='IC')
    plt.legend()
    plt.savefig('output/nn/nn_losses.png')




    #######################################################################
    # Oppgave 4.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
