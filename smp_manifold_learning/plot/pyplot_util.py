#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Nov 11 10:00:00 2019

@author: gsutanto
@comment: convenient utility functions for plotting in Python
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


#all_color_list = ['b','g','r','c','m','y','k']
all_color_list = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3', '#808000', '#000075', '#808080', '#000000']
plt_pause_time_secs = 0.05
                  
def hist(np_var, title, fig_num=1):
    plt.figure(fig_num)
    plt.hist(np_var, bins='auto')
    plt.title(title)
    plt.pause(plt_pause_time_secs)
    plt.show()
    return None

def plot_3D(X_list, Y_list, Z_list, title, 
            X_label, Y_label, Z_label, fig_num=1, 
            label_list=[''], color_style_list=[['b','-']], 
            is_showing_start_and_goal=False,
            N_data_display=-1,
            is_auto_line_coloring_and_styling=False, 
            is_showing_grid=True):
    N_dataset = len(X_list)
    assert (N_dataset == len(Y_list))
    assert (N_dataset == len(Z_list))
    assert (N_dataset == len(label_list))
    if (is_auto_line_coloring_and_styling):
        all_style_list = ['-','--','-.',':']
        color_style_list = [[c,s] for c in all_color_list for s in all_style_list]
    assert (N_dataset <= len(color_style_list))
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111, projection='3d')
    for d in range(N_dataset):
        if (N_data_display < 0):
            X = X_list[d]
            Y = Y_list[d]
            Z = Z_list[d]
        else:
            X = X_list[d][:N_data_display]
            Y = Y_list[d][:N_data_display]
            Z = Z_list[d][:N_data_display]
        color = color_style_list[d][0]
        linestyle = color_style_list[d][1]
        labl = label_list[d]
        ax.plot(X, Y, Z, c=color, ls=linestyle, label=labl)
        if (is_showing_start_and_goal):
            ax.scatter(X_list[d][0], Y_list[d][0], Z_list[d][0], 
                       c=color, label='start '+labl, marker='o')
            ax.scatter(X_list[d][-1], Y_list[d][-1], Z_list[d][-1], 
                       c=color, label='end '+labl, marker='^', s=200)
    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)
    ax.set_zlabel(Z_label)
    ax.legend()
    ax.grid(is_showing_grid)
    ax.set_title(title)
    plt.pause(plt_pause_time_secs)
    plt.show()
    return None

def plot_2D(X_list, Y_list, title, 
            X_label, Y_label, Y_ERR_list=None, fig_num=1,
            label_list=[''], color_style_list=[['b','-']],
            err_color_style_list=None,
            is_showing_start_and_goal=False,
            start_idx_data_display=0,
            N_data_display=-1,
            is_auto_line_coloring_and_styling=False,
            is_showing_grid=True,
            save_filepath=None,
            ylim_top=None, ylim_bottom=None):
    N_dataset = len(X_list)
    assert (N_dataset == len(Y_list))
    if (Y_ERR_list is not None) and (err_color_style_list is not None):
        assert (N_dataset == len(Y_ERR_list))
        assert (N_dataset == len(err_color_style_list))
    assert (N_dataset == len(label_list))
    if (is_auto_line_coloring_and_styling):
        all_style_list = ['-','--','-.',':']
        color_style_list = [[c,s] for s in all_style_list for c in all_color_list]
        assert (N_dataset <= len(color_style_list))
    fig = plt.figure(fig_num)
    is_updating_ylims = False
    if ylim_top is None:
        [_, ylim_top] = plt.ylim()
    else:
        is_updating_ylims = True
    if ylim_bottom is None:
        [ylim_bottom, _] = plt.ylim()
    else:
        is_updating_ylims = True
    if is_updating_ylims:
        plt.ylim(ylim_bottom, ylim_top)
    ax = fig.add_subplot(111)
    for d in range(N_dataset):
        if (N_data_display < 0):
            X = X_list[d][start_idx_data_display:]
            Y = Y_list[d][start_idx_data_display:]
            if (Y_ERR_list is not None) and (err_color_style_list is not None):
                Y_ERR = Y_ERR_list[d][start_idx_data_display:]
                err_color = err_color_style_list[d]
        else:
            X = X_list[d][start_idx_data_display:N_data_display]
            Y = Y_list[d][start_idx_data_display:N_data_display]
            if (Y_ERR_list is not None) and (err_color_style_list is not None):
                Y_ERR = Y_ERR_list[d][start_idx_data_display:N_data_display]
                err_color = err_color_style_list[d]
        color = color_style_list[d][0]
        linestyle = color_style_list[d][1]
        labl = label_list[d]
        ax.plot(X, Y, c=color, ls=linestyle, label=labl)
        if (Y_ERR_list is not None):
            ax.fill_between(X, Y-Y_ERR, Y+Y_ERR,
                            edgecolor=err_color, facecolor=err_color)
        if (is_showing_start_and_goal):
            ax.scatter(X_list[d][0], Y_list[d][0], 
                       c=color, label='start '+labl, marker='o')
            ax.scatter(X_list[d][-1], Y_list[d][-1], 
                       c=color, label='end '+labl, marker='^', s=200)
    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)
    ax.legend()
    ax.grid(is_showing_grid)
    ax.set_title(title)
    if (save_filepath is not None):
        assert (os.path.isdir(os.path.abspath(os.path.join(save_filepath, os.pardir)))), "Parent directory of %s does not exist!" % save_filepath
        print("save_filepath = ", save_filepath)
        plt.savefig(save_filepath + '.png', bbox_inches='tight')
    else:
        plt.pause(plt_pause_time_secs)
        plt.show()
    return None

def errorbar_2D(X_list, Y_list, E_list, title, 
                X_label, Y_label, fig_num=1, 
                label_list=[''], color_style_list=[['b',None]], 
                N_data_display=-1,
                is_auto_line_coloring_and_styling=False, 
                is_showing_grid=True, 
                X_lim=None, 
                save_filepath=None):
    N_dataset = len(X_list)
    assert (N_dataset == len(Y_list))
    assert (N_dataset == len(E_list))
    assert (N_dataset == len(label_list))
    if (is_auto_line_coloring_and_styling):
        all_style_list = ['-','--','-.',':']
        color_style_list = [[c,s] for s in all_style_list for c in all_color_list]
        assert (N_dataset <= len(color_style_list))
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111)
    for d in range(N_dataset):
        if (N_data_display < 0):
            X = X_list[d]
            Y = Y_list[d]
            E = E_list[d]
        else:
            X = X_list[d][:N_data_display]
            Y = Y_list[d][:N_data_display]
            E = E_list[d][:N_data_display]
        color = color_style_list[d][0]
        labl = label_list[d]
        ax.errorbar(x=X, y=Y, yerr=E, c=color, ls='None', label=labl, fmt='o')
    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)
    if (X_lim is not None):
        ax.set_xlim(X_lim)
    ax.legend()
    ax.grid(is_showing_grid)
    ax.set_title(title)
    if (save_filepath is not None):
        assert (os.path.isdir(os.path.abspath(os.path.join(save_filepath, os.pardir)))), "Parent directory of %s does not exist!" % save_filepath
        plt.savefig(save_filepath + '.png', bbox_inches='tight')
    else:
        plt.pause(plt_pause_time_secs)
        plt.show()
    return None

def scatter_3D(X_list, Y_list, Z_list, title, 
               X_label, Y_label, Z_label, fig_num=1, 
               label_list=[''], color_style_list=[['b','o']],
               is_auto_line_coloring_and_styling=False, 
               is_showing_grid=True):
    N_dataset = len(X_list)
    assert (N_dataset == len(Y_list))
    assert (N_dataset == len(Z_list))
    assert (N_dataset == len(label_list))
    if (is_auto_line_coloring_and_styling):
        all_style_list = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        color_style_list = [[c,s] for s in all_style_list for c in all_color_list]
        assert (N_dataset <= len(color_style_list)), "Not enough color style variations to cover all datasets!"
    else:
        assert (N_dataset <= len(color_style_list))
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111, projection='3d')
    for d in range(N_dataset):
        X = X_list[d]
        Y = Y_list[d]
        Z = Z_list[d]
        color = color_style_list[d][0]
        markerstyle = color_style_list[d][1]
        labl = label_list[d]
        ax.scatter(X, Y, Z, c=color, marker=markerstyle, label=labl)
    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)
    ax.set_zlabel(Z_label)
    ax.legend()
    ax.grid(is_showing_grid)
    ax.set_title(title)
    plt.pause(plt_pause_time_secs)
    plt.show()
    return None

def scatter_2D(X_list, Y_list, title, 
               X_label, Y_label, fig_num=1, 
               label_list=[''], color_style_list=[['b','o']],
               is_auto_line_coloring_and_styling=False, 
               is_showing_grid=True):
    N_dataset = len(X_list)
    assert (N_dataset == len(Y_list))
    assert (N_dataset == len(label_list))
    if (is_auto_line_coloring_and_styling):
        all_style_list = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        color_style_list = [[c,s] for s in all_style_list for c in all_color_list]
        assert (N_dataset <= len(color_style_list)), "Not enough color style variations to cover all datasets!"
    else:
        assert (N_dataset == len(color_style_list))
    plt.figure(fig_num)
    for d in range(N_dataset):
        X = X_list[d]
        Y = Y_list[d]
        color = color_style_list[d][0]
        markerstyle = color_style_list[d][1]
        labl = label_list[d]
        plt.scatter(X, Y, c=color, marker=markerstyle, label=labl)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.title(title)
    plt.legend()
    plt.grid(is_showing_grid)
    plt.pause(plt_pause_time_secs)
    plt.show()
    return None

def subplot_ND(NDtraj_list, title, 
               Y_label_list, fig_num=1, 
               label_list=[''], color_style_list=[['b','o']],
               is_auto_line_coloring_and_styling=False, 
               is_showing_grid=True,
               save_filepath=None,
               N_data_display=-1,
               dt=1, X_label='Time Index', start_idx_data_display=0):
    assert (len(NDtraj_list) >= 1)
    N_traj_to_plot = len(NDtraj_list)
    assert (len(label_list) == N_traj_to_plot)
    D = NDtraj_list[0].shape[1]
    assert (len(Y_label_list) == D)
    if (is_auto_line_coloring_and_styling):
        all_color_list = ['b','g','r','c','m','y','k']
        all_style_list = ['-','--','-.',':']
        color_style_list = [[c,s] for s in all_style_list for c in all_color_list]
        assert (N_traj_to_plot <= len(color_style_list)), "Not enough color style variations to cover all datasets!"
    
    ax = [None] * D
    fig, ax = plt.subplots(D, sharex=True, sharey=True)
    
    for n_traj_to_plot in range(N_traj_to_plot):
        assert (NDtraj_list[n_traj_to_plot].shape[1] == D)
        traj_label = label_list[n_traj_to_plot]
        for d in range(D):
            if (D == 1):
                axd = ax
            else:
                axd = ax[d]
            if (n_traj_to_plot == 0):
                if (d == 0):
                    axd.set_title(title)
                axd.set_ylabel(Y_label_list[d])
                if (d == D-1):
                    axd.set_xlabel(X_label)
            if N_data_display <= 0:
                traj_y = NDtraj_list[n_traj_to_plot][:,d]
                traj_x = np.array(range(NDtraj_list[n_traj_to_plot].shape[0])) * dt
            else:
                assert(NDtraj_list[n_traj_to_plot].shape[0] >= N_data_display), \
                    "NDtraj_list[n_traj_to_plot].shape[0] = %d; N_data_display = %d" % \
                    (NDtraj_list[n_traj_to_plot].shape[0], N_data_display)
                traj_y = NDtraj_list[n_traj_to_plot][start_idx_data_display:N_data_display, d]
                traj_x = np.array(range(start_idx_data_display, N_data_display)) * dt
            color = color_style_list[n_traj_to_plot][0]
            linestyle = color_style_list[n_traj_to_plot][1]
            axd.plot(traj_x, traj_y,
                     c=color, ls=linestyle, 
                     label=traj_label)
            if (n_traj_to_plot == 0):
                axd.grid(is_showing_grid)
    if (D == 1):
        ax.legend()
    else:
        ax[0].legend()
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
#    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    if (save_filepath is not None):
        final_path = os.path.abspath(os.path.join(save_filepath, os.pardir))
        if not os.path.isdir(final_path):
            os.makedirs(final_path)
        assert (os.path.isdir(os.path.abspath(os.path.join(save_filepath, os.pardir)))), "Parent directory of %s does not exist!" % save_filepath
        plt.savefig(save_filepath + '.png', bbox_inches='tight')
        print("Saved figure in %s" % (save_filepath + '.png'))
    else:
        plt.pause(plt_pause_time_secs)
        plt.show()
    return None


if __name__ == '__main__':
    plt.close('all')
    
    N_data_points = 1000
    x = np.array(range(N_data_points))/(1.0 * (N_data_points-1))
    
    x_pow_1 = np.power(x,1)
    x_pow_8 = np.power(x,8)
    D_display = 2
    data_dim_list = [None] * D_display
    for i in range(D_display):
        data_dim_list[i] = list()
        if (i == 0):
            for j in range(2):
                data_dim_list[i].append(x)
        elif (i == 1):
            data_dim_list[i].append(x_pow_1)
            data_dim_list[i].append(x_pow_8)
    plot_2D(X_list=data_dim_list[0], 
            Y_list=data_dim_list[1], 
            title='Powers of x', 
            X_label='x', 
            Y_label='Value', 
            fig_num=2, 
            label_list=['x','x^8'], 
            color_style_list=[['b','-'],['r','-']],
            save_filepath=None
            )
    
    x_pow_1_e = 0.05 * x
    x_pow_8_e = 0.05 * 8 * x
    D_display = 3
    data_dim_list = [None] * D_display
    for i in range(D_display):
        data_dim_list[i] = list()
        if (i == 0):
            for j in range(2):
                data_dim_list[i].append(x)
        elif (i == 1):
            data_dim_list[i].append(x_pow_1)
            data_dim_list[i].append(x_pow_8)
        elif (i == 2):
            data_dim_list[i].append(x_pow_1_e)
            data_dim_list[i].append(x_pow_8_e)
    errorbar_2D(X_list=data_dim_list[0], 
                Y_list=data_dim_list[1], 
                E_list=data_dim_list[2], 
                title='Error Bars of x', 
                X_label='x', 
                Y_label='Value', 
                fig_num=3, 
                label_list=['x','x^8'], 
                color_style_list=[['b',None],['r',None]], 
                N_data_display=10, 
                save_filepath=None
                )
    
    x_pow_2 = np.power(x,2)
    x_pow_3 = np.power(x,3)
    x_pow_4 = np.power(x,4)
    x_pow_5 = np.power(x,5)
    TwoDtraj_list = [np.zeros((N_data_points, 2))] * 2
    TwoDtraj_list[0] = np.vstack([x_pow_1, x_pow_8, x_pow_4]).T
    TwoDtraj_list[1] = np.vstack([x_pow_2, x_pow_3, x_pow_5]).T
    subplot_ND(NDtraj_list=TwoDtraj_list, 
               title='SubPlot Test', 
               Y_label_list=['x','y','z'], 
               fig_num=4, 
               label_list=['pow 1 vs pow 8 vs pow 4', 'pow 2 vs pow 3 vs pow 5'], 
               is_auto_line_coloring_and_styling=True)
