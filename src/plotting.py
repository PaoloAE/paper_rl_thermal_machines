from __future__ import print_function
import numpy as np
import itertools
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.gridspec as gridspec
from pathlib import Path
from IPython import display
from IPython.display import Video

import sys
sys.path.append(os.path.join('..','lib'))
import sac
import sac_tri

#some constants
ANIMATION_DIR_NAME = "anims"
PLOT_DIR_NAME = "plots" 

"""
This module contains functions to visualize data that was logged with sac.SacTrain and by
sac_tri.SacTrain. In includes visualizing plots in Jupyter notebook, exporting them as pdfs and
producing animations (only used for testing).
"""

#Most useful functions

def plot_sac_logs(log_dir, is_tri, actions_per_log=6000, running_reward_file=None, running_loss_file=None,
                actions_file=None, actions_to_plot=400,plot_to_file_line = None, suppress_show=False,
                save_plot = False, extra_str="", actions_ylim=None):
    """
    Produces and displays in a Jupyter notebook a single plot with the running reward, the loss function
    and the last chosen actions. This function can also save the plot to .pdf in the PLOT_DIR_NAME folder

    Args:
        log_dir (str): location of the folder with all the logging
        is_tri (bool): if the logged data has the discrete action (True) or not (False)
        actions_per_log (int): number of actions taken between logs. Corresponds to LOG_STEPS hyperparam while training
        running_reward_file (str): location of the txt file with the running reward. If None, default location is used
        running_loss_file (str): location of the txt file with the loss function. If None, default location is used
        actions_file (str): location of the txt file with the actions. If None, default location is used
        actions_to_plot (int): how many of the last actions to show
        plot_to_file_lin (int): plots logs only up to a given file line. In None, all data is used
        suppress_show (bool): if True, it won't display the plot
        save_plot (bool): If True, is saved the plot as a pdf in PLOT_DIR_NAME
        extra_str (str): string to append to the file name of the saved plot
        actions_ylim (tuple): a 2 element tuple specifying the y_lim of the actions plot

    """
    #i create the file location strings if they are not passed it
    running_reward_file, running_loss_file, actions_file = \
                        sac_logs_file_location(log_dir, is_tri, running_reward_file, running_loss_file, actions_file)
    
    #i check if the files exist
    running_reward_exists = Path(running_reward_file).exists()
    running_loss_exists = Path(running_loss_file).exists()
    actions_exists = Path(actions_file).exists()
    
    #count the number of plots to do
    quantities_to_log = int(running_reward_exists) + 2*int(running_loss_exists) + int(actions_exists)
    
    #create the matplotlib subplots
    fig, axes = plt.subplots(quantities_to_log, figsize=(7,quantities_to_log*2.7))
    if quantities_to_log == 1:
        axes = [axes]
    axis_ind = 0
    
    #plot the running reward
    if running_reward_exists:
        plot_running_reward_on_axis(running_reward_file, axes[axis_ind], plot_to_file_line)
        axis_ind += 1
    
    #plot the running loss
    if running_loss_exists:
        plot_running_loss_on_axis(running_loss_file, axes[axis_ind],axes[axis_ind+1], plot_to_file_line)
        axis_ind += 2
    
    #plot the last actions
    if actions_exists:
            plot_actions_on_axis(actions_file, axes[axis_ind], is_tri, actions_to_plot=actions_to_plot,
                    plot_to_file_line= None if plot_to_file_line is None else int(plot_to_file_line*actions_per_log),
                    actions_ylim=actions_ylim )
            axis_ind += 1 
    
    #compact view
    fig.tight_layout()  
    
    #save the plot if requested
    if save_plot:
        plot_folder = os.path.join(log_dir, PLOT_DIR_NAME)
        Path(plot_folder).mkdir(parents=True, exist_ok=True)
        plot_file_name = os.path.join(plot_folder, f"plt{extra_str}.pdf")
        fig.savefig(plot_file_name, bbox_inches='tight')
    
    #display the plot if requested
    if not suppress_show:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    plt.close() 

def sac_paper_plot(log_dir, det_policy_sublocation,act_0,act_1,act_2,is_tri,actions_to_plot_large=10,
                    actions_to_plot_small = 100,actions_per_log = 6000, custom_colors=None,prot_linewidth=2.7,
                    reward_linewidth = None,plot_file_name = None,small_action_ylim=None,large_action_ylim=None,
                    reward_ylim=None,reward_ylabel=None, reward_plot_extra_args = None,
                    reward_plot_extra_kwargs = None,extra_cycles=None,extra_cycles_linewidth=None,
                    reward_legend_labels=None,action_legend_lines=None,action_legend_text=None,
                       action_legend_location=None):
    """
    Makes a plot in the style of Fig. 3 and 4 of the manuscript. Top panel shows the running reward, the 3 central
    plots show a zoom of the chosen actions at 3 points during training, and the last panel shows a zoom of the final 
    chosen actions chosen in a deterministic way

    Args:
        log_dir (str): location of the logging folder
        det_policy_sublocation (str): location inside log_dir where the deterministic policy file is located
        act_0 (int): step around which to display the first zoom of the chosen action. Expressed in steps 
        act_1 (int): step around which to display the second zoom of the chosen action. Expressed in steps 
        act_2 (int): step around which to display the third zoom of the chosen action. Expressed in steps 
        is_tri (bool): whether there is a discrete action (True), or not
        actions_to_plot_large (int): how many actions to display in the lower panel (zoomed deterministic actions)
        actions_to_plot_small (int): how many actions to display in the 3 small panels 
        actions_per_log (int): number of action chosen between logs. Corresponds to the hyperparameter LOG_STEPS
        custom_colors(list(colors)): list of colors for the discrete actions
        prot_linewidth (float): linewidth for the actions
        reward_linewidth (float): linewidth of the running reward
        plot_file_name (str): if specified, it saves the figure, with this name, in the PLOT_DIR_NAME dir
        small_action_ylim (list(float)): ylim for the 3 small panels showing the actions
        large_action_ylim (list(float)): ylim for the lower panel showing the deterministic actions
        reward_ylim (list(float)): ylim for the reward plot
        reward_ylabel (str): y label for the reward plot
        reward_plot_extra_args (list): if sepcificed, reward_plot.plot is called with these args
        reward_plot_extra_kwargs (dict): if sepcificed, reward_plot.plot is called with reward_plot_extra_args and
            reward_plot_extra_kwargs
        extra_cycles: allows to display additional cycles on the lower panel. See plot_actions_on_axis for details
        extra_cycles_linewidth: linewidth of the additional cycles on the lower panel
        reward_legend_labels (list(str)): list of strings for the reward plot legend
        action_legend_lines (list(Line2D)): list of Line2D objects to generate the legend for the lower plot
        action_legend_text (str): list of strings for the legend of the lower plot
        action_legend_location (str): location of the legend of the lower plot   
    """

    #get location of files
    running_reward_file, _, actions_file = \
                                    sac_logs_file_location(log_dir, is_tri, None,None,None)

    #font size
    matplotlib.rcParams.update({'font.size': 14, "text.usetex": True,
                                'text.latex.preamble' : r'\usepackage{amsmath}\usepackage{physics}'})

    #create the axis (subplots)
    fig = plt.figure(constrained_layout=True, figsize=(6,5))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios = [1,0.7,1])
    reward_ax = fig.add_subplot(gs[0, :])
    prot_0_ax = fig.add_subplot(gs[1, 0])
    prot_1_ax = fig.add_subplot(gs[1, 1],sharey=prot_0_ax)
    prot_2_ax = fig.add_subplot(gs[1, 2],sharey=prot_0_ax)
    prot_final_ax = fig.add_subplot(gs[2, :])
    plt.setp(prot_1_ax.get_yticklabels(), visible=False)
    plt.setp(prot_2_ax.get_yticklabels(), visible=False)          

    #set the reward axis
    plot_running_reward_on_axis(running_reward_file, reward_ax, plot_to_file_line = None, linewidth=reward_linewidth,
                                        custom_color = "black", lines_to_mark = [nearest_int(act_0/actions_per_log),
                                        nearest_int(act_1/actions_per_log),nearest_int(act_2/actions_per_log)],
                                        custom_mark_color="black",ylim=reward_ylim,ylabel=reward_ylabel,
                                        plot_extra_args=reward_plot_extra_args, plot_extra_kwargs=reward_plot_extra_kwargs,
                                        legend_labels=reward_legend_labels)

    #set the three actions axis
    plot_actions_on_axis(actions_file, prot_0_ax, is_tri=is_tri, actions_to_plot=actions_to_plot_small,
                                actions_ylim=small_action_ylim,plot_to_file_line=act_0,
                                custom_colors=custom_colors,constant_steps=True, linewidth = prot_linewidth,
                                two_xticks=True)
    plot_actions_on_axis(actions_file, prot_1_ax, is_tri=is_tri, actions_to_plot=actions_to_plot_small,
                                plot_to_file_line=act_1,ylabel="",custom_colors=custom_colors,
                                constant_steps=True, linewidth = prot_linewidth,two_xticks=True)
    plot_actions_on_axis(actions_file, prot_2_ax, is_tri=is_tri, actions_to_plot=actions_to_plot_small,
                                plot_to_file_line=act_2, ylabel="",custom_colors=custom_colors,
                                constant_steps=True, linewidth = prot_linewidth, two_xticks=True)

    #set the final protocol axis
    plot_actions_on_axis(log_dir + det_policy_sublocation, prot_final_ax, is_tri, actions_to_plot=actions_to_plot_large,
                                actions_ylim=large_action_ylim,plot_to_file_line=None,
                                custom_colors=custom_colors, constant_steps=True, k_notation=False, x_count_from_zero=True,
                                linewidth = prot_linewidth, xlabel="$t[dt]$",extra_cycles=extra_cycles,
                                extra_cycles_linewidth=extra_cycles_linewidth, legend_lines=action_legend_lines,
                                legend_text=action_legend_text, legend_location=action_legend_location)

    #add the (a) (b) (c) labels
    reward_ax.text(-0.11,-0.38, r'(a)', transform=reward_ax.transAxes )
    prot_0_ax.text(-0.4,-0.55, r'(b)', transform=prot_0_ax.transAxes )
    prot_final_ax.text(-0.11,-0.38, r'(c)', transform=prot_final_ax.transAxes )

    #save if necessary
    if plot_file_name is not None:
        plot_folder = os.path.join(log_dir, PLOT_DIR_NAME)
        Path(plot_folder).mkdir(parents=True, exist_ok=True)
        plot_file_name = os.path.join(plot_folder, plot_file_name)
        plt.savefig(plot_file_name)

    #show
    plt.show()

def animate_sac_plot_logs(log_dir, is_tri=False, actions_per_log=6000, running_reward_file=None, running_loss_file=None,
    actions_file=None, actions_to_plot=800, suppress_show=False, actions_ylim = None, extra_str = "", ms_delay = 200,
    start_percentage = 0., end_percentage = 1., skip_lines = 5):    
    """
    Produces an animation showing the running reward, loss functions and chosen action during the training. Visually similar to
    plot_sac_logs. It exports it to file, and can display it right into the Jupyter Notebook.
    It's very slow and could be written much better, but eventually it does work.

    Args:
        log_dir (str): location of the folder with all the logging
        is_tri (bool): if the logged data has the discrete action (True) or not (False)
        actions_per_log (int): number of actions taken between logs. Corresponds to LOG_STEPS hyperparam while training
        running_reward_file (str): location of the txt file with the running reward. If None, default location is used
        running_loss_file (str): location of the txt file with the loss function. If None, default location is used
        actions_file (str): location of the txt file with the actions. If None, default location is used
        actions_to_plot (int): how many actions to show at each frame
        suppress_show (bool): if True, it won't display the plot (but it still saves it as file)
        actions_ylim (tuple): a 2 element tuple specifying the y_lim of the actions plot
        extra_str (str): string to append to the file name of the saved plot
        ms_delay (float): milliseconds to wait between frames
        start_percentage (float): at what percentage (0 to 1) of the total training should the animation start
        end_percentage (float): at what percentage (0 to 1) of the total training should the animation stop
        skip_lines (int): how many lines of the running reward and of the running loss to skip betwee frames 
            (1 means that every logged point is available)
    """
    #i create the file locations if they are not passed it
    running_reward_file, running_loss_file, actions_file = \
                        sac_logs_file_location(log_dir, is_tri, running_reward_file, running_loss_file, actions_file)
    
    #i check if the files exist
    running_reward_exists = Path(running_reward_file).exists()
    running_loss_exists = Path(running_loss_file).exists()
    actions_exists = Path(actions_file).exists()

    #count the number of plots
    quantities_to_log = int(running_reward_exists) + 2*int(running_loss_exists) + int(actions_exists)
    
    #create the figure and the axes 
    fig, axes = plt.subplots(quantities_to_log, figsize=(7,quantities_to_log*2.7))
    fig.tight_layout()

    #create the tuple for fargs in frame_func
    fargs = (axes, running_reward_exists,running_loss_exists,actions_exists,\
        running_reward_file,running_loss_file,actions_file,actions_to_plot,actions_per_log,is_tri,actions_ylim)

    #prepare the iterator for the animation frames
    max_line_index = get_number_lines(running_reward_file, running_loss_file, actions_file) -1
    start_line = int(np.round(max_line_index*start_percentage))
    end_line = int(np.round(max_line_index*end_percentage))
    start_line = max(start_line,2)
    frames = range(start_line, end_line, skip_lines)

    #create the directory for storing the animation
    anim_folder = os.path.join(log_dir, ANIMATION_DIR_NAME)
    Path(anim_folder).mkdir(parents=True, exist_ok=True)
    anim_file_name = os.path.join(anim_folder, f"anim{extra_str}.mp4")
    
    #create the animator
    animator = ani.FuncAnimation(fig, anim_produce_frame, interval = ms_delay, frames = frames, fargs = fargs)
    animator.save(anim_file_name)
    plt.close()

    #show the video if requested
    if not suppress_show:
        display.display(Video(anim_file_name))

#functions mainly used internally

def sac_logs_file_location(log_dir, is_tri, running_reward_file, running_loss_file, actions_file):
    """
    Returns the location of the logging files. If they are passed, it doesn't change them.
    If they are None, it returns the default location

    Args:
        log_dir (str): location of the logging folder
        is_tri (bool): if the discrete action in included (True) or not (False)
        running_reward_file (str): location of reward file. If None, default location is returned
        running_loss_file (str): location of loss file. If None, default location is returned
        actions_file (str): location of actions file. If None, default location is returned

    Returns:
        running_reward_file (str): location of reward file
        running_loss_file (str): location of loss file
        actions_file (str): location of actions file
    """
    if is_tri:
        sac_module = sac_tri
    else:
        sac_module = sac
    if running_reward_file is None:
        running_reward_file = os.path.join(log_dir, sac_module.SacTrain.RUNNING_REWARD_FILE_NAME)
    if running_loss_file is None:
        running_loss_file = os.path.join(log_dir, sac_module.SacTrain.RUNNING_LOSS_FILE_NAME)
    if actions_file is None:
        actions_file = os.path.join(log_dir, sac_module.SacTrain.ACTIONS_FILE_NAME)
    return (running_reward_file, running_loss_file, actions_file)

def plot_running_reward_on_axis(file_location, axis, plot_to_file_line = None,ylabel = None,
                                xlabel = None, xticks = None, xticklabels=None, yticks = None, yticklabels=None,
                                k_notation=True, linewidth=None, custom_color=None, lines_to_mark = None, custom_mark_color = None,
                                ylim=None,plot_extra_args=None,plot_extra_kwargs=None,legend_labels=None):
    """
    Produces a plot of the running reward on a given matplot lib axis

    Args:
        file_location (str): location of the file with the running reward
        axis (matplotlib axis): the axis on which to do the plot
        plot_to_file_line (int): plot data up to this file line. If None, plots till the end
        ylabel (str): custom string for y axis
        xlabel (str): custom string for x axis
        xticks (list(float)): custom list of x ticks
        xticklabels (list(str)): custom list of x tick strings
        yticks (list(float)): custom list of y ticks
        yticklabels (list(str)): custom list of y tick strings
        k_notation (bool): if True, displays number of x axis using "k" to represent 1000
        linewidth (float): linewidth of line
        custom_color: color of the line
        lines_to_mark (list(int)): adds a circle around the points corresponding to the specified lines in the file
        custom_mark_color: color of the circles at the points specified by lines_to_mark
        ylim (tuple(int)): ylim delimiter
        plot_extra_args: will call the function axis.plot passing in these custom args
        plot_extra_kwargs: will call the function axis.plot passing in plot_extra_args and plot_extra_kwargs
        legend_labels (list(str)): list of strings for the legend labels
    """
    #setup the plot
    if legend_labels is None:
        label_iter = itertools.cycle([None])
    else:
        label_iter = itertools.cycle(legend_labels)
    if ylabel is None:
        ylabel = "$G$"
    if xlabel is None:
        xlabel = "step"
    #load the data
    plot_data = np.loadtxt(file_location).reshape(-1,2)
    if lines_to_mark is not None:
        points_to_mark = plot_data[lines_to_mark]
    if plot_to_file_line is not None:
        plot_data = plot_data[:plot_to_file_line+1]
    #perform the plot and set the labels
    axis.plot(plot_data[:,0],plot_data[:,1], linewidth=linewidth, color=custom_color, label=next(label_iter))
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    #additional details of the plot
    if k_notation:
        axis.xaxis.set_major_formatter(lambda x,y: num_to_k_notation(x) )
    if lines_to_mark is not None:
        axis.scatter(points_to_mark[:,0],points_to_mark[:,1], color=custom_mark_color)
    if ylim is not None:
        axis.set_ylim(ylim)
    if plot_extra_args is not None:
        if plot_extra_kwargs is None:
            axis.plot(*plot_extra_args, label=next(label_iter))
        else:
            axis.plot(*plot_extra_args, **plot_extra_kwargs, label=next(label_iter))
    if xticks is not None:
        axis.set_xticks(xticks)
    if xticklabels is not None:
        axis.set_xticklabels(xticklabels)
    if yticks is not None:
        axis.set_yticks(yticks)
    if yticklabels is not None:
        axis.set_yticklabels(yticklabels)
    if legend_labels is not None:
        axis.legend(loc="best",fancybox=True, framealpha=0.,borderaxespad=  0.1,handlelength=1.1, ncol=1)

def plot_running_loss_on_axis(file_location, axis1, axis2, plot_to_file_line = None, q_ylabel = None,
                                pi_ylabel = None, q_xlabel = None,pi_xlabel = None,  k_notation=True,
                                pi_k_notation=False, q_yticks = None, pi_yticks = None):
    """
    Produces a plot of the running losses on the 2 given matplot lib axis
    Args:
        file_location (str): location of the file with the running loss
        axis1 (matplotlib axis): the axis on which to do the first loss function
        axis2 (matplotlib axis): the axis on which to do the second loss function
        plot_to_file_line (int): plot data up to this file line. If None, plots till the end
        q_ylabel (str): custom string for y axis of the first loss function (q loss)
        pi_ylabel (str): custom string for y axis of the second loss function (pi loss)
        q_xlabel (str): custom string for x axis of first loss function (q loss)
        pi_xlabel (str): custom string for x axis of second loss function (pi loss)
        k_notation (bool): if True, displays number of x axis using "k" to represent 1000
        pi_k_notation(bool): if True, displays number of y axis of second loss using "k" to represent 1000
        q_yticks(list(float)): custom list of y ticks for the first loss
        pi_yticks(list(float)): custom list of y ticks for the second loss
    """
    #load data
    plot_data = np.loadtxt(file_location).reshape(-1,3)
    if plot_to_file_line is not None:
        plot_data = plot_data[:plot_to_file_line+1]
    #prepare labels
    if q_ylabel is None:
        q_ylabel = "Q Running Loss"
    if pi_ylabel is None:
        pi_ylabel = "Pi Running Loss"
    if q_xlabel is None:
        q_xlabel = "step"
    if pi_xlabel is None:
        pi_xlabel = "step"
    #plot q loss on first axis
    axis1.set_yscale("log")
    axis1.plot(plot_data[:,0],plot_data[:,1])
    axis1.set_ylabel(q_ylabel)
    axis1.set_xlabel(q_xlabel)
    #plot pi loss on second axis
    axis2.plot(plot_data[:,0],plot_data[:,2])
    axis2.set_ylabel(pi_ylabel)
    axis2.set_xlabel(pi_xlabel)
    #set the k-notation
    if k_notation:
        axis1.xaxis.set_major_formatter(lambda x,y: num_to_k_notation(x) )
        axis2.xaxis.set_major_formatter(lambda x,y: num_to_k_notation(x) )
    if pi_k_notation:
        axis2.yaxis.set_major_formatter(lambda x,y: num_to_k_notation(x) )
    #set yticks
    if q_yticks is not None:
        axis1.set_yticks(q_yticks)
    if pi_yticks is not None:
        axis2.set_yticks(pi_yticks)

def plot_actions_on_axis(file_location, axis, is_tri, actions_to_plot=1200, plot_to_file_line = None,
                        actions_ylim = None, ylabel = None, xlabel = None, xticks = None, xticklabels=None,
                        yticks = None, yticklabels=None, k_notation = True, linewidth=None, custom_colors = None,
                        constant_steps=False, x_count_from_zero=False,  two_xticks=False,extra_cycles=None,
                        extra_cycles_linewidth=None, legend_lines=None, legend_text=None, legend_location="best",
                        legend_cols = None):
    """
    Produces a plot of the last chosen actions on a given matplot lib axis

    Args:
        file_location (str): location of the file with the actions
        axis (matplotlib axis): the axis on which to do the plot
        is_tri (bool): True if the discrete action is present
        actions_to_plot (int): how many actions to display in the plot
        plot_to_file_line (int): plot data up to this file line. If None, plots till the end
        actions_ylim (list(float)): delimiter for the y axis
        ylabel (str): custom string for y axis
        xlabel (str): custom string for x axis
        xticks (list(float)): custom list of x ticks
        xticklabels (list(str)): custom list of x tick strings
        yticks (list(float)): custom list of y ticks
        yticklabels (list(str)): custom list of y tick strings
        k_notation (bool): if True, displays number of x axis using "k" to represent 1000
        linewidth (float): linewidth of line
        custom_colors(list(colors)): list of colors for the discrete actions
        constant_steps (bool): if True, it plots the actions as piecewise constant with dashed line
            for the jumps. Otherwise each action is just a dot
        x_count_from_zero (bool): if true, sets x axis from zero. Otherwise from step index
        two_xticks (bool): if True, it only places 2 x_ticks at smallest and largest value
        extra_cycles: used to overlap additional given functions on the plot. It must be a list of
            tuples as the form (function, range, color).
        extra_cycles_linewidth (float): dith of the extra cycle line
        legend_lines(list(Line2D)): list of Line2D objects to generate the legend
        legend_text (list(str)): list of strings for the legend
        legend_location: matplotlib legend_location
        legend_cols (int): number of columns for the legend
    """
    
    #set labels
    if ylabel is None:
        ylabel = "$u$"
    if xlabel is None:
        xlabel = "$t$" if x_count_from_zero else "step"   
    
    #set the color iterator
    if custom_colors is None:
        if is_tri:
            color_iter = itertools.cycle(["orange","cornflowerblue","limegreen"])
        else:
            color_iter = itertools.cycle(["black"])
    else:
        color_iter = itertools.cycle(custom_colors)

    #prevents weird +e5 from axis labels
    axis.ticklabel_format(useOffset=False)
    #load data
    plot_data = np.loadtxt(file_location)
    if plot_to_file_line is None:
        plot_to_file_line = plot_data.shape[0]-1
    plot_to_file_line = min(plot_to_file_line,plot_data.shape[0]-1)
    actions_to_plot = min(actions_to_plot, plot_data.shape[0])
    data_to_plot = plot_data[(plot_to_file_line-actions_to_plot+1):(plot_to_file_line+1)]
    if actions_ylim is not None:
        axis.set_ylim(actions_ylim)
    #plot actions
    if is_tri:
        if constant_steps:
            tri_colors = [next(color_iter),next(color_iter),next(color_iter)]
            #if counting from zero, i replace the x values with growing numbers from zero
            if x_count_from_zero:
                data_to_plot[:,0] = np.arange(data_to_plot.shape[0])
            #compute the step to draw the length of the last point
            dt = data_to_plot[-1,0] - data_to_plot[-2,0]
            #first I do the vertical dahsed line, so they get covered
            for i in range(data_to_plot.shape[0]-1):
                axis.plot([data_to_plot[i+1,0],data_to_plot[i+1,0]],[data_to_plot[i,2],data_to_plot[i+1,2]], color="lightgray",
                        linewidth=0.8)
            #horizontal lines
            data_to_plot = np.concatenate((data_to_plot, np.array([[data_to_plot[-1,0]+dt,0.,0.]])))
            for i in range(data_to_plot.shape[0]-1):   
                    axis.plot([data_to_plot[i,0],data_to_plot[i+1,0]],[data_to_plot[i,2],data_to_plot[i,2]], 
                                color=tri_colors[nearest_int(data_to_plot[i,1])], linewidth=linewidth)
        else:
            #only plots a dot for each action
            tri_act_data = data_to_plot[:,1]
            zero_mask = np.abs(tri_act_data) < 0.00001
            one_mask = np.abs(tri_act_data-1.) < 0.00001
            two_mask = np.abs(tri_act_data-2.) < 0.00001
            zero_data = data_to_plot[zero_mask]
            one_data = data_to_plot[one_mask]
            two_data = data_to_plot[two_mask]
            for i in range(2,data_to_plot.shape[1]):
                axis.scatter(zero_data[:,0],zero_data[:,i], color = next(color_iter))
                axis.scatter(one_data[:,0],one_data[:,i], color = next(color_iter))
                axis.scatter(two_data[:,0],two_data[:,i], color = next(color_iter))
    else:
        if constant_steps:
            color = next(color_iter)
            #if counting from zero, i replace the x values with growing numbers from zero
            if x_count_from_zero:
                data_to_plot[:,0] = np.arange(data_to_plot.shape[0])
            #compute the step to draw the length of the last point
            dt = data_to_plot[-1,0] - data_to_plot[-2,0]
            #first I do the vertical dahsed line, so they get covered
            for i in range(data_to_plot.shape[0]-1):
                axis.plot([data_to_plot[i+1,0],data_to_plot[i+1,0]],[data_to_plot[i,1],data_to_plot[i+1,1]], color="lightgray",
                    linewidth=0.8)
            #horizontal lines
            data_to_plot = np.concatenate( (data_to_plot, np.array([[data_to_plot[-1,0]+dt,0.]]) ))
            for i in range(data_to_plot.shape[0]-1):  
                axis.plot([data_to_plot[i,0],data_to_plot[i+1,0]],[data_to_plot[i,1],data_to_plot[i,1]], color=color, linewidth=linewidth)
        else:
            for i in range(1,data_to_plot.shape[1]):
                axis.scatter(data_to_plot[:,0],data_to_plot[:,i], color = next(color_iter))
    #additional details of plot
    axis.set_ylabel(ylabel)
    if xlabel != "":
        axis.set_xlabel(xlabel)
    if xticks is not None:
        axis.set_xticks(xticks)
    elif two_xticks:
        axis.set_xticks([ data_to_plot[0,0], data_to_plot[-1,0] ])
    if yticks is not None:
        axis.set_yticks(yticks)
    if k_notation:
        axis.xaxis.set_major_formatter(lambda x,y: num_to_k_notation(x) )
    if xticklabels is not None:
        axis.set_xticklabels(xticklabels)
    if yticklabels is not None:
        axis.set_yticklabels(yticklabels)

    #plot an extra function, if necessary
    if extra_cycles is not None:
        if not isinstance(extra_cycles[0], list) and not isinstance(extra_cycles[0], tuple):
            extra_cycles = [extra_cycles]
        #create the x-y points to plot
        for extra_cycle in extra_cycles:
            extra_func, extra_range, extra_color = extra_cycle
            x = np.linspace(extra_range[0],extra_range[1],250)
            y = [extra_func(x_val) for x_val in x]
            axis.plot(x,y, color=extra_color, linewidth=extra_cycles_linewidth, dashes=[4/extra_cycles_linewidth,2/extra_cycles_linewidth])
    
    #do the legend if necessary
    if legend_lines is not None:
        if legend_cols is None:
            legend_cols = len(legend_lines)
        axis.legend(legend_lines,legend_text,loc=legend_location,fancybox=False, framealpha=0.,borderaxespad=0.,
                    ncol=legend_cols,handlelength=0.5) #handlelength=subPlots[subIndex].legendLineLength

def num_to_k_notation(tick, tex=True):
    """
    Used to produce ticks with a "k" indicating 1000

    Args:
        tick (float): value of the tick to be converted to a string
        tex (bool): if true, it will wrap the string in $$, making it look more "latex-like"

    Returns:
        tick_str (str): the string for the tick
    """
    tick_str = str(int(tick // 1000))
    end_not_stripped = str(int(tick % 1000))
    end = end_not_stripped.rstrip('0')
    if len(end) > 0:
        end = ("0"*(3-len(end_not_stripped))) + end
        tick_str += f".{end}"
    if tex:
        tick_str = "$" + tick_str + "$"
    tick_str += "k"
    return tick_str

def nearest_int(num):
    """ return the nearest integer to the float number num """
    return int(np.round(num))

def produce_otto_cycle(u_min,u_max,t1,t2,t3,t4,dt,t_range):
    """
    produces an extra_cycles for plot_actions_on_axis() or sac_paper_plot() representing
    an Otto cycle, i.e. a trapezoidal cycle. 

    Args:
        u_min (float): minimum value of u in the Otto cycle
        u_max (float): maximum value of u in the Otto cycle
        t1 (float): duration of the first stroke 
        t2 (float): duration of the second stroke
        t3 (float): duration of the third stroke
        t4 (float): duration of the forth stroke
        dt (float): duration of each time-step
        t_range (tuple(float)): time interval (ti, tf) where to plot the Otto cycle
    """
    
    t,t_fin = t_range
    t = t/dt
    t_fin = t_fin/dt
    t1 = t1/dt
    t2 = t2/dt
    t3 = t3/dt
    t4 = t4/dt
    times = [t1,t2,t3,t4]
    current_stroke = 0

    def otto_cycle(t,u_min,u_max,t1,t2,t3,t4):
        """ auxillary function describing the Otto cycle as a function of t"""
        t_tot = t1+t2+t3+t4
        #make it periodic setting it to first period
        t = t % t_tot
        if t< t1:
            return u_min
        elif t< t1+t2:
            return u_min + (u_max-u_min)*(t-t1)/t2
        elif t< t1+t2+t3:
            return u_max
        else:
            return u_max + (u_min-u_max)*(t-t1-t2-t3)/t4
    
    def stroke_color(stroke):
        """ auxillary function describing the color of the otto cycle """
        if stroke ==0:
            return "cornflowerblue"   
        elif stroke ==1:
            return "limegreen"
        elif stroke==2:
            return "orange"
        else:
            return "limegreen"

    extra_cycles = []
    while t < t_fin:
        new_t = min(t + times[current_stroke], t_fin)
        extra_cycles.append([lambda t,u_min=u_min,u_max=u_max,t1=t1,t2=t2,t3=t3,t4=t4: otto_cycle(t,u_min,u_max,t1,t2,t3,t4),
               [t,new_t],stroke_color(current_stroke)])
        t = new_t
        current_stroke = (current_stroke+1)%4
    return extra_cycles

def anim_produce_frame(up_to_line, *fargs):
    """
    This function is called by the animator of matplotlib to produce each frame of the animation.
    The animator is created in animate_sac_plot_logs()

    Args:
        up_to_line (int): produce the logs up to the specified file line of the reward and loss files
        *fargs: arguments necessary to produce the frame. See animate_sac_plot_logs for details
    """
    #unpack *fargs
    axes,running_reward_exists,running_loss_exists,actions_exists,\
        running_reward_file,running_loss_file,actions_file,actions_to_plot, \
        actions_per_log,is_tri,actions_ylim  = fargs
    #produce the plots for the current frame
    axis_ind = 0
    if running_reward_exists:
        axes[axis_ind].clear()
        plot_running_reward_on_axis(running_reward_file, axes[axis_ind], up_to_line)
        axis_ind += 1
    if running_loss_exists:
        axes[axis_ind].clear()
        axes[axis_ind+1].clear()
        plot_running_loss_on_axis(running_loss_file, axes[axis_ind],axes[axis_ind+1], up_to_line)
        axis_ind += 2
    if actions_exists:
        axes[axis_ind].clear()
        plot_actions_on_axis(actions_file,axes[axis_ind],is_tri,actions_to_plot=actions_to_plot,
                                 plot_to_file_line=int(up_to_line*actions_per_log),
                                 actions_ylim=actions_ylim)

def get_number_lines(running_reward_file, running_loss_file, action_count_file):
    """
    Returns the number of lines in the reward and loss files
    
    Args:
        running_reward_file (str): location of the reward file
        running_loss_file (str): location of the loss file
        action_count_file (str): location of the actions file

    Raises:
        NameError: if all files don't exist
    
    Returns:
        lines (int): number of lines in the file
     """
    if Path(running_reward_file).exists():
        data = np.loadtxt(running_reward_file).reshape(-1,2)
        return data.shape[0]
    if Path(running_loss_file).exists():
        data = np.loadtxt(running_loss_file).reshape(-1,2)
        return data.shape[0]
    if Path(action_count_file).exists():
        data = np.loadtxt(action_count_file).reshape(-1,2)
        return data.shape[0]
    raise NameError("No files to count lines")



