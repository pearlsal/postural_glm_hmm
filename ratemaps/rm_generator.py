import csv
import os
import sys
import time

import pickle
from random import shuffle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy import *
import scipy.ndimage.filters
import scipy.io
import scipy.stats

import numpy as np
import math
import copy

from .uuuuu import *


def plot_1d_values(xvals, yvals, y2vals, occs, means, stds, lqs, uqs, occupancy_thresh_1d, tit, use_quantile):
    """
    instead using 1.96, before use 2.
    """
    if len(xvals) != len(yvals):
        print('should all be same', tit, len(xvals), len(yvals), len(y2vals), len(occs), len(means), len(stds))
    if np.sum(~np.isnan(stds)) > 0:
        if use_quantile:
            xx = xvals[~np.isnan(lqs)]
            yyA = lqs[~np.isnan(lqs)]
            yyB = uqs[~np.isnan(lqs)]
        else:
            xx = xvals[~np.isnan(stds)]
            yyA = means[~np.isnan(stds)] + 1.96 * stds[~np.isnan(stds)]
            yyB = means[~np.isnan(stds)] - 1.96 * stds[~np.isnan(stds)]
        plt.plot(xx, yyA, '.', color='black')
        plt.plot(xx, yyB, '.', color='black')
        for i in range(len(xx)):
            plt.plot([xx[i], xx[i]], [yyA[i], yyB[i]], '-', color='black')
        plt.plot(xx, means[~np.isnan(stds)], '.', color='black')
    plt.plot(xvals[occs > occupancy_thresh_1d], yvals[occs > occupancy_thresh_1d], 'o', color='blue')
    plt.plot(xvals[occs > occupancy_thresh_1d], y2vals[occs > occupancy_thresh_1d], '+', color='blue')
    plt.title(tit, fontsize=24)


def plot_ratemaps(ax, data, occupancy, occupancy_thresh, title, polar_mode, color_map='jet', yticks=None, ylabels=None):
    if polar_mode:
        rm_shape = data.shape
        bins_size = np.ones(rm_shape[0]) * (2 * np.pi / rm_shape[0])
        arc_loc = np.arange(0.0, 2 * np.pi, 2 * np.pi / rm_shape[0]) - np.pi + np.pi / rm_shape[0]
        n_layers = rm_shape[1]

        max_val = np.nanmax(data)
        min_val = np.nanmin(data)

        val_bins = np.linspace(min_val, max_val, 257)

        map_cols_ind = np.zeros(rm_shape, 'i')
        for i in range(256):
            col_inds = np.logical_and(data >= val_bins[i], data < val_bins[i + 1])
            map_cols_ind[col_inds] = i

        map_cols_ind[occupancy < occupancy_thresh] = 0

        cmap = plt.colormaps[color_map]
        for i in range(n_layers):
            plot_color = cmap(map_cols_ind[:, i])
            plt.bar(arc_loc, 1, width=bins_size, bottom=i, color=plot_color, edgecolor=plot_color)
        plt.ylim(0, rm_shape[0])
        # ax.set_theta_direction(-1)
        # ax.set_theta_offset(np.pi / 2.0)
        if yticks is not None:
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels)
        ax.set_xticks(np.pi / 180. * np.linspace(0, 360, 8, endpoint=False))
        ax.set_xticklabels(['0$^\circ$', '45$^\circ$', '90$^\circ$', '135$^\circ$', '180$^\circ$',
                             '-135$^\circ$', '-90$^\circ$', '-45$^\circ$'])
        plt.grid(axis='y')
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, float(np.nanmax(data))))
        plt.colorbar(sm)
        plt.title(title, fontsize=24)
    else:
        masked_array = np.ma.masked_where(occupancy < float(occupancy_thresh), data)
        cmap = copy.copy(matplotlib.cm.get_cmap(color_map))
        # cmap = matplotlib.cm.jet
        cmap.set_bad('white')
        imshow_obj = plt.imshow(np.transpose(masked_array), cmap=cmap, interpolation='nearest', origin='lower')
        plt.title(title, fontsize=24)
        clb = plt.colorbar()
        imshow_obj.set_clim(0., max(np.ravel(data)))
        clb.draw_all()
        # plt.axis('off')


def ratemap_generator(data, cell_index=(10, 11), temp_offsets=0, n_bins_1d=36, smoothing_par_1d=1,
                      occupancy_thresh_1d=0.4, occupancy_thresh_2d=0.4, smoothing_par_2d=(1.15, 1.15),
                      n_shuffles=1000, shuffle_range=(15, 60), use_quantile=True,
                      use_time_bins=True, periodic=True,
                      velocity_par=(60, 18, 20), spatial_par=30,
                      self_motion_par=(0, 80, -5, 80, 3),
                      include_derivatives=True, extra_2d_tasks=None, comparing=False, pie_style=True,
                      color_map='jet', pl_subplot=(14, 10), pl_size=(70, 70), limit_y=True,
                      seeds=None, debug_mode=False):
    """
    Perform shuffling. Calculate firing rate. Generate rate maps.
    :param data:
    :param cell_index: tuple, index of the cells included in the data.
    :param temp_offsets: int or np.array([..., -250, -150, -100, -50, -25, -16, 0, 16, 25, 50, 100, 250, ...).
    :param n_bins_1d: number of bins that used in the 1d factors.
    :param occupancy_thresh_1d: Minimum bin occupancy (seconds).
    :param occupancy_thresh_2d: Minimum bin occupancy (seconds).
    :param smoothing_par_1d: Band width of Gaussian smoothing (bins).
    :param smoothing_par_2d: Width and Height of Gaussian smoothing (bins).
    :param comparing: False(default), if True plot 2 groups of datasets side by side to compare.
    :param n_shuffles: 1000 (default), number of shuffles.
            you need enough for it to be well approximated by a Gaussian, assuming it can be.
            Each rate map distributions for the scores are calculated using
            random shuffles between +/- [MIN,MAX] relative to zero.
            Note, it is IMPORTANT that the MIN/MAX values are outside the range for the temporal offsets!!!
    :param shuffle_range: tuple, Min/Max value for the shuffling range (second).
    :param use_quantile: if True, the shuffle bar is 95% quantiles. Otherwise, 95% confidence interval.
    :param use_time_bins: boolean, False (default). method to use for mapping cell activities to tracking frames.
    :param periodic: boolean, if True, when smoothing rate map w.r.t angles, consider the angle values are periodic.
    :param velocity_par: tuple of length 3. (max speed, n_bins of speed, n_bins of angle). Default (60, 18, 20)
    :param spatial_par: int, number of bins for spatial map.
    :param self_motion_par: parameters for control self-motion firing rate, (min_dx, max_dx, min_dy, max_dy, bin size),
            Min/Max value for the horizontal axis (cm/s), Min/Max value for the vertical axis (cm/s).
            if None, then no self-motion rate map will be created. The default values are (0, 80, -5, 80, 3).
    :param include_derivatives: if True (default), 2d rate maps will be generated for some of the 1st. order derivatives
    :param extra_2d_tasks: dict, extra 2d tasks for generating 2d rate maps.
    :param pie_style: if True, when one of the 2d factors is periodic, the Pie style is applied.
    :param color_map: str, the name of color map. Default using <jet>.
    :param pl_subplot: tuple of length 2, number of rows and columns of subplots. Default (14, 10)
    :param pl_size: tuple of length 2, figure size (width, height). Default (70, 70).
    :param limit_y: boolean, if True, the save y lims are used for every single plot.
    :param seeds: int/None, random seed for generating random numbers. If None, a seed will be generated randomly.
    :param debug_mode: False (default). if True, all steps information will be printed. Set to False on clusters.
    :return:
    """

    # check data
    if not isinstance(cell_index, tuple):
        raise TypeError('cell_index should be a tuple.')

    if isinstance(seeds, int):
        np.random.seed(seeds)
    else:
        cute_seed = np.random.randint(100000)
        np.random.seed(cute_seed)

    include_velocity_map = False
    include_self_motion = False
    include_spatial_map = False

    # read parameters
    if self_motion_par is not None:
        if isinstance(self_motion_par, tuple):
            if len(self_motion_par) == 5:
                include_self_motion = True

    if velocity_par is not None:
        if isinstance(velocity_par, tuple):
            if len(velocity_par) == 3:
                include_velocity_map = True

    if spatial_par is not None:
        if isinstance(spatial_par, int):
            include_spatial_map = True

    output_file_prefix = data['output_file_prefix']
    output_file_prefix = '%s_shuffling_%05d_times' % (output_file_prefix, n_shuffles)
    #
    temp_offsets = np.ravel(temp_offsets) / 1000.

    frame_rate = data['framerate']
    overall_framerate = data['overall_framerate']
    s_ind = data['session_indicator']
    frame_times = data['frame_times']
    time_bins = data['time_bins']

    session_ts = data['session_ts']
    tracking_ts = data['tracking_ts']

    startaltbins = data['startaltbins']
    endaltbins = data['endaltbins']

    # load cell data
    cell_names = [data['cell_names'][i] for i in cell_index]
    cell_activities = [data['cell_activities'][i] for i in cell_index]
    n_cells = len(cell_names)
    if debug_mode:
        print('Making rate maps for {} cells, with index {}'.format(n_cells, cell_index))

    # load special 2d rm needed data
    animal_loc = data['animal_location']
    dxs = data['dxs']
    dys = data['dys']
    # load 1d rm needed data
    factor1d = data['possiblecovariates']
    xaxis1d = data['possiblecovariatesnames']
    bins1d = data['possiblecovariatesbins']
    bounds1d = data['possiblecovariatesbounds']

    keys_factor_1d = list(factor1d.keys())
    keys_factor_1d.sort()
    n_keys_1d = len(keys_factor_1d)

    factor_torus = {}
    for j in range(n_keys_1d):
        da_key = keys_factor_1d[j]
        if periodic:
            factor_torus[da_key] = bounds1d[da_key][2]
        else:
            factor_torus[da_key] = False

    tracking_validating = validating_tracking_frame(frame_times, [session_ts[0]], [session_ts[1]])
    p1_validating = validating_tracking_frame(frame_times, startaltbins[1::2], endaltbins[1::2])
    p2_validating = validating_tracking_frame(frame_times, startaltbins[0::2], endaltbins[0::2])

    binned_factors = map_factor_to_bins(factor1d, bins1d)

    binned_factors_occ = get_1d_factor_bin_occ(binned_factors, n_bins_1d, tracking_validating, frame_rate, s_ind)
    p1_binned_occ = get_1d_factor_bin_occ(binned_factors, n_bins_1d, p1_validating, frame_rate, s_ind)
    p2_binned_occ = get_1d_factor_bin_occ(binned_factors, n_bins_1d, p2_validating, frame_rate, s_ind)

    rate_maps_data, shuffled_means, shuffled_stds, shuffled_lower, shuffled_upper = get_shuffled_data(
        factor1d, bins1d, factor_torus, binned_factors_occ, cell_names, cell_activities,
        session_ts, tracking_ts, overall_framerate, time_bins, n_bins_1d, n_shuffles, shuffle_range, use_time_bins,
        smoothing_par_1d, occupancy_thresh_1d)

    n_temp_offset = len(temp_offsets)
    base_cell_frames = {}
    part1_cell_frames = {}
    part2_cell_frames = {}
    for i in range(n_cells):
        da_cell_name = cell_names[i]
        cell_data = cell_activities[i]

        base_cell_frames[da_cell_name] = []
        part1_cell_frames[da_cell_name] = []
        part2_cell_frames[da_cell_name] = []
        for j in range(n_temp_offset):
            cell_frames = map_cell_activity_to_tracking_frame(
                cell_data, temp_offsets[j], overall_framerate, tracking_ts, time_bins,
                [session_ts[0]], [session_ts[1]], use_time_bins)
            cell_frames_p1 = map_cell_activity_to_tracking_frame(
                cell_data, temp_offsets[j], overall_framerate, tracking_ts, time_bins,
                startaltbins[1::2], endaltbins[1::2], use_time_bins)
            cell_frames_p2 = map_cell_activity_to_tracking_frame(
                cell_data, temp_offsets[j], overall_framerate, tracking_ts, time_bins,
                startaltbins[0::2], endaltbins[0::2], use_time_bins)
            base_cell_frames[da_cell_name].append(cell_frames)
            part1_cell_frames[da_cell_name].append(cell_frames_p1)
            part2_cell_frames[da_cell_name].append(cell_frames_p2)

    dict_2d = get_default_2d_tasks(include_derivatives)
    if extra_2d_tasks is not None:
        dict_2d = add_2d_tasks(dict_2d, extra_2d_tasks)

    n_1d_map = len(factor1d) * 2
    n_2d_map = 0
    if include_spatial_map:
        n_2d_map = n_2d_map + 2
    if include_self_motion:
        n_2d_map = n_2d_map + len(dxs[0, :]) * 2
    if include_velocity_map:
        n_2d_map = n_2d_map + 2

    n_2d_map = n_2d_map + len(dict_2d) * 2

    pl_subplot_row = np.ceil(n_2d_map / 10) + np.ceil(n_1d_map / 10)
    pl_subplot = (int(pl_subplot_row), 10)

    # include_velocity_map = False
    # include_self_motion = False
    # include_spatial_map = False

    # -----------------------------------------------------------
    #                       start plot
    # -----------------------------------------------------------
    fig_width, fig_height = np.ravel(pl_size)
    plot_rows, plot_cols = np.ravel(pl_subplot)
    if debug_mode:
        print('available monkeys', np.sort(list(xaxis1d.keys())))

    for cellnum in np.arange(n_cells):
        da_cell_name = cell_names[cellnum]
        print_cell = cell_index[cellnum]
        cell_data = cell_activities[cellnum]

        print('plotting cell {}, named {} ...'.format(print_cell, da_cell_name))

        with PdfPages('%s_%04d_%s.pdf' % (output_file_prefix, print_cell, da_cell_name)) as pdf:
            for i in np.arange(len(temp_offsets)):
                cell_frames = base_cell_frames[da_cell_name][i]
                cell_frames_p1 = part1_cell_frames[da_cell_name][i]
                cell_frames_p2 = part2_cell_frames[da_cell_name][i]

                fig = plt.figure(i + 1, figsize=(fig_width, fig_height))
                plt.clf()
                subplot_index = 1
                # plot 2d rate maps
                if dict_2d is not None:
                    print('plotting 2D rate maps ...')
                    keys2d = list(dict_2d.keys())
                    keys2d.sort()
                    for j in range(len(keys2d)):
                        polar_mode = False
                        da_2d_key = keys2d[j]
                        k2d1 = dict_2d[da_2d_key][0]
                        k2d2 = dict_2d[da_2d_key][1]
                        binned_fac1 = binned_factors[k2d1]
                        binned_fac2 = binned_factors[k2d2]
                        bins1 = bins1d[k2d1]
                        bins2 = bins1d[k2d2]
                        raw_rm, smoothed_rm, occ_2d = get_2d_ratemap(
                            binned_fac1, binned_fac2, bins1, bins2, tracking_validating, cell_frames, frame_rate,
                            smoothing_par_2d, s_ind, factor_torus[k2d1], factor_torus[k2d2])

                        info_rate_2d = calculate_skinfo_rate(smoothed_rm, occ_2d)
                        rate_maps_data['{}-{}-raw_rm_2d'.format(da_cell_name, da_2d_key[2:])] = raw_rm
                        rate_maps_data['{}-{}-smoothed_rm_2d'.format(da_cell_name, da_2d_key[2:])] = smoothed_rm
                        rate_maps_data['{}-{}-occ_2d'.format(da_cell_name, da_2d_key[2:])] = occ_2d
                        rate_maps_data['{}-{}-info_rate'.format(da_cell_name, da_2d_key[2:])] = info_rate_2d

                        if factor_torus[k2d1] and not factor_torus[k2d2] and pie_style:
                            polar_mode = True

                        xval2 = np.round(np.diff(bins2) * 0.5 + bins2[:-1], 2)
                        xval2 = np.append(xval2, np.round(bins2[-1], 2))
                        y_ticks = np.linspace(0, len(bins2) - 1, 6).astype(int)
                        y_labels = xval2[y_ticks]

                        tit = '{}'.format(da_2d_key[2:].replace('_', ' '))
                        ax = plt.subplot(plot_rows, plot_cols, subplot_index, polar=polar_mode)
                        plot_ratemaps(ax, raw_rm, occ_2d, occupancy_thresh_2d, tit, polar_mode, color_map,
                                      y_ticks, y_labels)
                        subplot_index += 1

                        tit = '    Smoothed ({})'.format(info_rate_2d)
                        ax = plt.subplot(plot_rows, plot_cols, subplot_index, polar=polar_mode)
                        plot_ratemaps(ax, smoothed_rm, occ_2d, occupancy_thresh_2d, tit, polar_mode, color_map,
                                      y_ticks, y_labels)
                        subplot_index += 1

                # plot sel motion maps
                if include_self_motion:
                    print('plotting self motion maps ...')
                    for swb in range(len(data['settings']['selfmotion_window_size'])):
                        raw_rm_sf, smoothed_rm_sf, occ_sf = get_self_motion_map(
                            dxs[:, swb], dys[:, swb], self_motion_par, tracking_validating, cell_frames, frame_rate,
                            smoothing_par_2d, s_ind)

                        info_rate_sf = calculate_skinfo_rate(smoothed_rm_sf, occ_sf)
                        rate_maps_data['{}-self_motion_{}-raw_rm'.format(da_cell_name, swb)] = raw_rm_sf
                        rate_maps_data['{}-self_motion_{}-smoothed_rm'.format(da_cell_name, swb)] = smoothed_rm_sf
                        rate_maps_data['{}-self_motion_{}-occ'.format(da_cell_name, swb)] = occ_sf
                        rate_maps_data['{}-self_motion_{}-info_rate'.format(da_cell_name, swb)] = info_rate_sf

                        tit = 'Self motion {}'.format(data['settings']['selfmotion_window_size'][swb])
                        ax = plt.subplot(plot_rows, plot_cols, subplot_index)
                        plot_ratemaps(ax, raw_rm_sf, occ_sf, occupancy_thresh_2d, tit, False, color_map, None, None)
                        subplot_index += 1

                        tit = '    Smoothed ({})'.format(info_rate_sf)
                        ax = plt.subplot(plot_rows, plot_cols, subplot_index)
                        plot_ratemaps(ax, smoothed_rm_sf, occ_sf, occupancy_thresh_2d, tit, False, color_map, None, None)
                        subplot_index += 1

                # ------------------------------- plot velocity tuning
                if include_velocity_map:
                    print('plotting velocity maps ...')
                    body_dir_angles = data['possiblecovariates']['C Body_direction'] * math.pi / 180
                    speeds = data['possiblecovariates']['B Speeds']
                    raw_rm_vc, smoothed_rm_vc, occ_vc, speed_bins = get_velocity_map(
                        body_dir_angles, speeds, velocity_par, tracking_validating, cell_frames, frame_rate,
                        factor_torus['C Body_direction'], smoothing_par_2d, s_ind)

                    info_rate_vc = calculate_skinfo_rate(smoothed_rm_vc, occ_vc)

                    xval2 = np.round(np.diff(speed_bins) * 0.5 + speed_bins[:-1], 2)
                    xval2 = np.append(xval2, np.round(speed_bins[-1], 2))
                    y_ticks = np.linspace(0, len(speed_bins) - 1, 6).astype(int)
                    y_labels = xval2[y_ticks]

                    tit = 'Velocity'
                    ax = plt.subplot(plot_rows, plot_cols, subplot_index, polar=pie_style)
                    plot_ratemaps(ax, raw_rm_vc.T, occ_vc.T, occupancy_thresh_2d, tit, pie_style, color_map,
                                  y_ticks, y_labels)
                    subplot_index += 1

                    tit = '    Smoothed ({})'.format(info_rate_vc)
                    ax = plt.subplot(plot_rows, plot_cols, subplot_index, polar=pie_style)
                    plot_ratemaps(ax, smoothed_rm_vc.T, occ_vc.T, occupancy_thresh_2d, tit, pie_style, color_map,
                                  y_ticks, y_labels)
                    subplot_index += 1

                # ------------------------------- plot spatial maps
                if include_spatial_map:
                    print('plotting spatial maps ...')
                    raw_rm_sp, smoothed_rm_sp, occ_sp = get_spatial_map(
                        animal_loc, spatial_par, tracking_validating, cell_frames, frame_rate, smoothing_par_2d, s_ind)

                    info_rate_sp = calculate_skinfo_rate(smoothed_rm_sp, occ_sp)

                    tit = 'Space'
                    ax = plt.subplot(plot_rows, plot_cols, subplot_index)
                    plot_ratemaps(ax, raw_rm_sp, occ_sp, occupancy_thresh_2d, tit, False, color_map, None, None)
                    subplot_index += 1

                    tit = '    Smoothed ({})'.format(info_rate_sp)
                    ax = plt.subplot(plot_rows, plot_cols, subplot_index)
                    plot_ratemaps(ax, smoothed_rm_sp, occ_sp, occupancy_thresh_2d, tit, False, color_map, None, None)
                    subplot_index += 1

                while np.mod(subplot_index - 1, 10) != 0:
                    plt.subplot(plot_rows, plot_cols, subplot_index)
                    plt.axis('off')
                    subplot_index += 1

                # start 1d plots
                print('plotting 1D rate maps ...')
                max_ylim = -10

                axis_list = []
                info_rates_vec = []
                for j in range(n_keys_1d):
                    da_key = keys_factor_1d[j]

                    occ = binned_factors_occ[da_key]
                    xvals, rawrm, smrm = get_1d_ratemap(factor1d[da_key], bins1d[da_key], occ,
                                                        cell_frames, factor_torus[da_key], smoothing_par_1d)

                    info_rate = calculate_skinfo_rate(rawrm, occ)
                    info_rates_vec.append(info_rate)

                    xvals, rawrm_p1, smrm_p1 = get_1d_ratemap(factor1d[da_key], bins1d[da_key], p1_binned_occ[da_key],
                                                              cell_frames_p1, factor_torus[da_key], smoothing_par_1d)

                    xvals, rawrm_p2, smrm_p2 = get_1d_ratemap(factor1d[da_key], bins1d[da_key], p2_binned_occ[da_key],
                                                              cell_frames_p2, factor_torus[da_key], smoothing_par_1d)

                    axis_list.append(plt.subplot(plot_rows, plot_cols, subplot_index))
                    tit = da_key[2:].replace('_', ' ')
                    tit = '{} ({})'.format(tit, info_rate)
                    plot_1d_values(xvals, smrm, rawrm, occ, shuffled_means[da_key][:, cellnum],
                                   shuffled_stds[da_key][:, cellnum], shuffled_lower[da_key][:, cellnum],
                                   shuffled_upper[da_key][:, cellnum], occupancy_thresh_1d, tit, use_quantile)
                    valid_p1 = (p1_binned_occ[da_key] > occupancy_thresh_1d)
                    plt.plot(xvals[valid_p1], smrm_p1[valid_p1], 'o', color='green')
                    plt.plot(xvals[valid_p1], rawrm_p1[valid_p1], '+', color='green')
                    valid_p2 = (p2_binned_occ[da_key] > occupancy_thresh_1d)
                    plt.plot(xvals[valid_p2], smrm_p2[valid_p2], 'o', color='red')
                    plt.plot(xvals[valid_p2], rawrm_p2[valid_p2], '+', color='red')
                    plt.xlabel(xaxis1d[da_key])
                    subplot_index += 1

                    if max_ylim < axis_list[j].get_ylim()[1]:
                        max_ylim = axis_list[j].get_ylim()[1]


                    plt.subplot(plot_rows, plot_cols, subplot_index)
                    vals = factor1d[da_key]
                    plt.hist(vals[~np.isnan(vals)], 40)
                    subplot_index += 1

                    # save data
                    rate_map_1d_data = {'bin_center': xvals, 'raw_firing_rate': rawrm, 'occupancy': occ,
                                        'smoothed_firing_rate': smrm,
                                        'shuffle_mean': shuffled_means[da_key][:, cellnum],
                                        'shuffle_std': shuffled_stds[da_key][:, cellnum],
                                        'shuffle_lower_quant': shuffled_lower[da_key][:, cellnum],
                                        'shuffle_upper_quant': shuffled_upper[da_key][:, cellnum],
                                        'raw_firing_rate_p1': rawrm_p1, 'smoothed_firing_rate_p1': smrm_p1,
                                        'occupancy_p1': p1_binned_occ[da_key], 'raw_firing_rate_p2': rawrm_p2,
                                        'smoothed_firing_rate_p2': smrm_p2, 'occupancy_p2': p2_binned_occ[da_key]}

                    rate_maps_data['{}-{}-data_1d'.format(cell_names[cellnum], da_key[2:])] = rate_map_1d_data
                    rate_maps_data['{}-{}-info_rate'.format(cell_names[cellnum], da_key[2:])] = info_rate
                # set ylim
                if limit_y:
                    for j in range(len(axis_list)):
                        axis_list[j].set_ylim([0, max_ylim])

                good_max_ylim = max_ylim

                sort_order = np.argsort(info_rates_vec)[::-1]
                best_fac = np.ravel(keys_factor_1d)[sort_order]
                for b_ind in range(4):
                    best_fac[b_ind] = best_fac[b_ind][2:].replace('_', ' ')

                # fig title
                tit = "{} : {} : {}\nTop 1D info rates from: {}, {}, {}, {}".format(
                    output_file_prefix, cell_names[cellnum], temp_offsets[i],
                    best_fac[0], best_fac[1], best_fac[2], best_fac[3])

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.gcf().suptitle(tit, fontsize=40)
                if abs(temp_offsets[i] - 0) < 0.001:
                    plt.savefig('%s_%04d_%s_at_zero.png' % (output_file_prefix, print_cell, cell_names[cellnum]))

                # save data
                save_file_name = '{}_{:04d}_{}_for_recreating_rate_maps.pkl'.format(
                    output_file_prefix, print_cell, cell_names[cellnum])

                infile = open(save_file_name, 'wb')
                pickle.dump(rate_maps_data, infile, protocol=pickle.HIGHEST_PROTOCOL)
                infile.close()
                # scipy.io.savemat(save_file_name, rate_maps_data)

            pdf.savefig()
            plt.close()

        if comparing:
            # load special 2d rm needed data
            cf_animal_loc = data['cf_animal_location']
            cf_dxs = data['cf_dxs']
            cf_dys = data['cf_dys']
            cf_factor1d = data['cf_possiblecovariates']
            cf_bins1d = data['cf_possiblecovariatesbins']

            cf_keys_factor_1d = list(cf_factor1d.keys())

            binned_factors = map_factor_to_bins(cf_factor1d, cf_bins1d)

            binned_factors_occ = get_1d_factor_bin_occ(binned_factors, n_bins_1d, tracking_validating, frame_rate, s_ind)
            p1_binned_occ = get_1d_factor_bin_occ(binned_factors, n_bins_1d, p1_validating, frame_rate, s_ind)
            p2_binned_occ = get_1d_factor_bin_occ(binned_factors, n_bins_1d, p2_validating, frame_rate, s_ind)

            cf_rate_maps_data, cf_shuffled_means, cf_shuffled_stds, cf_shuffled_lower, cf_shuffled_upper = \
                get_shuffled_data(cf_factor1d, cf_bins1d, factor_torus, binned_factors_occ, cell_names, cell_activities,
                                  session_ts, tracking_ts, overall_framerate, time_bins, n_bins_1d, n_shuffles,
                                  shuffle_range, use_time_bins, smoothing_par_1d, occupancy_thresh_1d)

            with PdfPages('%s_%04d_%s_comparing.pdf' % (output_file_prefix, print_cell, da_cell_name)) as pdf:
                cf_plot_rows = plot_rows * 2
                for i in np.arange(len(temp_offsets)):
                    cell_frames = base_cell_frames[da_cell_name][i]
                    cell_frames_p1 = part1_cell_frames[da_cell_name][i]
                    cell_frames_p2 = part2_cell_frames[da_cell_name][i]

                    fig = plt.figure(i + 1, figsize=(fig_width, fig_height * 2))
                    plt.clf()
                    subplot_index = 1
                    # plot 2d rate maps
                    if dict_2d is not None:
                        print('plotting 2D rate maps ...')
                        for j in range(len(keys2d)):
                            polar_mode = False
                            da_2d_key = keys2d[j]
                            k2d1 = dict_2d[da_2d_key][0]
                            k2d2 = dict_2d[da_2d_key][1]
                            if k2d1 not in cf_keys_factor_1d or k2d2 not in cf_keys_factor_1d:
                                continue
                            binned_fac1 = binned_factors[k2d1]
                            binned_fac2 = binned_factors[k2d2]
                            bins1 = bins1d[k2d1]
                            bins2 = bins1d[k2d2]

                            if factor_torus[k2d1] and not factor_torus[k2d2] and pie_style:
                                polar_mode = True

                            raw_rm = rate_maps_data['{}-{}-raw_rm_2d'.format(da_cell_name, da_2d_key[2:])]
                            smoothed_rm = rate_maps_data['{}-{}-smoothed_rm_2d'.format(da_cell_name, da_2d_key[2:])]
                            occ_2d = rate_maps_data['{}-{}-occ_2d'.format(da_cell_name, da_2d_key[2:])]
                            info_rate_2d = rate_maps_data['{}-{}-info_rate'.format(da_cell_name, da_2d_key[2:])]

                            cf_raw_rm, cf_smoothed_rm, cf_occ_2d = get_2d_ratemap(
                                binned_fac1, binned_fac2, bins1, bins2, tracking_validating, cell_frames, frame_rate,
                                smoothing_par_2d, s_ind, factor_torus[k2d1], factor_torus[k2d2])

                            cf_info_rate_2d = calculate_skinfo_rate(cf_smoothed_rm, cf_occ_2d)

                            xval2 = np.round(np.diff(bins2) * 0.5 + bins2[:-1], 2)
                            xval2 = np.append(xval2, np.round(bins2[-1], 2))
                            y_ticks = np.linspace(0, len(bins2) - 1, 6).astype(int)
                            y_labels = xval2[y_ticks]

                            tit = '{}'.format(da_2d_key[2:].replace('_', ' '))
                            ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index, polar=polar_mode)
                            plot_ratemaps(ax, raw_rm, occ_2d, occupancy_thresh_2d, tit, polar_mode, color_map,
                                          y_ticks, y_labels)
                            subplot_index += 1

                            tit = '    Smoothed ({})'.format(info_rate_2d)
                            ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index, polar=polar_mode)
                            plot_ratemaps(ax, smoothed_rm, occ_2d, occupancy_thresh_2d, tit, polar_mode, color_map,
                                          y_ticks, y_labels)
                            subplot_index += 1

                            tit = 'Cf. {}'.format(da_2d_key[2:].replace('_', ' '))
                            ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index, polar=polar_mode)
                            plot_ratemaps(ax, cf_raw_rm, cf_occ_2d, occupancy_thresh_2d, tit, polar_mode, color_map,
                                          y_ticks, y_labels)
                            subplot_index += 1

                            tit = '    Cf. Smoothed ({})'.format(cf_info_rate_2d)
                            ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index, polar=polar_mode)
                            plot_ratemaps(ax, cf_smoothed_rm, cf_occ_2d, occupancy_thresh_2d, tit, polar_mode, color_map,
                                          y_ticks, y_labels)
                            subplot_index += 1

                    # plot sel motion maps
                    if include_self_motion:
                        print('plotting self motion maps for comparing ...')
                        for swb in range(len(data['settings']['selfmotion_window_size'])):
                            if len(data['settings']['selfmotion_window_size']) == 1:
                                ixess = ixess.reshape((len(ixess), 1))
                                jyess = jyess.reshape((len(jyess), 1))
                            cf_raw_rm_sf, cf_smoothed_rm_sf, cf_occ_sf = get_self_motion_map(
                                cf_dxs[:, swb], cf_dys[:, swb], self_motion_par, tracking_validating, cell_frames,
                                frame_rate, smoothing_par_2d, s_ind)

                            cf_info_rate_sf = calculate_skinfo_rate(smoothed_rm_sf, occ_sf)

                            rate_maps_data['{}-self_motion_{}-raw_rm'.format(da_cell_name, swb)] = raw_rm_sf
                            rate_maps_data['{}-self_motion_{}-smoothed_rm'.format(da_cell_name, swb)] = smoothed_rm_sf
                            rate_maps_data['{}-self_motion_{}-occ'.format(da_cell_name, swb)] = occ_sf
                            rate_maps_data['{}-self_motion_{}-info_rate'.format(da_cell_name, swb)] = info_rate_sf

                            tit = 'Self motion {}'.format(data['settings']['selfmotion_window_size'][swb])
                            ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index)
                            plot_ratemaps(ax, raw_rm_sf, occ_sf, occupancy_thresh_2d, tit, False, color_map,
                                          None, None)
                            subplot_index += 1

                            tit = '    Smoothed ({})'.format(info_rate_sf)
                            ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index)
                            plot_ratemaps(ax, smoothed_rm_sf, occ_sf, occupancy_thresh_2d, tit, False, color_map,
                                          None, None)
                            subplot_index += 1

                            tit = 'Cf. Self motion {}'.format(data['settings']['selfmotion_window_size'][swb])
                            ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index)
                            plot_ratemaps(ax, cf_raw_rm_sf, cf_occ_sf, occupancy_thresh_2d, tit, False, color_map,
                                          None, None)
                            subplot_index += 1

                            tit = '    Cf. Smoothed ({})'.format(cf_info_rate_sf)
                            ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index)
                            plot_ratemaps(ax, cf_smoothed_rm_sf, cf_occ_sf, occupancy_thresh_2d, tit, False, color_map,
                                          None, None)
                            subplot_index += 1

                    # plot velocity tuning
                    if include_velocity_map:
                        print('plotting velocity maps for comparing ...')
                        cf_body_dir_angles = data['cf_possiblecovariates']['C Body_direction'] * math.pi / 180
                        cf_speeds = data['cf_possiblecovariates']['B Speeds']

                        cf_raw_rm_vc, cf_smoothed_rm_vc, cf_occ_vc, speed_bins = get_velocity_map(
                            cf_body_dir_angles, cf_speeds, velocity_par, tracking_validating, cell_frames, frame_rate,
                            factor_torus['C Body_direction'], smoothing_par_2d, s_ind)

                        cf_info_rate_vc = calculate_skinfo_rate(cf_smoothed_rm_vc, cf_occ_vc)

                        xval2 = np.round(np.diff(speed_bins) * 0.5 + speed_bins[:-1], 2)
                        xval2 = np.append(xval2, np.round(speed_bins[-1], 2))
                        y_ticks = np.linspace(0, len(speed_bins) - 1, 6).astype(int)
                        y_labels = xval2[y_ticks]

                        tit = 'Velocity'
                        ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index, polar=pie_style)
                        plot_ratemaps(ax, raw_rm_vc.T, occ_vc.T, occupancy_thresh_2d, tit, pie_style, color_map,
                                      y_ticks, y_labels)
                        subplot_index += 1

                        tit = '    Smoothed ({})'.format(info_rate_vc)
                        ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index, polar=pie_style)
                        plot_ratemaps(ax, smoothed_rm_vc.T, occ_vc.T, occupancy_thresh_2d, tit, pie_style, color_map,
                                      y_ticks, y_labels)
                        subplot_index += 1

                        tit = 'Cf. Velocity'
                        ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index, polar=pie_style)
                        plot_ratemaps(ax, cf_raw_rm_vc.T, cf_occ_vc.T, occupancy_thresh_2d, tit, pie_style, color_map,
                                      y_ticks, y_labels)
                        subplot_index += 1

                        tit = '    Cf. Smoothed ({})'.format(cf_info_rate_vc)
                        ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index, polar=pie_style)
                        plot_ratemaps(ax, cf_smoothed_rm_vc.T, cf_occ_vc.T, occupancy_thresh_2d, tit, pie_style,
                                      color_map, y_ticks, y_labels)
                        subplot_index += 1

                    # plot spatial maps
                    if include_spatial_map:
                        print('plotting spatial maps for comparing ...')
                        cf_raw_rm_sp, cf_smoothed_rm_sp, cf_occ_sp = get_spatial_map(
                            cf_animal_loc, spatial_par, tracking_validating, cell_frames, frame_rate, smoothing_par_2d,
                            s_ind)

                        cf_info_rate_sp = calculate_skinfo_rate(cf_smoothed_rm_sp, cf_occ_sp)

                        tit = 'Space'
                        ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index)
                        plot_ratemaps(ax, raw_rm_sp, occ_sp, occupancy_thresh_2d, tit, False, color_map,
                                      None, None)
                        subplot_index += 1

                        tit = '    Smoothed ({})'.format(info_rate_sp)
                        ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index)
                        plot_ratemaps(ax, smoothed_rm_sp, occ_sp, occupancy_thresh_2d, tit, False, color_map,
                                      None, None)
                        subplot_index += 1

                        tit = 'Cf. Space'
                        ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index)
                        plot_ratemaps(ax, cf_raw_rm_sp, cf_occ_sp, occupancy_thresh_2d, tit, False, color_map,
                                      None, None)
                        subplot_index += 1

                        tit = '    Cf. Smoothed ({})'.format(cf_info_rate_sp)
                        ax = plt.subplot(cf_plot_rows, plot_cols, subplot_index)
                        plot_ratemaps(ax, cf_smoothed_rm_sp, cf_occ_sp, occupancy_thresh_2d, tit, False, color_map,
                                      None, None)
                        subplot_index += 1

                    while np.mod(subplot_index - 1, 10) != 0:
                        plt.subplot(cf_plot_rows, plot_cols, subplot_index)
                        plt.axis('off')
                        subplot_index += 1

                    # start 1d plots
                    print('plotting 1D rate maps for comparing ...')

                    cf_axis_list = []
                    cf_info_rates_vec = []
                    for j in range(n_keys_1d):
                        da_key = keys_factor_1d[j]
                        if da_key not in cf_keys_factor_1d:
                            continue

                        cf_occ = binned_factors_occ[da_key]
                        cf_xvals, cf_rawrm, cf_smrm = get_1d_ratemap(
                            cf_factor1d[da_key], cf_bins1d[da_key], cf_occ, cell_frames,
                            factor_torus[da_key], smoothing_par_1d)

                        cf_info_rate = calculate_skinfo_rate(cf_rawrm, cf_occ)
                        cf_info_rates_vec.append(cf_info_rate)

                        cf_xvals, cf_rawrm_p1, cf_smrm_p1 = get_1d_ratemap(
                            cf_factor1d[da_key], cf_bins1d[da_key], p1_binned_occ[da_key],
                            cell_frames_p1, factor_torus[da_key], smoothing_par_1d)

                        cf_xvals, cf_rawrm_p2, cf_smrm_p2 = get_1d_ratemap(
                            cf_factor1d[da_key], cf_bins1d[da_key], p2_binned_occ[da_key],
                            cell_frames_p2, factor_torus[da_key], smoothing_par_1d)

                        cf_rate_map_1d_data = {'bin_center': xvals, 'raw_firing_rate': rawrm, 'occupancy': occ,
                                               'smoothed_firing_rate': smrm,
                                               'shuffle_mean': cf_shuffled_means[da_key][:, cellnum],
                                               'shuffle_std': cf_shuffled_stds[da_key][:, cellnum],
                                               'shuffle_lower_quant': cf_shuffled_lower[da_key][:, cellnum],
                                               'shuffle_upper_quant': cf_shuffled_upper[da_key][:, cellnum],
                                               'raw_firing_rate_p1': rawrm_p1, 'smoothed_firing_rate_p1': smrm_p1,
                                               'occupancy_p1': p1_binned_occ[da_key], 'raw_firing_rate_p2': rawrm_p2,
                                               'smoothed_firing_rate_p2': smrm_p2, 'occupancy_p2': p2_binned_occ[da_key]}

                        cf_rate_maps_data['{}-{}-data_1d'.format(cell_names[cellnum], da_key[2:])] = cf_rate_map_1d_data
                        cf_rate_maps_data['{}-{}-info_rate'.format(cell_names[cellnum], da_key[2:])] = cf_info_rate

                        rate_map_1d_data = rate_maps_data['{}-{}-data_1d'.format(cell_names[cellnum], da_key[2:])]
                        info_rate = rate_maps_data['{}-{}-info_rate'.format(cell_names[cellnum], da_key[2:])]

                        xvals = rate_map_1d_data['bin_center']
                        rawrm = rate_map_1d_data['raw_firing_rate']
                        occ = rate_map_1d_data['occupancy']
                        smrm = rate_map_1d_data['smoothed_firing_rate']
                        shuffle_mean_1d = rate_map_1d_data['shuffle_mean']
                        shuffle_std_1d = rate_map_1d_data['shuffle_std']
                        shuffle_lq_1d = rate_map_1d_data['shuffle_lower_quant']
                        shuffle_uq_1d = rate_map_1d_data['shuffle_upper_quant']
                        rawrm_p1 = rate_map_1d_data['raw_firing_rate_p1']
                        smrm_p1 = rate_map_1d_data['smoothed_firing_rate_p1']
                        occ_p1 = rate_map_1d_data['occupancy_p1']
                        rawrm_p2 = rate_map_1d_data['raw_firing_rate_p2']
                        smrm_p2 = rate_map_1d_data['smoothed_firing_rate_p2']
                        occ_p2 = rate_map_1d_data['occupancy_p2']

                        cf_axis_list.append(plt.subplot(cf_plot_rows, plot_cols, subplot_index))
                        tit = da_key[2:].replace('_', ' ')
                        tit = '{} ({})'.format(tit, info_rate)
                        plot_1d_values(xvals, smrm, rawrm, occ, shuffle_mean_1d, shuffle_std_1d, shuffle_lq_1d,
                                       shuffle_uq_1d, occupancy_thresh_1d, tit, use_quantile)
                        valid_p1 = (occ_p1 > occupancy_thresh_1d)
                        plt.plot(xvals[valid_p1], smrm_p1[valid_p1], 'o', color='green')
                        plt.plot(xvals[valid_p1], rawrm_p1[valid_p1], '+', color='green')
                        valid_p2 = (occ_p2 > occupancy_thresh_1d)
                        plt.plot(xvals[valid_p2], smrm_p2[valid_p2], 'o', color='red')
                        plt.plot(xvals[valid_p2], rawrm_p2[valid_p2], '+', color='red')
                        plt.xlabel(xaxis1d[da_key])
                        subplot_index += 1

                        if max_ylim < cf_axis_list[j].get_ylim()[1]:
                            max_ylim = cf_axis_list[j].get_ylim()[1]

                        plt.subplot(cf_plot_rows, plot_cols, subplot_index)
                        vals = factor1d[da_key]
                        plt.hist(vals[~np.isnan(vals)], 40)
                        subplot_index += 1

                        cf_axis_list.append(plt.subplot(cf_plot_rows, plot_cols, subplot_index))
                        tit = da_key[2:].replace('_', ' ')
                        tit = 'Cf. {} ({})'.format(tit, info_rate)
                        plot_1d_values(cf_xvals, cf_smrm, cf_rawrm, cf_occ, cf_shuffled_means[da_key][:, cellnum],
                                       cf_shuffled_stds[da_key][:, cellnum], cf_shuffled_lower[da_key][:, cellnum],
                                       cf_shuffled_upper[da_key][:, cellnum], occupancy_thresh_1d, tit, use_quantile)
                        valid_p1 = (p1_binned_occ[da_key] > occupancy_thresh_1d)
                        plt.plot(cf_xvals[valid_p1], cf_smrm_p1[valid_p1], 'o', color='green')
                        plt.plot(cf_xvals[valid_p1], cf_rawrm_p1[valid_p1], '+', color='green')
                        valid_p2 = (p2_binned_occ[da_key] > occupancy_thresh_1d)
                        plt.plot(cf_xvals[valid_p2], cf_smrm_p2[valid_p2], 'o', color='red')
                        plt.plot(cf_xvals[valid_p2], cf_rawrm_p2[valid_p2], '+', color='red')
                        plt.xlabel(xaxis1d[da_key])
                        subplot_index += 1

                        if max_ylim < cf_axis_list[j].get_ylim()[1]:
                            max_ylim = cf_axis_list[j].get_ylim()[1]

                        plt.subplot(cf_plot_rows, plot_cols, subplot_index)
                        vals = cf_factor1d[da_key]
                        plt.hist(vals[~np.isnan(vals)], 40)
                        subplot_index += 1

                        if limit_y:
                            for k in range(len(cf_axis_list)):
                                cf_axis_list[k].set_ylim([0, good_max_ylim])

                    cf_sort_order = np.argsort(cf_info_rates_vec)[::-1]
                    cf_best_fac = np.ravel(keys_factor_1d)[cf_sort_order]
                    for b_ind in range(4):
                        cf_best_fac[b_ind] = cf_best_fac[b_ind][2:].replace('_', ' ')

                    # fig title
                    tit = "{} : {} : {}\nTop 1D info rates from original: {}, {}, {}, {}. \n " \
                          "from compare: {}, {}, {}, {}".format(
                        output_file_prefix, cell_names[cellnum], temp_offsets[i],
                        best_fac[0], best_fac[1], best_fac[2], best_fac[3],
                        cf_best_fac[0], cf_best_fac[1], cf_best_fac[2], cf_best_fac[3])

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.gcf().suptitle(tit, fontsize=40)
                    if abs(temp_offsets[i] - 0) < 0.001:
                        plt.savefig(
                            '%s_%04d_%s_at_zero_comparing.png' % (output_file_prefix, print_cell, cell_names[cellnum]))

                pdf.savefig()
                plt.close()



    print("\n")
    print("         \\|||||/        ")
    print("         ( O O )         ")
    print("|--ooO-----(_)----------|")
    print("|                       |")
    print("|   Rate Maps Finished  |")
    print("|                       |")
    print("|------------------Ooo--|")
    print("         |__||__|        ")
    print("          ||  ||         ")
    print("         ooO  Ooo        ")








