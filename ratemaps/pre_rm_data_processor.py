
import os
import sys
import scipy.io
import numpy as np
import math

import pickle
from scipy.optimize import minimize


def get_time_split(data, use_even_odd_minutes=True):
    session_ts = data['session_ts']
    if use_even_odd_minutes:
        print('calculating even odd minutes ....')
        otherbins = np.arange(session_ts[0], session_ts[1] + 60., 60)
        otherbins[otherbins > session_ts[1]] = session_ts[1]
        if abs(otherbins[-1] - otherbins[-2]) < 1:
            otherbins = otherbins[:(-1)]
        if abs(otherbins[-1] - otherbins[-2]) < 59.99999999999999999:
            print('FYI: one of the time chunks that should be one minute long is really just {} '
                  'seconds and is currently included in the analysis'.format(abs(otherbins[-1] - otherbins[-2])))
        startaltbins = otherbins[:(-1)]
        endaltbins = otherbins[1:]
    else:
        startaltbins = np.ravel(np.array([session_ts[0], 0.5 * (session_ts[0] + session_ts[1])]))
        endaltbins = np.ravel(np.array([0.5 * (session_ts[0] + session_ts[1]), session_ts[1]]))

    return startaltbins, endaltbins


def make_angles_continue(angs, degrees=True):
    if degrees:
        svec = np.array([-180, 0, 180])
    else:
        svec = np.array([-180, 0, 180]) / 180 * np.pi
    processed_ang = angs[~np.isnan(angs)].copy()
    n_ang = len(processed_ang)
    shift_vec = np.zeros(n_ang)
    final_ang = np.zeros(len(angs))
    final_ang[:] = np.nan
    temp_ang = np.zeros(n_ang)
    temp_ang[:] = np.nan
    temp_ang[0] = processed_ang[0]
    # diff_ang = np.diff(processed_ang)
    for i in range(1,n_ang):
        diff_ang = processed_ang[i] - processed_ang[i-1]
        d_vec = diff_ang + svec
        idx = np.argmin(abs(d_vec))
        shift_vec[i] = shift_vec[i-1] + svec[idx]
        temp_ang[i] = processed_ang[i] + shift_vec[i]
    final_ang[~np.isnan(angs)] = temp_ang
    return final_ang


def calculate_1st_derivatives(values: np.ndarray, time_width: np.ndarray, param: int, periodic: bool, method='cfd'):
    """
    calculate the 1st order differentiations of a single factor
    :param values: array(num frames)
    :param time_width:
    :param param: int, one side step width, number of frames before and after the current frame
    :param periodic: bool
    :param method: string, default 'cfd' central finite difference, could be 'fps' five-point stencil also
    :param sid: session_indicator
    :return:
    """
    nf = len(values)
    value_der = np.zeros(nf)
    value_der[:] = np.nan

    exceed_ind = []
    if method == 'fps':
        for t in range(nf):
            tt = t + np.array([- 2 * param, - param, param, 2 * param])
            if np.any(tt[:2] < 0) or np.any(tt[2:] > nf - 1) or np.any(np.isnan(values[tt])):
                continue
            value_der[t] = - values[tt[3]] + 8 * values[tt[2]] - 8 * values[tt[1]] + values[tt[0]]
            if periodic:
                value_der[t] = value_der[t] - 360 if value_der[t] > 180 else value_der[t]
                value_der[t] = value_der[t] + 360 if value_der[t] < -180 else value_der[t]

            value_der[t] = value_der[t] / (12 * time_width[t])
            if periodic and np.logical_or(value_der[t] > 180, value_der[t] < -180):
                exceed_ind.append(t)
    else:
        for t in range(nf):
            ts = t - param
            te = t + param
            if ts < 0 or te > nf - 1 or np.isnan(values[ts]) or np.isnan(values[te]):
                continue
            value_der[t] = (values[te] - values[ts])
            if periodic:
                value_der[t] = value_der[t] - 360 if value_der[t] > 180 else value_der[t]
                value_der[t] = value_der[t] + 360 if value_der[t] < -180 else value_der[t]
            value_der[t] = value_der[t] / (2 * time_width[t])
            if periodic and np.logical_or(value_der[t] > 180, value_der[t] < -180):
                exceed_ind.append(t)

    return value_der, exceed_ind


def calculate_2nd_derivatives(values_1st: np.ndarray, time_width: np.ndarray, param: int, periodic: bool, method='cfd'):
    """
    calculate the 1st and 2nd order differentiations of a single factor
    :param values: array(num frames)
    :param time_width:
    :param param: int, one side step width, number of frames before and after the current frame
    :param periodic: bool
    :param method: string, default 'cfd' central finite difference, could be 'fps' five-point stencil also
    :param sid: session_indicator
    :return:
    """

    nf = len(values_1st)
    value_der = np.zeros(nf)
    value_der[:] = np.nan

    value_der, exceed_ind = calculate_1st_derivatives(values_1st, time_width, param, False, method)
    # exceed_ind = []
    # if method == 'fps':
    #     for t in range(nf):
    #         tt = t + np.array([- 2 * param, - param, param, 2 * param])
    #         if np.any(tt[:2] < 0) or np.any(tt[2:] > nf - 1) or np.any(np.isnan(val[tt])):
    #             continue
    #         value_der[t] = (- val[tt[3]] + 16 * val[tt[2]] - 30 * val[t] + 16 * val[tt[1]] - val[tt[0]]) / (
    #                     12 * time_width[t] * time_width[t])
    #         if periodic and np.logical_or(value_der[t] > 180, value_der[t] < -180):
    #             exceed_ind.append(t)
    # else:
    #     for t in range(nf):
    #         ts = t - param
    #         te = t + param
    #         if ts < 0 or te > nf - 1 or np.isnan(val[ts]) or np.isnan(val[te]) or np.isnan(val[t]):
    #             continue
    #         value_der[t] = (val[te] - 2 * val[t] + val[ts]) / (2 * time_width[t] * time_width[t])
    #         if periodic and np.logical_or(value_der[t] > 180, value_der[t] < -180):
    #             exceed_ind.append(t)

    return value_der, exceed_ind


def get_derivatives(factors, bounds, xaxis, framerate, param, method='cfd', session_indicator=None, avoid_2nd=True):
    all_keys = list(factors.keys())
    nkeys = len(all_keys)

    if len(np.ravel(framerate)) == 1:
        time_width1 = param[0] / framerate
        time_width2 = param[1] / framerate
    else:
        time_width1 = np.zeros(len(session_indicator))
        time_width2 = np.zeros(len(session_indicator))
        for t in range(len(session_indicator)):
            time_width1[t] = param[0] / framerate[session_indicator[t]]
            time_width2[t] = param[1] / framerate[session_indicator[t]]

    for j in range(nkeys):
        tkey = all_keys[j]
        print(tkey)
        vals = factors[tkey]
        nvals = len(vals)

        if len(np.ravel(framerate)) == 1:
            tw1 = np.repeat(time_width1, nvals)
            tw2 = np.repeat(time_width2, nvals)
        else:
            tw1 = time_width1.copy()
            tw2 = time_width2.copy()

        first_der, exceed_ind1 = calculate_1st_derivatives(vals, tw1, param[0], bounds[tkey][2], method)

        factors['%s_1st_der' % tkey] = first_der + 0.
        bounds['%s_1st_der' % tkey] = [0, 0, bounds[tkey][2]]
        if bounds[tkey][2]:
            xaxis['%s_1st_der' % tkey] = 'degrees per second'
        else:
            xaxis['%s_1st_der' % tkey] = 'cm per second'

        if avoid_2nd:
            continue

        if 'Speeds' in tkey:
            continue

        second_der, exceed_ind2 = calculate_2nd_derivatives(first_der, tw2, param[1], bounds[tkey][2], method)
        factors['%s_2nd_der' % tkey] = second_der + 0.
        bounds['%s_2nd_der' % tkey] = [0, 0, bounds[tkey][2]]
        if bounds[tkey][2]:
            xaxis['%s_2nd_der' % tkey] = 'degrees per second squared'
        else:
            xaxis['%s_2nd_der' % tkey] = 'cm per second squared'
    return factors, bounds, xaxis


# get super bounds!!!
def get_bins_with_enough_in_each(values, minval, maxval, num_bins_1d, frame_rate, session_ind):
    bins = np.linspace(minval, maxval, num_bins_1d + 1)
    occupancy = np.zeros(len(bins) - 1)
    invalid_ind = np.isnan(values)
    values = values[~invalid_ind]
    if session_ind is not None:
        valid_session_ind = session_ind[~invalid_ind]
    else:
        valid_session_ind = np.zeros(len(values), 'i')
    for i in range(1, len(bins), 1):
        if len(np.ravel(frame_rate)) == 1:
            occupancy[i - 1] = float(np.sum((values > bins[i - 1]) * (values <= bins[i]))) / float(
                frame_rate)  # in seconds
        else:
            total_mat = []
            for j in range(len(frame_rate)):
                scount = (valid_session_ind == j)
                part_mat = float(np.sum((values > bins[i - 1]) * (values <= bins[i]) * scount)) / float(
                    frame_rate[j])  # in seconds
                total_mat.append(part_mat)
            total_num = np.sum(total_mat)
            occupancy[i - 1] = total_num

    return bins, occupancy


def get_super_bounds(factors, bounds, num_bins_1d, occupancy_thresh_1d, capture_frame_rate, sid):
    all_da_bins = {}
    all_keys = list(factors.keys())
    for i in range(len(all_keys)):
        da_key = all_keys[i]
        print(da_key)
        values = factors[da_key]  # values
        ips = bounds[da_key]  # bound

        minval = ips[0]  # lower limit of the bound
        maxval = ips[1]  # upper limit of the bound
        if abs(minval) < 0.00001 and abs(maxval) < 0.00001:
            minval = np.nanmin(values)
            maxval = np.nanmax(values)
            for j in range(10000):
                bins, occ = get_bins_with_enough_in_each(values, minval, maxval, num_bins_1d, capture_frame_rate, sid)
                goodtogo = True
                if occ[0] < occupancy_thresh_1d:
                    minval = minval + 0.05 * (bins[1] - bins[0])
                    goodtogo = False
                if occ[-1] < occupancy_thresh_1d:
                    maxval = maxval - 0.05 * (bins[-1] - bins[-2])
                    goodtogo = False
                if goodtogo:
                    break
                if not goodtogo and j > 9999:
                    for k in range(10):
                        print(('SHIT! Could not find good bounds for the variable!!', da_key))

        bins, occ = get_bins_with_enough_in_each(values, minval, maxval, num_bins_1d, capture_frame_rate, sid)
        all_da_bins[da_key] = bins + 0.
        bounds[da_key][0] = bins[0]
        bounds[da_key][1] = bins[-1]
    return all_da_bins, bounds


def print_self_motion_info(full_data, settings):
    num_par = len(settings['selfmotion_window_size'])
    for i in range(num_par):
        print('To get all the values of the movement vectors for window size {}, '
              'you would need a min and max horizontal value of {} {} and '
              'a min and max vertical value of {} {} for the second'.format(
            settings['selfmotion_window_size'][i], np.nanmin(full_data['dxs'][:, i]), np.nanmax(full_data['dxs'][:, i]),
            np.nanmin(full_data['dys'][:, i]), np.nanmax(full_data['dys'][:, i])))


def get_default_1d_factor(data):
    factor1d = {'B Speeds': data['speed_vec'],
                'C Body_direction': data['body_direction'],
                'D Allo_head_direction': data['allo_head_ang'][:, 2],
                'G Neck_elevation': data['animal_loc'][:, 2] * 100,
                'K Ego3_Head_roll': data['ego3_head_ang'][:, 0],
                'L Ego3_Head_pitch': data['ego3_head_ang'][:, 1],
                'M Ego3_Head_azimuth': data['ego3_head_ang'][:, 2],
                'N Back_pitch': data['opt_back_ang'][:, 0],
                'O Back_azimuth': data['opt_back_ang'][:, 1],
                'P Ego2_head_roll': data['ego2_head_ang'][:, 0],
                'Q Ego2_head_pitch': data['ego2_head_ang'][:, 1],
                'R Ego2_head_azimuth': data['ego2_head_ang'][:, 2]}
    return factor1d


def get_default_1d_bounds():
    # zeros mean optimize it for me!! So it is min, max, periodic (True or False)
    bounds1d = {'B Speeds': [0, 0, False],
                'C Body_direction': [-180, 180, True],
                'D Allo_head_direction': [-180, 180, True],
                'G Neck_elevation': [0, 0, False],
                'K Ego3_Head_roll': [-180, 180, True],
                'L Ego3_Head_pitch': [-180, 180, True],
                'M Ego3_Head_azimuth': [-180, 180, True],
                'N Back_pitch': [-60, 60, True],
                'O Back_azimuth': [-60, 60, True],
                'P Ego2_head_roll': [-180, 180, True],
                'Q Ego2_head_pitch': [-180, 180, True],
                'R Ego2_head_azimuth': [-180, 180, True]}
    return bounds1d


def get_default_1d_axis():
    axis1d = {'B Speeds': 'cm per second',
               'C Body_direction': 'angles',
               'D Allo_head_direction': 'angles',
               'G Neck_elevation': 'cm',
               'K Ego3_Head_roll': 'ccw --- angles --- cw',
               'L Ego3_Head_pitch': 'down --- angles --- up',
               'M Ego3_Head_azimuth': 'left --- angles --- right',
               'N Back_pitch': 'down --- angles --- up',
               'O Back_azimuth': 'left --- angles --- right',
               'P Ego2_head_roll': 'ccw --- angles --- cw',
               'Q Ego2_head_pitch': 'down --- angles --- up',
               'R Ego2_head_azimuth': 'left --- angles --- right'}

    return axis1d


def filter_animal_spacial_location(factor1d, dxs, dys, animal_loc, bbtrans_xy, spatial_diameter):
    rad_squared = (spatial_diameter / 2.) ** 2
    dd = (animal_loc[:, 0] - bbtrans_xy[0]) ** 2 + (animal_loc[:, 1] - bbtrans_xy[1]) ** 2
    beyond_radius = dd > rad_squared

    if np.sum(beyond_radius) > 0:
        inds = beyond_radius
        animal_loc[inds] = np.nan
        dxs[inds] = np.nan
        dys[inds] = np.nan

        all_keys = list(factor1d.keys())
        for da_key in all_keys:
            vals = factor1d[da_key]
            vals[inds] = np.nan
            factor1d[da_key] = vals

    num = np.sum(~np.isnan(animal_loc[:, 0]))
    print('After filtering, the remaining tracked data is (for the neck point) {}, '
          'which is {} percent of the total data length'.format(num, 100. * num / float(len(animal_loc[:, 0]))))

    return factor1d, dxs, dys, animal_loc


def filter_speed(factor1d, dxs, dys, animal_loc, speed_range):
    below_thresh = factor1d['B Speeds'] < speed_range[0]
    above_thresh = factor1d['B Speeds'] > speed_range[1]
    all_keys = list(factor1d.keys())

    if np.sum(below_thresh) > 0:
        inds = below_thresh
        animal_loc[inds] = np.nan
        dxs[inds] = np.nan
        dys[inds] = np.nan

        for da_key in all_keys:
            vals = factor1d[da_key]
            vals[inds] = np.nan
            factor1d[da_key] = vals

    if np.sum(above_thresh) > 0:
        inds = above_thresh
        animal_loc[inds] = np.nan
        dxs[inds] = np.nan
        dys[inds] = np.nan

        for da_key in all_keys:
            vals = factor1d[da_key]
            vals[inds] = np.nan
            factor1d[da_key] = vals

    return factor1d, dxs, dys, animal_loc


def greater_than_ignore_nan(x, value):
    not_nans = ~np.isnan(x)
    inds = np.zeros(len(x), dtype=bool)
    inds[not_nans] = x[not_nans] > value
    return inds


def less_than_ignore_nan(x, value):
    not_nans = ~np.isnan(x)
    inds = np.zeros(len(x), dtype=bool)
    inds[not_nans] = x[not_nans] < value
    return inds


def filter_1d_factor(factor1d, dxs, dys, animal_loc, factor_name, split_values=(0, 1)):
    print('processing split by 1d factor ...')

    split_val = factor1d[factor_name]
    gt = greater_than_ignore_nan(split_val, split_values[1])
    lt = less_than_ignore_nan(split_val, split_values[0])
    inds = (gt + lt) > 0.5
    print('Num combined is', np.sum((split_val < split_values[0]) + (split_val > split_values[1])))
    print('Num to remove then is', np.sum(inds), 'since the total is', len(inds))
    if np.sum(inds) > 0:
        all_keys = list(factor1d.keys())
        for da_key in all_keys:
            vals = factor1d[da_key]
            vals[inds] = np.nan
            factor1d[da_key] = vals

        animal_loc[inds] = np.nan
        dxs[inds] = np.nan
        dys[inds] = np.nan

    return factor1d, dxs, dys, animal_loc


def get_pre_rm_default_data(data, use_even_odd_minutes=True, speed_type='jump', window_size=250):
    filename = data['file_info']
    print('File to be working', filename)

    # get time split
    startaltbins, endaltbins = get_time_split(data, use_even_odd_minutes)


    # update settings
    settings = data['settings'].copy()
    settings['use_even_odd_minutes'] = use_even_odd_minutes

    # update self_motion data
    num_par = len(settings['selfmotion_window_size'])

    if speed_type == 'jump':
        sfmat_ind = np.arange(2 * num_par)
        speed_ind = np.arange(num_par)
    elif speed_type == 'cum':
        sfmat_ind = np.arange(2 * num_par, 4 * num_par)
        speed_ind = np.arange(num_par, 2 * num_par)
    else:
        raise Exception('Not defined speed definitation !!!')

    full_data = data['matrix_data'][0].copy()
    full_data['animal_loc'] = full_data['sorted_point_data'][:, 4, :]
    full_data['selfmotion_mat'] = full_data['selfmotion'][:, sfmat_ind]
    full_data['speeds_mat'] = full_data['speeds'][:, speed_ind]
    full_data['dxs'] = full_data['selfmotion_mat'][:, 0: 2 * num_par: 2]
    full_data['dys'] = full_data['selfmotion_mat'][:, 1: 2 * num_par: 2]

    if len(data['matrix_data']) == 2:
        print('Two animal data included.')
        exist_animal2 = True
        full_data2 = data['matrix_data'][1].copy()
        full_data2['animal_loc'] = full_data2['sorted_point_data'][:, 4, :]
        full_data2['selfmotion_mat'] = full_data2['selfmotion'][:, sfmat_ind]
        full_data2['speeds_mat'] = full_data2['speeds'][:, speed_ind]
        full_data2['dxs'] = full_data2['selfmotion_mat'][:, 0: 2 * num_par: 2]
        full_data2['dys'] = full_data2['selfmotion_mat'][:, 1: 2 * num_par: 2]
    elif len(data['matrix_data']) == 1:
        exist_animal2 = False
    else:
        raise Exception('data structure is wrong. Seems like no animal or more than 2 animals are stored.')

    # print self_motion info
    print_self_motion_info(full_data, settings)
    if exist_animal2:
        print_self_motion_info(full_data2, settings)

    if window_size not in settings['selfmotion_window_size']:
        print('Possible values for window_size are {}, set to {}.'.format(
            settings['selfmotion_window_size'], settings['selfmotion_window_size'][-1]))
        window_size = settings['selfmotion_window_size'][-1]
    ind4speed = [ind for ind in range(len(settings['selfmotion_window_size'])) if
                 window_size == settings['selfmotion_window_size'][ind]][0]
    full_data['speed_vec'] = full_data['speeds_mat'][:, ind4speed]
    if exist_animal2:
        full_data2['speed_vec'] = full_data2['speeds_mat'][:, ind4speed]

    bounds1d = get_default_1d_bounds()
    xaxis1d = get_default_1d_axis()
    factor1d = get_default_1d_factor(full_data)
    if exist_animal2:
        factor1d_cf = get_default_1d_factor(full_data2)

    # making output file name
    output_file_prefix = filename
    if output_file_prefix[-1] != '_':
        output_file_prefix = output_file_prefix + '_'

    if use_even_odd_minutes:
        output_file_prefix = '%seo' % output_file_prefix
    else:
        output_file_prefix = '%sfs' % output_file_prefix

    mat_data = {'output_file_prefix': output_file_prefix,
                'exist_animal2': exist_animal2,
                'startaltbins': startaltbins,
                'endaltbins': endaltbins,
                'settings': settings,
                'session_indicator': data['session_indicator'],
                'frame_times': data['frame_times'],
                'time_bins': data['time_bins'],
                'framerate': data['framerate'],
                'overall_framerate': data['overall_framerate'],
                'session_ts': data['session_ts'],
                'tracking_ts': data['tracking_ts'],
                'cell_names': data['cell_names'],
                'cell_activities': data['cell_activities'],
                'dxs': full_data['dxs'], 'dys': full_data['dys'], 'animal_location': full_data['animal_loc'],
                'possiblecovariates': factor1d,
                'possiblecovariatesnames': xaxis1d,
                'possiblecovariatesbounds': bounds1d}

    if exist_animal2:
        mat_data['cf_dxs'] = full_data2['dxs']
        mat_data['cf_dys'] = full_data2['dys']
        mat_data['cf_animal_location'] = full_data2['animal_loc']
        mat_data['cf_possiblecovariates'] = factor1d_cf
        mat_data['cf_possiblecovariatesnames'] = xaxis1d
        mat_data['cf_possiblecovariatesbounds'] = bounds1d

    return mat_data


def set_boundaries(boundary, bounds1d):
    mkeys = boundary.keys()
    bkeys = bounds1d.keys()
    for dm_key in mkeys:
        if dm_key in bkeys:
            bounds1d[dm_key] = boundary[dm_key]
    return bounds1d


def add_factor(mat_data, factor_name, factor, bounds, x_axis, aid=1):
    if aid == 1:
        mat_data['possiblecovariates'][factor_name] = factor
        mat_data['possiblecovariatesbounds'][factor_name] = bounds
        mat_data['possiblecovariatesnames'][factor_name] = x_axis
    else:
        mat_data['cf_possiblecovariates'][factor_name] = factor
        mat_data['cf_possiblecovariatesbounds'][factor_name] = bounds
        mat_data['cf_possiblecovariatesnames'][factor_name] = x_axis
    return mat_data


def get_rm_pre_data(data, use_even_odd_minutes=True, speed_type='jump', window_size=250,
                    include_factor=None, method='cfd', derivatives_param=(10, 10), avoid_2nd=True,
                    boundary=None, filter_by_speed=None, filter_by_spatial=None, filter_by_factor=None,
                    num_bins_1d=36, occupancy_thresh_1d=0.401,
                    save_data=True):

    """
    Purpose
    -------------
    generate the data for later use (upload to clusters) for making rate maps.

    Inputs
    -------------
    data : see from data_generator() or merge_comparing_data().

    boundary : if any, see tutorial for example.

    speed_type : 'jump' (default) or 'cum'. Methods used to calculate the speed of the animal.

    derivatives_param : parameters used to calculate derivatives of factors, e.g (10,10).

    window_size : 250 (default). The time interval (ms) we used to calculate speed in self motion.
                       The default selfmotion parameters are (150, 250). So here we can choose 150 or 250.
                       If the selfmotion paramters are changed, here need to change to the corresponding values.

    speed_filter_var : (0, 40).

    spatial_filter_diameter : 1 (default) cm.

    use_even_odd_minutes : True (default),  else first half / second half.

    occupancy_thresh_1d : Minimum bin occupancy (seconds)

    smoothing_par : Width and Height of Gaussian smoothing (bins), Parameters for self motion maps!!


    Outputs
    -------------


    """
    # set default
    is_filtered_by_speed = False
    is_filtered_by_spatial = False
    is_filtered_by_factor = False

    if filter_by_speed is not None:
        if not isinstance(filter_by_speed, tuple) or len(filter_by_speed) != 2:
            raise Exception('filter_by_speed need to be a tuple with length 2.')
        else:
            is_filtered_by_speed = True

    if filter_by_spatial is not None:
        if not isinstance(filter_by_spatial, float):
            raise Exception('filter_by_spatial need to be a float.')
        else:
            is_filtered_by_spatial = True

    if filter_by_factor is not None:
        if not isinstance(filter_by_factor, tuple) or len(filter_by_factor) != 3:
            raise Exception('filter_by_factor need to be a tuple with length 3.')
        else:
            is_filtered_by_factor = True

    if include_factor is not None:
        if not isinstance(include_factor, dict):
            raise Exception('include_factor must be a dict and with keys factor_name, factor, bounds, x_axis, aid.')
        else:
            added_factor_keys = list(include_factor.keys())
            valid_keys = ['factor_name', 'factor', 'bounds', 'x_axis', 'animal_id']
            for i in range(len(valid_keys)):
                if valid_keys[i] not in added_factor_keys:
                    raise Exception('include_factor must have keys factor_name, factor, bounds, x_axis, aid.')

    #

    # check data and get default data
    dt_keys = list(data.keys())
    if 'file_info' not in dt_keys:
        raise Exception('file_info is not in the data !!! Please see data_generator().')

    mat_data = get_pre_rm_default_data(data, use_even_odd_minutes, speed_type, window_size)

    # add new factor
    if include_factor is not None:
        for i in range(len(include_factor['factor_name'])):
            mat_data = add_factor(mat_data, include_factor['factor_name'][i], include_factor['factor'][i],
                                  include_factor['bounds'][i], include_factor['x_axis'][i],
                                  include_factor['animal_id'][i])

    # calculate derivatives
    print('calculate derivatives ....')
    factors1d, bounds1d, xaxis1d = get_derivatives(mat_data['possiblecovariates'],
                                                   mat_data['possiblecovariatesbounds'],
                                                   mat_data['possiblecovariatesnames'],
                                                   mat_data['framerate'],
                                                   derivatives_param, method, mat_data['session_indicator'],
                                                   avoid_2nd)
    # filter
    dxs = mat_data['dxs']
    dys = mat_data['dys']
    animal_loc = mat_data['animal_location']

    if is_filtered_by_speed:
        factors1d, dxs, dys, animal_loc = filter_speed(factors1d, dxs, dys, animal_loc, filter_by_speed)

    if is_filtered_by_spatial:
        factors1d, dxs, dys, animal_loc = filter_animal_spacial_location(
            factors1d, dxs, dys, animal_loc, filter_by_spatial)

    if is_filtered_by_factor:
        factors1d, dxs, dys, animal_loc = filter_1d_factor(
            factors1d, dxs, dys, animal_loc, filter_by_factor[0], (filter_by_factor[1], filter_by_factor[2]))

    # set boundary
    if boundary is not None:
        bounds1d = set_boundaries(boundary, bounds1d)

    # get super bounds and bins
    print('get super bound and bins ....')
    bins1d, bounds1d = get_super_bounds(factors1d, bounds1d, num_bins_1d, occupancy_thresh_1d,
                                        mat_data['framerate'], mat_data['session_indicator'])

    mat_data['possiblecovariates'] = factors1d
    mat_data['possiblecovariatesbounds'] = bounds1d
    mat_data['possiblecovariatesnames'] = xaxis1d
    mat_data['possiblecovariatesbins'] = bins1d
    mat_data['dxs'] = dxs
    mat_data['dys'] = dys
    mat_data['animal_location'] = animal_loc

    if mat_data['exist_animal2']:
        print('calculate derivatives for compare data ....')
        factors1d_cf, bounds1d_cf, xaxis1d_cf = get_derivatives(mat_data['cf_possiblecovariates'],
                                                                mat_data['cf_possiblecovariatesbounds'],
                                                                mat_data['cf_possiblecovariatesnames'],
                                                                mat_data['framerate'],
                                                                derivatives_param, method,
                                                                mat_data['session_indicator'],
                                                                avoid_2nd)

        dxs_cf = mat_data['cf_dxs']
        dys_cf = mat_data['cf_dys']
        animal_loc_cf = mat_data['cf_animal_location']
        # filter
        if is_filtered_by_speed:
            factors1d_cf, dxs_cf, dys_cf, animal_loc_cf = filter_speed(
                factors1d_cf, dxs_cf, dys_cf, animal_loc_cf, filter_by_speed)

        if is_filtered_by_spatial:
            factors1d_cf, dxs_cf, dys_cf, animal_loc_cf = filter_animal_spacial_location(
                factors1d_cf, dxs_cf, dys_cf, animal_loc_cf, filter_by_spatial)

        if is_filtered_by_factor:
            factors1d_cf, dxs_cf, dys_cf, animal_loc_cf = filter_1d_factor(
                factors1d_cf, dxs_cf, dys_cf, animal_loc_cf, filter_by_factor[0],
                (filter_by_factor[1], filter_by_factor[2]))

        # set boundary
        if boundary is not None:
            bounds1d_cf = set_boundaries(boundary, bounds1d_cf)

        print('get super bound and bins for compare data ....')
        bins1d_cf, bounds1d_cf = get_super_bounds(factors1d_cf, bounds1d_cf, num_bins_1d, occupancy_thresh_1d,
                                                  mat_data['framerate'], mat_data['session_indicator'])

        mat_data['cf_possiblecovariates'] = factors1d_cf
        mat_data['cf_possiblecovariatesbounds'] = bounds1d_cf
        mat_data['cf_possiblecovariatesnames'] = xaxis1d_cf
        mat_data['cf_possiblecovariatesbins'] = bins1d_cf
        mat_data['cf_dxs'] = dxs_cf
        mat_data['cf_dys'] = dys_cf
        mat_data['cf_animal_location'] = animal_loc_cf

    # making output file name
    output_file_prefix = mat_data['output_file_prefix']
    if is_filtered_by_speed or is_filtered_by_spatial or is_filtered_by_factor:
        output_file_prefix = '{}_filtered_by'.format(output_file_prefix)

    if is_filtered_by_speed:
        output_file_prefix = '%s_speed_from_%05d_to_%05d' % (output_file_prefix, filter_by_speed[0], filter_by_speed[1])
    if is_filtered_by_spatial:
        output_file_prefix = '%s_spatial_diameter_%05d' % (output_file_prefix, filter_by_spatial)
    if is_filtered_by_factor:
        output_file_prefix = '%s_factor_%s_from_%09d_to_%09d' % (
            output_file_prefix, filter_by_factor[0], int(round(1000. * filter_by_factor[1])),
            int(round(1000. * filter_by_factor[2])))

    if save_data:
        # scipy.io.savemat('ok4rms_%s.mat' % output_file_prefix, mat_data)
        a_file = open('rm_pre_data_%s.pkl' % output_file_prefix, "wb")
        pickle.dump(mat_data, a_file)
        a_file.close()

    name_file = 'rm_pre_data_' + f'{output_file_prefix}' + '.pkl'
    return mat_data, name_file



