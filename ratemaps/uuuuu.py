import numpy as np
import math


def validating_tracking_frame(tracking_times, cell_start_times, cell_end_times, debug_mode=False):
    """
    Validate tracking frames in a given session. Valid means the tracking frame is inside the cell time stamp ranges.
    :param tracking_times: size(1, number_of_frames)
    :param cell_start_times:
    :param cell_end_times:
    :param debug_mode: if True, print information
    :return: a vector of booleans to indicate if a frame is inside the cell times or not
    """

    bool_vec = (tracking_times >= cell_start_times[0]) * (tracking_times <= cell_end_times[0])
    for i in np.arange(1, len(cell_start_times), 1):
        bool_vec += (tracking_times >= cell_start_times[i]) * (tracking_times <= cell_end_times[i])

    if debug_mode:
        print('Tracking start time according to neuralynx {}, \n '
              'First interpolated points {}, Last interpolated points {}, \n '
              'Tracking end time according to neuralynx {}, \n'
              'Those last ones are used for going from time bins to movement bins... \n'
              'consider using smaller bins but then you have to interpolate \n'
              'or record at higher temporal resolution'.format(
            tracking_times[0], tracking_times[:10], tracking_times[-10:], tracking_times[-1]))

    return bool_vec


def validating_cell_data(data, cell_start_times, cell_end_times):
    """
    Validate cell activities in a given session. Valid means the cell activity is inside the cell time stamp ranges.
    :param data: ndarray, single-cell activities. each element is a time point that the cell fires (in seconds).
    :param cell_start_times:
    :param cell_end_times:
    :return: a vector of booleans to indicate if an activity is inside the time range or not.
    """
    valid_ind = (data <= cell_end_times[0]) * (data >= cell_start_times[0])
    for i in np.arange(1, len(cell_start_times), 1):
        valid_ind += (data >= cell_start_times[i]) * (data <= cell_end_times[i])

    return valid_ind


def map_factor_to_bins(factors, bins):
    """

    :param factors: dict, time series of all factor, i.e. speed, head_pitch....
    :param bins: dict, factor bins for all factors, i.e. for head pitch, the bins can be [0-10, 10-20, ...]
    :return:
    """
    binned_factors = {}
    all_keys = list(factors.keys())
    for da_key in all_keys:
        da_factor = factors[da_key]
        da_bin = bins[da_key]
        factor_bin_vals = np.ones(len(da_factor))
        factor_bin_vals[:] = np.nan
        for i in range(1, len(da_bin), 1):
            valid_ind = (da_factor > da_bin[i - 1]) * (da_factor <= da_bin[i])
            factor_bin_vals[valid_ind] = i - 1
        binned_factors[da_key] = factor_bin_vals
    return binned_factors


def get_1d_factor_bin_occ(binned_factors, n_bins, frame_validating, frame_rate, sid):
    """
    This function gets the total time (in seconds) of a factor occupying a factor bin.
    :param binned_factors
    :param n_bins
    :param frame_validating:
    :param frame_rate: int if single session, and vector if artificial merged session
    :param sid: session ids, used when there are artificial merged sessions
    :return:
    """

    binned_occ_dict = {}
    all_keys = list(binned_factors.keys())
    for da_key in all_keys:
        factor_bin_vals = binned_factors[da_key]

        binned_occ = np.zeros(n_bins)
        if len(np.ravel(frame_rate)) == 1:
            for i in range(n_bins):
                binnocc = frame_validating * (factor_bin_vals == i)
                binned_occ[i] = float(np.sum(binnocc)) / float(frame_rate)  # in sec

        else:
            for i in range(n_bins):
                binnocc = frame_validating * (factor_bin_vals == i)
                total_mat = []
                for j in range(len(frame_rate)):
                    scount = (sid == j)
                    part_mat = float(np.sum(binnocc * scount)) / float(frame_rate[j])  # in sec
                    total_mat.append(part_mat)
                binned_occ[i] = np.sum(total_mat)

        binned_occ_dict[da_key] = binned_occ
    return binned_occ_dict


def map_cell_activity_to_tracking_frame(data, time_delay, frame_rate, tracking_ts, time_bins,
                                        cell_start_times, cell_end_times, use_time_bins):
    """
    Map each activity of a single cell to a tracking frame.
    :param data: np.array, single cell activities
    :param time_delay:
    :param frame_rate: overall frame rate
    :param tracking_ts:
    :param time_bins:
    :param cell_start_times:
    :param cell_end_times:
    :param use_time_bins:
    :return: a vector with the same size of data, each element is a tracking frame that the cell fires in the
    corresponding time period
    """
    valid_ind = validating_cell_data(data, cell_start_times, cell_end_times)

    if np.sum(valid_ind) < 1:
        return []

    if not use_time_bins:
        cell_data = data[valid_ind]
        cell_frame = np.round((cell_data - tracking_ts[0] + time_delay) * frame_rate)
    else:
        cell_data = data[valid_ind] + time_delay
        cell_frame = np.ones(len(cell_data), 'i') * -1
        for i in range(len(time_bins) - 1):
            cell_ok = np.logical_and(cell_data >= time_bins[i], cell_data < time_bins[i + 1])
            if len(cell_ok) > 0:
                cell_frame[cell_ok] = i
        if np.any(cell_frame < 0):
            raise ValueError('no cell frame should be smaller than 0, check the code.')
    cell_frame = cell_frame.astype(int)
    return cell_frame


def gaussian_smoothing_1d(raw_acc, smoothing_par, periodic):
    smoothed_acc = raw_acc.copy()
    xb2 = smoothing_par ** 2
    number_of_bins = len(raw_acc)
    if not periodic:
        for x in np.arange(number_of_bins):
            dist = np.exp(-0.5 * (np.arange(number_of_bins) - x) ** 2 / xb2) + 0.
            numer = np.sum(dist * raw_acc)
            denom = np.sum(dist)
            smoothed_acc[x] = numer / denom + 0.
    else:
        xx = np.arange(number_of_bins)
        circle_x = np.arange(3 * number_of_bins)
        circle_y = np.hstack([raw_acc, raw_acc, raw_acc])
        check_ind = xx + number_of_bins
        for i in np.arange(number_of_bins):
            cind = check_ind[i]
            dist = np.exp(-0.5 * (np.arange(len(circle_x)) - circle_x[cind]) ** 2 / xb2) + 0.
            numer = np.sum(dist * circle_y)
            denom = np.sum(dist)
            smoothed_acc[i] = numer / denom + 0.
    return smoothed_acc


def get_1d_ratemap(factor, bins, binned_occ, cell_frame, periodic=False, smoothing_par=1, debug_mode=False):
    """
    get firing rate of a single cell for a 1d factor
    :param factor: time series data of a single factor
    :param bins: the bin bounds of the factor
    :param binned_occ: occupancy in each bin of the factor
    :param cell_frame:
    :param smoothing_par: int, smoothing band. using Gaussain kernel.
    :param periodic: boolean, indicate if the factor is periodical or not.
    :param debug_mode: False (default).
    :return:
    """
    # print 'Fraction of NaNs in the movement data = fraction of bins that are crap', sum(ixes<-1)/float(len(ixes)),
    # this makes a True / False vector of all time points inside the start/end times that we want to analyse
    if isinstance(cell_frame, list):
        if not cell_frame:
            return []

    number_of_bins = len(bins) - 1
    bin_centers = 0.5 * (bins[1:] + bins[:(-1)])

    is_outside_tracking = np.logical_or(cell_frame > len(factor) - 1, cell_frame < 0)
    if np.sum(is_outside_tracking) > 0:
        print('Spikes skipped due to empty spots in tracking', np.sum(is_outside_tracking))
    celldata_inside_tracking = cell_frame[~is_outside_tracking]
    is_nan_tracking = np.isnan(factor[celldata_inside_tracking])
    cell_ok = celldata_inside_tracking[~is_nan_tracking]
    num_cell_ok = len(cell_ok)

    binned_acc_mat = np.zeros((num_cell_ok, number_of_bins))
    for i in range(num_cell_ok):
        ifac = factor[cell_ok[i]]
        binned_acc_mat[i] = np.logical_and(bins[0:number_of_bins] < ifac, ifac <= bins[1:len(bins)]) * 1
        if np.sum(binned_acc_mat[i]) > 1:
            raise Exception('crap, we have an error!!!')

    # number of times that a cell fires in a factor bin
    binned_acc = np.sum(binned_acc_mat, 0)

    # convert to firing rates
    raw_firing_rates = np.zeros(binned_acc.shape)
    raw_firing_rates[binned_occ > 0] = binned_acc[binned_occ > 0] / binned_occ[binned_occ > 0]
    # raw_firing_rates[binned_occ < occ_thresh] = 0.

    # add some gaussian smoothing
    smoothed_firing_rates = gaussian_smoothing_1d(raw_firing_rates, smoothing_par, periodic)

    if debug_mode:
        num_outside_tracking = np.sum(is_outside_tracking)
        num_in_empty_spot = np.sum(is_nan_tracking)
        num_outside_bound = np.sum((np.sum(binned_acc_mat, 1) == 0))
        if num_in_empty_spot > 0:
            print('Spikes skipped due to empty spots in tracking', num_in_empty_spot)
        if num_outside_tracking > 0:
            print('Spikes skipped due to occurring before or after tracking', num_outside_tracking)
        if num_outside_bound > 0:
            print('Spikes skipped due to being outside bounds of rate map', num_outside_bound)

    return bin_centers, raw_firing_rates, smoothed_firing_rates


def get_gauss_kernel(size=(5, 10), center=(3, 6), b2=(1, 2)):
    kernel = np.zeros(size)
    for i in range(size[0]):
        kernel[i] = np.exp(- 0.5 * (i - center[0]) ** 2 / b2[0] - 0.5 * (np.arange(size[1]) - center[1]) ** 2 / b2[1])
    return kernel


def get_2d_ratemap(binned_fac1, binned_fac2, bins_fac1, bins_fac2, frame_validating, cell_frame, frame_rate,
                   smoothing_par, session_indicator,
                   is_fac1_periodic=False, is_fac2_periodic=False, debug_mode=False):
    """
    Calculate the rate maps for 2d factors, and 1d factor combinations.
    :param binned_fac1: vector, binned factor
    :param binned_fac2: vector, binned factor
    :param bins_fac1: bins of the factor
    :param bins_fac2: bins of the factor
    :param frame_validating: boolean vector that indicate each tracking frame is valid of not.
    :param cell_frame: the mapped frame index of each cell activity
    :param frame_rate: frame rate vector
    :param smoothing_par: tuple of length 2, gaussian smoothing bandwidth.
    :param session_indicator:
    :param is_fac1_periodic:
    :param is_fac2_periodic:
    :param debug_mode:
    :return:
    """

    if isinstance(cell_frame, list):
        if not cell_frame:
            return []

    n_frame = len(binned_fac1)
    n_bins1 = len(bins_fac1) - 1
    n_bins2 = len(bins_fac2) - 1
    n_bins = (n_bins1, n_bins2)

    is_outside_tracking = np.logical_or(cell_frame > n_frame - 1, cell_frame < 0)
    if np.sum(is_outside_tracking) > 0:
        print('Spikes skipped due to empty spots in tracking', np.sum(is_outside_tracking))
    valid_cell_frame = cell_frame[~is_outside_tracking]

    binned_occ = np.zeros(n_bins)
    for i in np.arange(n_bins[0]):
        icount = (binned_fac1 == i) * frame_validating
        for j in np.arange(n_bins[1]):
            if len(np.ravel(frame_rate)) == 1:
                binned_occ[i, j] = float(np.sum(icount * (binned_fac2 == j))) / float(frame_rate)  # in seconds
            else:
                total_mat = []
                for k in range(len(frame_rate)):
                    scount = (session_indicator == k)
                    part_mat = float(np.sum(icount * (binned_fac2 == j) * scount)) / float(frame_rate[k])
                    total_mat.append(part_mat)
                binned_occ[i, j] = np.sum(total_mat)

    # calculate the number of times that the cell fires in binA and binB
    empty_spot_fac1 = (binned_fac1 < 0) + (binned_fac1 > n_bins[0] - 1) + np.isnan(binned_fac1)
    empty_spot_fac2 = (binned_fac2 < 0) + (binned_fac2 > n_bins[1] - 1) + np.isnan(binned_fac2)
    binned_acc = np.zeros(n_bins)
    for i in range(len(valid_cell_frame)):
        ii = valid_cell_frame[i]
        if not empty_spot_fac1[ii] and not empty_spot_fac2[ii]:
            binned_acc[int(np.round(binned_fac1[ii])), int(np.round(binned_fac2[ii]))] += 1

    # convert to firing rates
    raw_firing_rate = np.zeros(n_bins)
    raw_firing_rate[binned_occ > 0] = binned_acc[binned_occ > 0] / binned_occ[binned_occ > 0]
    # raw_firing_rate[binned_occ < occupancy_thresh] = 0.

    fr_mat = raw_firing_rate.copy()
    center_x_start = 0
    center_y_start = 0

    if is_fac1_periodic:
        fr_mat = np.vstack([fr_mat, fr_mat, fr_mat])
        center_x_start = n_bins[0]

    if is_fac2_periodic:
        fr_mat = np.hstack([fr_mat, fr_mat, fr_mat])
        center_y_start = n_bins[1]

    gk_size = fr_mat.shape

    smoothed_firing_rate = raw_firing_rate.copy()
    xb2 = smoothing_par[0] ** 2
    yb2 = smoothing_par[1] ** 2
    # slow but doesnt matter so much
    for i in range(n_bins[0]):
        for j in range(n_bins[1]):
            center_x = i + center_x_start
            center_y = j + center_y_start
            dists = get_gauss_kernel(size=(gk_size[0], gk_size[1]), center=(center_x, center_y), b2=(xb2, yb2))
            numer_val = np.sum(dists * fr_mat)
            denom_val = np.sum(dists)
            smoothed_firing_rate[i, j] = numer_val / denom_val + 0.

            # if binned_occ[x, y] > occupancy_thresh:
            #     denom = 0.
            #     numer = 0.
            #     for xx in np.arange(n_bins_fac1):
            #         for yy in np.arange(n_bins_fac2):
            #             if binned_occ[xx, yy] > float(occupancy_thresh):
            #                 ddyy = yy - y  # do this because it is periodic!
            #                 # if(abs(ddyy)>0.5*Ngeneric2Dbins):
            #                 # ddyy = Ngeneric2Dbins-abs(ddyy)
            #                 dist = math.exp(-0.5 * (xx - x) ** 2 / xb2 - 0.5 * ddyy ** 2 / yb2) + 0.
            #                 numer += dist * raw_firing_rate[xx, yy]
            #                 denom += dist
            #     binned_acc[x, y] = numer / denom + 0.

    num_outside_tracking = np.sum(is_outside_tracking)
    num_in_empty_spot = np.sum(np.logical_and(empty_spot_fac1, empty_spot_fac2))
    if debug_mode:
        if num_in_empty_spot > 0:
            print('Spikes skipped due to empty spots in tracking or outside of range', num_in_empty_spot)
        if num_outside_tracking > 0:
            print('Spikes skipped due to occurring before or after tracking', num_outside_tracking)

    return raw_firing_rate, smoothed_firing_rate, binned_occ


# dxs = dxs[:, swb]
# dys = dys[:, swb]
# params = self_motion_par
# frame_validating = tracking_validating
# cell_frame = cell_frames
# smoothing_par = smoothing_par_2d
# session_indicator = s_ind


def get_self_motion_map(dxs, dys, params, frame_validating, cell_frame, frame_rate, smoothing_par,
                        session_indicator, debug_mode=False):

    if isinstance(cell_frame, list):
        if not cell_frame:
            return []

    n_frame = len(dxs)

    is_outside_tracking = np.logical_or(cell_frame > n_frame - 1, cell_frame < 0)
    if np.sum(is_outside_tracking) > 0:
        print('Spikes skipped due to empty spots in tracking', np.sum(is_outside_tracking))
    valid_cell_frame = cell_frame[~is_outside_tracking]

    min_dx, max_dx, min_dy, max_dy, bin_size = params

    min_nbin_x = int(np.ceil(abs(min_dx) / bin_size))
    max_nbin_x = int(np.ceil(abs(max_dx) / bin_size))
    min_nbin_y = int(np.ceil(abs(min_dy) / bin_size))
    max_nbin_y = int(np.ceil(abs(max_dy) / bin_size))

    binned_dx = min_nbin_x + np.floor(dxs / bin_size).astype(int)
    binned_dy = min_nbin_y + np.floor(dys / bin_size).astype(int)

    x_size = min_nbin_x + max_nbin_x
    y_size = min_nbin_y + max_nbin_y

    binned_occ = np.zeros((x_size, y_size))
    for i in range(x_size):
        icount = (binned_dx == i) * frame_validating
        for j in range(y_size):
            if len(np.ravel(frame_rate)) == 1:
                binned_occ[i, j] = float(np.sum(icount * (binned_dy == j))) / float(frame_rate)  # in seconds
            else:
                total_mat = []
                for k in range(len(frame_rate)):
                    scount = (session_indicator == k)
                    part_mat = float(np.sum(icount * (binned_dy == j) * scount)) / float(frame_rate[k])
                    total_mat.append(part_mat)
                binned_occ[i, j] = np.sum(total_mat)

    # calculate the number of times that the cell fires in binA and binB
    empty_spot_dx = (binned_dx < 0) + (binned_dx > x_size - 1)
    empty_spot_dy = (binned_dy < 0) + (binned_dy > y_size - 1)

    binned_acc = np.zeros((x_size, y_size))
    for i in np.arange(len(valid_cell_frame)):
        ii = valid_cell_frame[i]
        if not empty_spot_dx[ii] and not empty_spot_dy[ii]:
            binned_acc[int(np.round(binned_dx[ii])), int(np.round(binned_dy[ii]))] += 1

    # convert to firing rates
    raw_firing_rate = np.zeros((x_size, y_size))
    raw_firing_rate[binned_occ > 0] = binned_acc[binned_occ > 0] / binned_occ[binned_occ > 0]
    # binned_acc[binned_occ < occupancy_thresh] = 0.

    smoothed_firing_rate = raw_firing_rate.copy()
    # add some gaussian smoothing
    xb2 = smoothing_par[0] ** 2
    yb2 = smoothing_par[1] ** 2
    for y in range(y_size):
        for x in range(x_size):
            dists = get_gauss_kernel(size=(x_size, y_size), center=(x, y), b2=(xb2, yb2))
            numer_val = np.sum(dists * raw_firing_rate)
            denom_val = np.sum(dists)
            smoothed_firing_rate[x, y] = numer_val / denom_val + 0.
            # if binned_occ[x, y] > float(occupancy_thresh):
            #     denom = 0.
            #     numer = 0.
            #     for yy in range((nbin_min_yval + nbin_max_yval)):
            #         for xx in range(2 * nbin_max_xval):
            #             if (binned_occ[xx, yy] > float(occupancy_thresh)):
            #                 dist = math.exp(-0.5 * (xx - x) ** 2 / xvar - 0.5 * (yy - y) ** 2 / yvar) + 0.
            #                 numer += dist * rawbinned_acc[xx, yy]
            #                 denom += dist
            #     binned_acc[x, y] = numer / denom + 0.

    num_outside_tracking = np.sum(is_outside_tracking * 1)
    num_in_empty_spot = np.sum(np.logical_and(empty_spot_dx, empty_spot_dy))
    if debug_mode:
        if num_in_empty_spot > 0:
            print('Spikes skipped due to empty spots in tracking or outside of range', num_in_empty_spot)
        if num_outside_tracking > 0:
            print('Spikes skipped due to occurring before or after tracking', num_outside_tracking)

    return raw_firing_rate, smoothed_firing_rate, binned_occ


def get_spatial_map(spatial_values, n_bins, frame_validating, cell_frame, frame_rate,
                    smoothing_par, session_indicator=None, debug_mode=False):

    if isinstance(cell_frame, list):
        if not cell_frame:
            return []

    n_frames = len(spatial_values)

    is_outside_tracking = np.logical_or(cell_frame > n_frames - 1, cell_frame < 0)
    if np.sum(is_outside_tracking) > 0:
        print('Spikes skipped due to empty spots in tracking', np.sum(is_outside_tracking))
    valid_cell_frame = cell_frame[~is_outside_tracking]

    x_min = np.nanmin(spatial_values[:, 0])
    x_max = np.nanmax(spatial_values[:, 0])
    x_bins = np.linspace(x_min, x_max, n_bins + 1)
    y_min = np.nanmin(spatial_values[:, 1])
    y_max = np.nanmax(spatial_values[:, 1])
    y_bins = np.linspace(y_min, y_max, n_bins + 1)
    binned_x = np.ones(n_frames)
    binned_y = np.ones(n_frames)
    for i in range(n_bins - 1):
        valid_ind = (spatial_values[:, 0] >= x_bins[i]) * (spatial_values[:, 0] < x_bins[i + 1])
        binned_x[valid_ind] = i
    for i in range(n_bins - 1):
        valid_ind = (spatial_values[:, 1] >= y_bins[i]) * (spatial_values[:, 1] < y_bins[i + 1])
        binned_y[valid_ind] = i

    binned_occ = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        icount = (binned_x == i) * frame_validating
        for j in range(n_bins):
            if len(np.ravel(frame_rate)) == 1:
                binned_occ[i, j] = float(np.sum(icount * (binned_y == j))) / float(frame_rate)  # in seconds
            else:
                total_mat = []
                for k in range(len(frame_rate)):
                    scount = (session_indicator == k)
                    part_mat = float(np.sum(icount * (binned_y == j) * scount)) / float(frame_rate[k])
                    total_mat.append(part_mat)
                total_num = np.sum(total_mat)
                binned_occ[i, j] = np.sum(total_num)

    binned_acc = np.zeros((n_bins, n_bins))
    is_empty_spot = np.isnan(spatial_values[:, 1])
    for i in np.arange(len(valid_cell_frame)):
        ii = valid_cell_frame[i]
        if not is_empty_spot[ii]:
            binned_acc[int(np.round(binned_x[ii])), int(np.round(binned_y[ii]))] += 1

    # convert to firing rates
    raw_firing_rate = np.zeros((n_bins, n_bins))
    raw_firing_rate[binned_occ > 0] = binned_acc[binned_occ > 0] / binned_occ[binned_occ > 0]
    # binnedacc[binnedocc < occupancy_thresh] = 0.

    smoothed_firing_rate = raw_firing_rate.copy()
    # add some gaussian smoothing
    xb2 = smoothing_par[0] ** 2
    yb2 = smoothing_par[1] ** 2
    for y in range(n_bins):
        for x in range(n_bins):
            dists = get_gauss_kernel(size=(n_bins, n_bins), center=(x, y), b2=(xb2, yb2))
            numer_val = np.sum(dists * raw_firing_rate)
            denom_val = np.sum(dists)
            smoothed_firing_rate[x, y] = numer_val / denom_val + 0.

    # debug infor
    num_outside_tracking = np.sum(is_outside_tracking * 1)
    num_in_empty_spot = np.sum(is_empty_spot * 1)
    if debug_mode:
        if num_in_empty_spot > 0:
            print('Spikes skipped due to empty spots in tracking or outside of range', num_in_empty_spot)
        if num_outside_tracking > 0:
            print('Spikes skipped due to occurring before or after tracking', num_outside_tracking)

    return raw_firing_rate, smoothed_firing_rate, binned_occ


def get_velocity_map(body_dir_angles, speeds, params, tracking_validating, cell_frame, frame_rate, is_torus,
                     smoothing_par, session_indicator=None, debug_mode=False):

    velocity_max_speed = params[0]
    n_bin_speed = params[1]
    n_bin_angle = params[2]

    n_frames = len(body_dir_angles)

    is_outside_tracking = np.logical_or(cell_frame > n_frames - 1, cell_frame < 0)
    if np.sum(is_outside_tracking) > 0:
        print('Spikes skipped due to empty spots in tracking', np.sum(is_outside_tracking))
    valid_cell_frame = cell_frame[~is_outside_tracking]

    body_dir_angles = body_dir_angles + math.pi

    if np.nanmin(body_dir_angles) < -0.0001 or np.nanmax(body_dir_angles) > 2. * math.pi + 0.0001:
        raise Exception('body_dir_angles should between -pi to pi. Please check your data.')

    speed_bins = np.linspace(0, velocity_max_speed, n_bin_speed + 1)
    angle_bins = np.linspace(0, 2. * math.pi, n_bin_angle + 1)
    binned_speeds = np.ones(n_frames).astype(int) * -10
    binned_angles = np.ones(n_frames).astype(int) * -10

    for i in np.arange(n_bin_speed):
        valid_ind = (speeds >= speed_bins[i]) * (speeds < speed_bins[i + 1])
        binned_speeds[valid_ind] = i

    for i in np.arange(n_bin_angle):
        valid_ind = (body_dir_angles >= angle_bins[i]) * (body_dir_angles < angle_bins[i + 1])
        binned_angles[valid_ind] = i

    binned_occ = np.zeros((n_bin_speed, n_bin_angle))
    for i in np.arange(n_bin_speed):
        icount = (binned_speeds == i) * tracking_validating
        for j in np.arange(n_bin_angle):
            if len(np.ravel(frame_rate)) == 1:
                binned_occ[i, j] = float(np.sum(icount * (binned_angles == j))) / float(frame_rate)  # in seconds
            else:
                total_mat = []
                for k in range(len(frame_rate)):
                    scount = (session_indicator == k)
                    part_mat = float(np.sum(icount * (binned_angles == j) * scount)) / float(frame_rate[k])
                    total_mat.append(part_mat)
                binned_occ[i, j] = np.sum(total_mat)

    binned_acc = np.zeros((n_bin_speed, n_bin_angle))
    is_empty_spot = np.logical_or(binned_angles < 0, binned_speeds < 0)
    for i in np.arange(len(valid_cell_frame)):
        ii = valid_cell_frame[i]
        if not is_empty_spot[ii]:
            binned_acc[binned_speeds[ii], binned_angles[ii]] += 1

    # convert to firing rates
    raw_firing_rate = np.zeros((n_bin_speed, n_bin_angle))
    raw_firing_rate[binned_occ > 0] = binned_acc[binned_occ > 0] / binned_occ[binned_occ > 0]
    # binnedacc[binnedocc < occupancy_thresh] = 0.

    fr_mat = raw_firing_rate.copy()
    center_x_start = 0
    center_y_start = 0

    if is_torus:
        fr_mat = np.hstack([fr_mat, fr_mat, fr_mat])
        center_y_start = n_bin_angle

    gk_size = fr_mat.shape

    smoothed_firing_rate = raw_firing_rate.copy()
    xb2 = smoothing_par[0] ** 2
    yb2 = smoothing_par[1] ** 2
    # slow but doesnt matter so much
    for i in range(n_bin_speed):
        for j in range(n_bin_angle):
            center_x = i + center_x_start
            center_y = j + center_y_start
            dists = get_gauss_kernel(size=(gk_size[0], gk_size[1]), center=(center_x, center_y), b2=(xb2, yb2))
            numer_val = np.sum(dists * fr_mat)
            denom_val = np.sum(dists)
            smoothed_firing_rate[i, j] = numer_val / denom_val + 0.

    num_outside_tracking = np.sum(is_outside_tracking * 1)
    num_in_empty_spot = np.sum(is_empty_spot)
    if debug_mode:
        if num_in_empty_spot > 0:
            print('Spikes skipped due to empty spots in tracking or outside of range', num_in_empty_spot)
        if num_outside_tracking > 0:
            print('Spikes skipped due to occurring before or after tracking', num_outside_tracking)

    return raw_firing_rate, smoothed_firing_rate, binned_occ, speed_bins


def get_shuffled_firing_rates(factor, factor_bins, binned_occ, cell_frames, periodic=False,
                              smoothing_par=1, occupancy_thresh=0.4):

    cleaned_rm = np.zeros(len(factor_bins)-1)
    cleaned_rm[:] = np.nan
    xvals, raw_rm, sm_rm = get_1d_ratemap(factor, factor_bins, binned_occ, cell_frames, periodic, smoothing_par)
    cleaned_rm[binned_occ > occupancy_thresh] = sm_rm[binned_occ > occupancy_thresh]
    return cleaned_rm, binned_occ, raw_rm



# factors = factor1d
# bins = bins1d
# frame_rate = overall_framerate
# smoothing_par = smoothing_par_1d
# occupancy_thresh = occupancy_thresh_1d
# factors, bins = factor1d, bins1d
def get_shuffled_data(factors, bins, factor_torus, binned_occ, cell_names, cell_activities,
                      session_ts, tracking_ts, frame_rate, time_bins,
                      n_bins_1d, n_shuffles, shuffle_range, use_time_bins=True,
                      smoothing_par=1, occupancy_thresh=0.4):
    """

    :param factors:
    :param bins:
    :param factor_torus:
    :param binned_occ:
    :param cell_names:
    :param cell_activities:
    :param session_ts:
    :param tracking_ts:
    :param frame_rate: overall frame rate
    :param time_bins:
    :param n_bins_1d:
    :param n_shuffles:
    :param shuffle_range:
    :param use_time_bins:
    :param smoothing_par:
    :param occupancy_thresh:
    :return:
    """

    min_shuffle_offset = shuffle_range[0]  # in seconds!!
    max_shuffle_offset = shuffle_range[1]  # in seconds!!

    n_cells = len(cell_names)

    ratemaps_data = {}
    shuffled_means = {}
    shuffled_stds = {}
    shuffled_lower_quant = {}
    shuffled_upper_quant = {}
    all_keys = list(factors.keys())
    nkeys = len(all_keys)
    for j in range(nkeys):
        da_key = all_keys[j]
        shuffled_means[da_key] = np.zeros((n_bins_1d, n_cells))
        shuffled_stds[da_key] = np.zeros((n_bins_1d, n_cells))
        shuffled_means[da_key][:] = np.nan
        shuffled_stds[da_key][:] = np.nan
        shuffled_lower_quant[da_key] = np.zeros((n_bins_1d, n_cells))
        shuffled_upper_quant[da_key] = np.zeros((n_bins_1d, n_cells))
        shuffled_lower_quant[da_key][:] = np.nan
        shuffled_upper_quant[da_key][:] = np.nan

    random_numbers = np.random.rand(n_shuffles, n_cells) * 2 - 1
    toff_mat = random_numbers * (max_shuffle_offset - min_shuffle_offset) + min_shuffle_offset

    for cellnum in np.arange(n_cells):
        c_name = cell_names[cellnum]
        c_data = cell_activities[cellnum]
        cell_validating = validating_cell_data(c_data, [session_ts[0]], [session_ts[1]])
        cell_data = c_data[cell_validating]
        occ_mat = np.zeros((nkeys, n_bins_1d, n_shuffles))
        smoothed_rms = np.zeros((nkeys, n_bins_1d, n_shuffles))
        cleaned_rms = np.zeros((nkeys, n_bins_1d, n_shuffles))
        print(('Shuffling', 100 * float(cellnum + 1) / float(n_cells), 'going ... '))
        if use_time_bins:
            for i in np.arange(n_shuffles):
                t_offset = toff_mat[i, cellnum]
                shuffled_cell_data = cell_data + t_offset
                lower_than_lb = (shuffled_cell_data < session_ts[0])
                larger_than_ub = (shuffled_cell_data > session_ts[1])
                shuffled_cell_data[lower_than_lb] = shuffled_cell_data[lower_than_lb] + session_ts[1]
                shuffled_cell_data[larger_than_ub] = shuffled_cell_data[larger_than_ub] - session_ts[1]
                cell_frames = np.ones(len(shuffled_cell_data), 'i') * -1
                for j in range(len(time_bins) - 1):
                    cell_ok = np.logical_and(shuffled_cell_data >= time_bins[j], shuffled_cell_data < time_bins[j + 1])
                    if len(cell_ok) > 0:
                        cell_frames[cell_ok] = j
                if np.any(cell_frames < 0):
                    raise ValueError('no cell frame should be smaller than 0, check the code.')
                for j in range(nkeys):
                    da_key = all_keys[j]
                    cleaned_rms[j, :, i], occ_mat[j, :, i], smoothed_rms[j, :, i] = get_shuffled_firing_rates(
                        factors[da_key], bins[da_key], binned_occ[da_key], cell_frames, factor_torus[da_key],
                        smoothing_par, occupancy_thresh)
        else:
            base_cell_frame = np.round((cell_data - tracking_ts[0]) * frame_rate)
            base_cell_frame = base_cell_frame.astype(int)
            n_frames = (tracking_ts[1] - tracking_ts[0]) * frame_rate + 1
            for i in np.arange(n_shuffles):
                t_offset = toff_mat[i, cellnum]
                n_rolling_frames = int(np.round(t_offset * frame_rate))
                cell_frames = base_cell_frame + n_rolling_frames
                cell_frames[cell_frames > n_frames - 1] = cell_frames[cell_frames > n_frames - 1] - n_frames
                cell_frames[cell_frames < 0] = cell_frames[cell_frames < 0] + n_frames
                # cell_frames = np.roll(base_cell_frame, n_rolling_frames)

                for j in range(nkeys):
                    da_key = all_keys[j]
                    cleaned_rms[j, :, i], occ_mat[j, :, i], smoothed_rms[j, :, i] = get_shuffled_firing_rates(
                        factors[da_key], bins[da_key], binned_occ[da_key], cell_frames, factor_torus[da_key],
                        smoothing_par, occupancy_thresh)

        for j in range(nkeys):
            da_key = all_keys[j]
            for k in range(n_bins_1d):
                if np.sum(~np.isnan(cleaned_rms[j, k, :])) > 0:
                    nan_mean = np.nanmean(cleaned_rms[j, k, :])
                    nan_std = np.nanstd(cleaned_rms[j, k, :])
                    shuffled_means[da_key][k, cellnum] = nan_mean
                    shuffled_stds[da_key][k, cellnum] = nan_std
                    nan_lq = np.nanquantile(cleaned_rms[j, k, :], 0.025)
                    nan_uq = np.nanquantile(cleaned_rms[j, k, :], 0.975)
                    shuffled_lower_quant[da_key][k, cellnum] = nan_lq
                    shuffled_upper_quant[da_key][k, cellnum] = nan_uq

            ratemaps_data['%s-%s-acc_shuffles' % (c_name, da_key[2:])] = smoothed_rms[j, :, :]
            ratemaps_data['%s-%s-occ' % (c_name, da_key[2:])] = binned_occ[da_key]
    return ratemaps_data, shuffled_means, shuffled_stds, shuffled_lower_quant, shuffled_upper_quant



def calculate_skinfo_rate(data, occ):
    """
    Calculate Skaggs and McNaughton mutual information rate
    :param data: rate maps data
    :param occ: occupancy of factor
    :return: information rate of the cell in bits per spike,
            indicates how much knowing a variable X reduces the uncertainty of a variable Y.
    """
    occ_vec = np.ravel(occ)
    data_vec = np.ravel(data)
    n_data = len(occ_vec)
    overall_mean_fr = 0.
    p_occ = np.zeros(n_data)
    for occi in range(n_data):
        if np.sum(occ_vec) > 0.000001:
            p_occ[occi] = occ_vec[occi] / np.sum(occ_vec) + 0.
            overall_mean_fr += p_occ[occi] * data_vec[occi]
    info_rate = 0
    for occi in range(n_data):
        if data_vec[occi] > 0.000001:
            info_rate += p_occ[occi] * data_vec[occi] / overall_mean_fr * math.log2(data_vec[occi] / overall_mean_fr)
    return info_rate


def get_default_2d_tasks(include_derivatives=True, debug_mode=False):
    stuff2d = {}
    stuff2d['A Ego2_H_pitch_and_Ego3_H_pitch'] = ['Q Ego2_head_pitch', 'L Ego3_Head_pitch']
    stuff2d['B Ego2_H_roll_and_Ego3_H_roll'] = ['P Ego2_head_roll', 'K Ego3_Head_roll']
    stuff2d['C Ego2_H_azimuth_and_Ego3_H_azimuth'] = ['R Ego2_head_azimuth', 'M Ego3_Head_azimuth']

    stuff2d['D Ego2_H_pitch_and_allo_HD'] = ['Q Ego2_head_pitch', 'D Allo_head_direction']
    stuff2d['E Ego2_H_pitch_and_neck_elevation'] = ['Q Ego2_head_pitch', 'G Neck_elevation']

    stuff2d['E Ego2_H_pitch_and_Ego2_H_azimuth'] = ['Q Ego2_head_pitch', 'R Ego2_head_azimuth']
    stuff2d['E Ego2_H_pitch_and_Ego2_H_roll'] = ['Q Ego2_head_pitch', 'P Ego2_head_roll']
    stuff2d['F Ego2_H_azimuth_and_Ego2_H_roll'] = ['R Ego2_head_azimuth', 'P Ego2_head_roll']

    stuff2d['G Ego3_H_pitch_and_neck_elevation'] = ['L Ego3_Head_pitch', 'G Neck_elevation']
    stuff2d['G Ego3_H_pitch_and_Ego3_H_roll'] = ['L Ego3_Head_pitch', 'K Ego3_Head_roll']
    stuff2d['G Ego3_H_pitch_and_Ego3_H_azimuth'] = ['L Ego3_Head_pitch', 'M Ego3_Head_azimuth']
    stuff2d['H Ego3_H_azimuth_and_Ego3_H_roll'] = ['M Ego3_Head_azimuth', 'K Ego3_Head_roll']

    stuff2d['J Ego3_H_roll_and_B_azimuth'] = ['K Ego3_Head_roll', 'O Back_azimuth']

    stuff2d['P Speeds_and_Ego2_H_pitch'] = ['B Speeds', 'Q Ego2_head_pitch']
    stuff2d['Q Speeds_and_Ego3_H_pitch'] = ['B Speeds', 'L Ego3_Head_pitch']
    stuff2d['R Speeds_and_Ego3_H_azimuth'] = ['B Speeds', 'M Ego3_Head_azimuth']
    stuff2d['S Speeds_and_Ego3_H_roll'] = ['B Speeds', 'K Ego3_Head_roll']

    stuff2d['U B_pitch_and_B_direction'] = ['N Back_pitch', 'C Body_direction']
    stuff2d['V B_pitch_and_Ego2_H_pitch'] = ['N Back_pitch', 'Q Ego2_head_pitch']
    stuff2d['W B_pitch_and_Ego3_H_pitch'] = ['N Back_pitch', 'L Ego3_Head_pitch']
    stuff2d['X B_pitch_and_Ego3_H_azimuth'] = ['N Back_pitch', 'M Ego3_Head_azimuth']
    stuff2d['Y B_pitch_and_Ego3_H_roll'] = ['N Back_pitch', 'K Ego3_Head_roll']
    stuff2d['Z B_pitch_and_B_azimuth'] = ['N Back_pitch', 'O Back_azimuth']
    # allda2Dstuff['R Allo_H_roll_and_neck_elevation'] = ['F Allo_head_roll', 'G Neck_elevation']
    # allda2Dstuff['S Allo_H_roll_and_H_roll'] = ['F Allo_head_roll', 'K Head_roll']

    if include_derivatives:
        stuff2d['L Ego3_H_pitch_and_Allo_HD_1st_der'] = ['L Ego3_Head_pitch', 'D Allo_head_direction_1st_der']
        stuff2d['M Speeds_and_Allo_HD_1st_der'] = ['B Speeds', 'D Allo_head_direction_1st_der']
        stuff2d['N Speeds_and_BD_1st_der'] = ['B Speeds', 'C Body_direction_1st_der']
        stuff2d['T B_pitch_and_Allo_HD_1st_der'] = ['N Back_pitch', 'D Allo_head_direction_1st_der']

    if debug_mode:
        for dakey in stuff2d.keys():
            print(dakey)

    return stuff2d


def add_2d_tasks(tasks2d, insert_task_dict):
    insert_keys = list(insert_task_dict.keys())
    for da_key in insert_keys:
        tasks2d[da_key] = insert_task_dict[da_key]
    return tasks2d


