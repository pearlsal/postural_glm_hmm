import pickle
import os
import numpy as np
import datetime
import ssm
import pandas as pd
from functools import reduce
import itertools
from ratemaps import *
from spikestats.process_data import *
from spikestats.toolkits import *

# loading data and pre-inference processing#####

"""
path_raw_data == folder containing the "figshare" pickle files
path_output == folder where you have the files and where you want the additional files to be saved (!PLOT EXCEPTION!)
pickle_file == the above mentioned 
"""


def folder_structure():
    """
    Function to create the folder structure for the module
    """

    data_folder = f"data"
    if not os.path.exists("./" + data_folder):
        os.makedirs("./" + data_folder)
    path_data_folder = os.path.dirname(os.path.abspath("./" + data_folder)) + "/data/"

    raw_data_folder = f"raw_data"
    if not os.path.exists(path_data_folder + raw_data_folder):
        os.makedirs(path_data_folder + raw_data_folder)
    path_raw_data_folder = os.path.dirname(os.path.abspath(path_data_folder + raw_data_folder)) + "/raw_data/"

    process_data_folder = f"process_data"
    if not os.path.exists(path_data_folder + process_data_folder):
        os.makedirs(path_data_folder + process_data_folder)
    path_process_data_folder = os.path.dirname(os.path.abspath(path_data_folder + process_data_folder)) + \
                               "/process_data/"

    top_folder = f"results_folder"
    if not os.path.exists("./" + top_folder):
        os.makedirs("./" + top_folder)
    path_top_folder = os.path.dirname(os.path.abspath("./" + top_folder)) + "/results_folder/"

    info_dir = f"info_dir/"
    if not os.path.exists(path_top_folder + info_dir):
        os.makedirs(path_top_folder + info_dir)
    path_info_dir = os.path.dirname(os.path.abspath(path_top_folder + info_dir)) + "/info_dir/"

    analysis_dir = f"analysis_dir/"
    if not os.path.exists(path_top_folder + analysis_dir):
        os.makedirs(path_top_folder + analysis_dir)
    path_analysis_dir = os.path.dirname(os.path.abspath(path_top_folder + analysis_dir)) + "/analysis_dir/"

    plots_dir = f"plots_dir/"
    if not os.path.exists(path_top_folder + plots_dir):
        os.makedirs(path_top_folder + plots_dir)
    path_plots_dir = os.path.dirname(os.path.abspath(path_top_folder + plots_dir)) + "/plots_dir/"

    single_pred_dir = f"single_pred_dir/"
    if not os.path.exists(path_plots_dir + single_pred_dir):
        os.makedirs(path_plots_dir + single_pred_dir)
    path_single_pred_dir = os.path.dirname(os.path.abspath(path_plots_dir + single_pred_dir)) + "/single_pred_dir/"

    multi_pred_dir = f"multi_pred_dir/"
    if not os.path.exists(path_plots_dir + multi_pred_dir):
        os.makedirs(path_plots_dir + multi_pred_dir)
    path_multi_pred_dir = os.path.dirname(os.path.abspath(path_plots_dir + multi_pred_dir)) + "/multi_pred_dir/"

    dict_paths = {}
    dict_paths['path_data_folder'] = path_data_folder
    dict_paths['path_raw_data_folder'] = path_raw_data_folder
    dict_paths['path_process_data_folder'] = path_process_data_folder
    dict_paths['path_top_folder'] = path_top_folder
    dict_paths['path_info_dir'] = path_info_dir
    dict_paths['path_analysis_dir'] = path_analysis_dir
    dict_paths['path_plots_dir'] = path_plots_dir
    dict_paths['path_single_pred_dir'] = path_single_pred_dir
    dict_paths['path_multi_pred_dir'] = path_multi_pred_dir

    name_file = "dict_paths"
    a_file = open(path_info_dir + name_file, 'wb')
    pickle.dump(dict_paths, a_file)
    a_file.close()

    return path_data_folder, path_raw_data_folder, path_process_data_folder, path_top_folder, path_info_dir, \
           path_analysis_dir, path_plots_dir, path_single_pred_dir, path_multi_pred_dir


def processing_and_loading(path_raw_data_folder, path_process_data_folder, mat_file=None,
                           data_continous_ratemaps_pickle=None, data_binned_glm_pickle=None):
    """
    mat_file : original file with data from experiment (preprocessing already done)
    data_continous_ratemaps : it contains continous predictors with their names and cells' ID. Coming from "get_rm_pre_data()"
    data_binned_glm : discretized and sorted spikes in a matricial form. Coming from "prepare_data4glms()"
    """
    # TODO: assert mat_file or (data_continous_ratemaps_pickle and data_binned_glm_pickle)
    if mat_file is not None:
        # TODO: ask Ida about boundaries and a basic-general set of values
        boundaries = {'neck_elevation': (0., 0.36),
                      'back_ang': ((-60, 60), (-60, 60)),
                      'opt_back_ang': ((-60, 60), (-60, 60)),
                      'speeds': ((0, 120), (0, 120), (0, 120), (0, 120)),
                      'selfmotion': (
                          (-104.02846857, 83.66376106),  # TODO: ask here and all boundaries, general "safe" values
                          (-10.54837855, 175.69859119),
                          (-119.35408181, 96.78457956),
                          (-30.42151711, 161.00896119),
                          (-104.41416926, 101.5806638),
                          (-6.96034067, 403.66958929),
                          (-115.14593404, 93.70091392),
                          (-30.26921306, 242.64589367)),
                      'speeds_1st_der': ((-150, 150), (-150, 150), (-150, 150), (-150, 150)),
                      'neck_1st_der': (-0.1, 0.1),
                      'neck_2nd_der': (-0.8, 0.8),
                      'allo_head_1st_der': ((-400, 400), (-300, 300), (-400, 400)),
                      'allo_head_2nd_der': ((-4000, 4000), (-3000, 3000), (-4000, 4000)),
                      'bodydir_1st_der': (-300, 300),
                      'bodydir_2nd_der': (-1000, 1000),
                      'ego3_head_1st_der': ((-400, 400), (-300, 300), (-400, 400)),
                      'ego3_head_2nd_der': ((-4000, 4000), (-3000, 3000), (-4000, 4000)),
                      'ego2_head_1st_der': ((-400, 400), (-300, 300), (-400, 400)),
                      'ego2_head_2nd_der': ((-4000, 4000), (-3000, 3000), (-4000, 4000)),
                      'back_1st_der': ((-100, 100), (-100, 100)),
                      'back_2nd_der': ((-1000, 1000), (-1000, 1000)),
                      'opt_back_1st_der': ((-100, 100), (-100, 100)),
                      'opt_back_2nd_der': ((-1000, 1000), (-1000, 1000)),
                      'imu_allo_head_1st_der': ((-400, 400), (-300, 300), (-400, 400)),
                      'imu_allo_head_2nd_der': ((-4000, 4000), (-3000, 3000), (-4000, 4000)),
                      'imu_ego3_head_1st_der': ((-400, 400), (-300, 300), (-400, 400)),
                      'imu_ego3_head_2nd_der': ((-4000, 4000), (-3000, 3000), (-4000, 4000)),
                      'imu_ego2_head_1st_der': ((-400, 400), (-300, 300), (-400, 400)),
                      'imu_ego2_head_2nd_der': ((-4000, 4000), (-3000, 3000), (-4000, 4000))}

        mat = data_loader(path_raw_data_folder + mat_file)

        pre_data = data_generator(mat)

        # TODO: below as well, ask Ida about safe-general parameters

        data, name_file = get_rm_pre_data(pre_data, use_even_odd_minutes=True, speed_type='jump', window_size=250,
                                          include_factor=None, derivatives_param=(10, 10), boundary=boundaries,
                                          avoid_2nd=False,
                                          filter_by_speed=None, filter_by_spatial=None, filter_by_factor=None,
                                          num_bins_1d=36, occupancy_thresh_1d=0.4, save_data=True)

        a_file = open(path_process_data_folder + name_file, 'wb')
        pickle.dump(data, a_file)
        a_file.close()

        dataglm = prepare_data4glms(data)

        a_file = open(path_process_data_folder + 'binned_4_GLM_' + name_file, 'wb')
        pickle.dump(dataglm, a_file)
        a_file.close()

        data_continous_ratemaps = data
        data_binned_glm = dataglm
        print(f"from mat the glm keys are {data_binned_glm.keys()}")

    if (data_continous_ratemaps_pickle and data_binned_glm_pickle) is not None:
        with open(path_process_data_folder + data_continous_ratemaps_pickle, 'rb') as handle:
            data_continous_ratemaps = pickle.load(handle)
        with open(path_process_data_folder + data_binned_glm_pickle, 'rb') as handle:
            data_binned_glm = pickle.load(handle)

        print(f"the continous dictionary is {data_continous_ratemaps.keys()}")
        print(f"the binned dictionary is {data_binned_glm.keys()}")
        print(f"the binned dictionary is {data_binned_glm['spk_mat'][0]}")

    return data_continous_ratemaps, data_binned_glm


# -------------------------------------------------------------------------------------------------------------------- #

def get_data_information(path_info_dir, data_continous_ratemaps):
    """
    This function extract useful information of data and save them in a pickle
    """
    predictors_name_list = list(data_continous_ratemaps['possiblecovariates'].keys())
    cells_id = data_continous_ratemaps['cell_names']
    prefix_name = data_continous_ratemaps['output_file_prefix']
    splitted_name = prefix_name.split("_")
    animal_name = splitted_name[0] + "_" + splitted_name[2]

    dict_info = {}
    dict_info['predictors_name_list'] = predictors_name_list
    dict_info['cells_id'] = cells_id
    dict_info['animal_name'] = animal_name

    data_file_name = 'dictionary_information.pkl'
    with open(path_info_dir + data_file_name, "wb") as handle:
        pickle.dump(dict_info, handle)

    return predictors_name_list, cells_id, animal_name


# -------------------------------------------------------------------------------------------------------------------- #

def dict_parameters_hmm(path_info_dir, animal_name, num_dimen, num_categ_obs=2, N_iters=100, tolerance=10 ^ -5,
                        num_indep_neurons=2, num_predictors=3, max_num_states=3, observation_type="input_driven_obs",
                        transistion_type="inputdriven", optim_method="em", threshold_diff_pred=None):
    """
    This function includes all the parameters for the inference.
    The user has to insert manually the allowed value for the specific type of inference.
    The constraint description is in the file 'additional_details.txt' in the package folder
    """
    # TODO: put assert to check that the number of neurons is between 1 and N(tot number of neurons from length of cell_names)
    # TODO: put assert for the rest of the parameters
    dict_param = {}

    dict_param['num_dimen'] = num_dimen
    dict_param['num_categ_obs'] = num_categ_obs
    dict_param['N_iters'] = N_iters
    dict_param['tolerance'] = tolerance
    dict_param['num_indep_neurons'] = num_indep_neurons
    dict_param['num_predictors'] = num_predictors
    dict_param['max_num_states'] = max_num_states
    dict_param['num_states'] = max_num_states - 1
    dict_param['list_states'] = [a for a in range(2, max_num_states + 1)]  # !check the +1!##
    dict_param['observation_type'] = observation_type
    dict_param['transistion_type'] = transistion_type
    dict_param['optim_method'] = optim_method
    dict_param['animal_name'] = animal_name
    dict_param['threshold_diff_pred'] = threshold_diff_pred  # TODO: assert 0<thresh<1

    data_file_name = 'dictionary_parameters.pkl'
    with open(path_info_dir + data_file_name, "wb") as handle:
        pickle.dump(dict_param, handle)

    return dict_param


# -------------------------------------------------------------------------------------------------------------------- #


# TODO: use assert  to check if all the spike trains have the same length so that I can take ['spk_mat'][0] and..
def cells_selection_random(path_info_dir, data_continous_ratemaps, data_binned_glm, dict_param):
    """
    This function selects the number of neurons you insert in the dictionary for parameters and obtain all
    the necessary information and save them
    """
    cell_names_list = np.asarray(data_continous_ratemaps['cell_names'])
    # assert (num_indip_neu<len(cell_names_list))
    cells_to_infer = np.random.choice(cell_names_list, size=dict_param['num_indep_neurons'])
    cells_index_rnd = []
    for i in range(len(cells_to_infer)):
        cells_index_rnd.append(np.where(cell_names_list == cells_to_infer[i])[0][0])
    cells_index = np.sort(np.asarray(cells_index_rnd))
    tot_time = len(data_binned_glm['spk_mat'][0])

    text_content = f"ID cells used in this inference is {cells_to_infer} and the corresponding indices are {cells_index_rnd}"

    with open(path_info_dir + "description_parameters.txt", "w") as file:
        file.write(text_content)

    return cells_index, tot_time


# -------------------------------------------------------------------------------------------------------------------- #

def cells_selection_manual(path_info_dir, data_continous_ratemaps, data_binned_glm, dict_param, csv_file_cells):
    """
    Write the full name of the file including .csv
    CSV file structure: only one column with cells' names on the rows (e.g. 'imec0_cl0000_ch000'). Remember quotes or no?
    The default location is in the path_info_dir, so that the path is automatized. Remember to insert there
    ?or is it better to leave in the main folder and automatically move to the info_dir?
    """

    cell_names_array = np.array(data_continous_ratemaps['cell_names'])
    data_read = pd.read_csv(csv_file_cells)
    cells_index = []
    for i in range(dict_param['num_indep_neurons']):
        cells_index.append(np.where(cell_names_array == data_read.iloc[i, 0])[0][0])
        # TODO: if error on the component [0], means the cells are not matching (one reason for the error). Use assert

    tot_time = len(data_binned_glm['spk_mat'][0])

    text_content = f"ID cells selected for this inference is {data_read} and the corresponding indices are {cells_index}"
    with open(path_info_dir + "cells_list_and_indices.txt", "w") as file:
        file.write(text_content)

    return cells_index, tot_time


# -------------------------------------------------------------------------------------------------------------------- #


# TODO: capital print to warn about the deleted time points of the mask and
def data_structure(path_info_dir, path_analysis_dir, path_single_pred_dir, data_continous_ratemaps, data_binned_glm, dict_param,
                   tot_time, cell_index, predictors_name_list, predictor_file=None):
    """
    Main function to process the data (neurons' and predictors' time series)
    The "predictor_file" has to be in the
    """
    if predictor_file is not None:
        predictors_df = pd.read_csv(path_info_dir + predictor_file)
        print(f"the iloc approach is {predictors_df.iloc[:,0]}")
        print(f"the length of df is {len(predictors_df)}")
        predictors_name_list = [predictors_df.iloc[x,0].strip("'") for x in range(len(predictors_df))]
        print(f"test list comprehension is {predictors_name_list}")
        #predictors_name_list = predictors_df.astype(str).values.tolist()
        #print(f"test string is {predictors_name_list}")
        #predictors_name_list = list(itertools.chain.from_iterable(predictors_df.values.tolist()))
        #predictors_name_list = map(str.strip('"'), predictors_name_list)
        print(predictors_name_list)



    glmhmms_ista = []  # save the istances for the hmm class
    T_list = []  # masked total time
    tot_masked_indices_list = []  # array of the masked indices
    path_plots_list = []  # to save single predictor inference plots
    inputs_list = []  # post-processed predictors list

    date = datetime.date.today()

    name_upper_folder = f"{dict_param['animal_name']}_singlepredictor_{date}_run/"
    if not os.path.exists(path_single_pred_dir + name_upper_folder):
        os.makedirs(path_single_pred_dir + name_upper_folder)
    plots_folder = path_single_pred_dir + name_upper_folder
    print(f"NAME OF THE FOLDER IS {plots_folder}")
    print(f"IMPORTANT: given the presence of nans (missing points of the camera) part of the data are deleted." +
          f"If the number of missing points is 'small enough', it should not interfere with the inference."
          f"Below the ratio of missing points")

    for k in range(dict_param['num_predictors']):
        # take only the common (shared, redundant) indices
        mask_raw = np.ma.masked_where(
            np.isnan(data_continous_ratemaps['possiblecovariates'][f'{predictors_name_list[k]}']) == True,
            data_continous_ratemaps['possiblecovariates'][f'{predictors_name_list[k]}'],
            copy=True)
        tot_mask_indices = np.where(np.isfinite(mask_raw))[
            0]  # find the index for the finite elements(not nan)
        tot_masked_indices_list.append(tot_mask_indices)
        T_list.append(len(tot_mask_indices))

        # assign the processed predictors containing only the common indices and adding the bias term initialized to 1
        # (!important, the bias term shouldn't be initialized to zero)
        inpts = np.ones((dict_param['num_indep_neurons'], T_list[k], 2),
                        dtype=np.float64)
        inpts[:, :, 0] = data_continous_ratemaps['possiblecovariates'][f"{predictors_name_list[k]}"][tot_mask_indices]
        inpts = list(inpts)  # convert inpts to correct format   #in partiular if you have different length of the time
        inputs_list.append(inpts)

    # TODO: look for redundancy and modify it accordingly
    # #!create the structure for multiple neurons, thus a nested list for each first loop!##
    process_neur = []  # requested list structure for multiple neurons with sessions
    for i in range(len(dict_param['list_states'])):
        for j in range(dict_param['num_indep_neurons']):
            for k in range(dict_param['num_predictors']):
                selected_neur_mat = data_binned_glm['spk_mat'][
                    cell_index[j]]
                # binarization (possible presence of 2 spikes each bin)
                selected_neur_mat = np.where(selected_neur_mat == 0, selected_neur_mat, 1)
                # selecting the first neuron and taking only the shared indices to use the mask
                reduced_matrix = selected_neur_mat[tot_masked_indices_list[k]].astype(int)
                process_neur.append(reduced_matrix.reshape((T_list[k], 1)))  # reshape in the correct format
                # checking how many points are lost due to nans  in the predictors (no nans in the neural response)
                miss_points_ratio = (tot_time - T_list[k]) / tot_time
                # create a folder for the particular case (model, neuron and predictor) with .txt description###
                print(f"!Fraction missing points is {miss_points_ratio} for the model {i} neuron {j} and predictor {k}")
                name_folder = f"{dict_param['animal_name']}_states={dict_param['list_states'][i]}_numsess={dict_param['num_indep_neurons']}" \
                              + f"_max_iters={dict_param['N_iters']}_tolerance={dict_param['tolerance']}" + \
                              f"_pred={1}" + \
                              f"_obs={dict_param['observation_type']}_trans={dict_param['transistion_type']}/"
                path_current_pred_dir = plots_folder + "/" + name_folder
                if not os.path.exists(path_current_pred_dir):
                    os.makedirs(path_current_pred_dir)

                text_content = f"Single predictor and neuron inference. Predictor selection." "\n" \
                               f"{dict_param['animal_name']}_states={dict_param['list_states'][i]}" \
                               f"_numsess={dict_param['num_indep_neurons']}_max_iters={dict_param['N_iters']}" \
                               f"_tolerance={dict_param['tolerance']}_numpredict={1}" \
                               f"_tot_pred={dict_param['num_predictors']}_distal" "\n" \
                               f"obs={dict_param['observation_type']}_trans={dict_param['transistion_type']}" \
                               f"_method={dict_param['optim_method']}" "\n" \
                               f"fraction of missing points={miss_points_ratio}"

                with open(path_current_pred_dir + f"description_parameters_{predictors_name_list[k]}.txt", "w") as file:
                    file.write(text_content)
                # TODO: do a dictionary instead of list?
                path_plots_list.append(path_current_pred_dir)

                # create the istances for inference
                ista_glmhmm = ssm.HMM(dict_param['list_states'][i], dict_param['num_dimen'], 2,
                                      observations=dict_param['observation_type'],
                                      observation_kwargs=dict(C=dict_param['num_categ_obs']),
                                      transitions=dict_param['transistion_type'])
                glmhmms_ista.append(ista_glmhmm)

    # dictionary to store all the objects and istances to do the inference off module
    dict_objects = {}
    dict_objects["glmhmms_ista"] = glmhmms_ista
    dict_objects["process_neur"] = process_neur
    dict_objects["inputs_list"] = inputs_list
    dict_objects["T_list"] = T_list
    dict_objects["tot_masked_indices_list"] = tot_masked_indices_list
    dict_objects["path_plots_list"] = path_plots_list

    data_file_name = 'dict_objects.pkl'
    a_file = open(path_analysis_dir + data_file_name, "wb")
    pickle.dump(dict_objects, a_file)
    a_file.close()

    return glmhmms_ista, process_neur, inputs_list, T_list, tot_masked_indices_list, path_plots_list, plots_folder


# -------------------------------------------------------------------------------------------------------------------- #

###multiple predictors structure for inference###

# input all_predictors_list = list_from_other_function
# import a csv file with the same structure of the cells
# ? TODO: compare the strings to be sure they are correct?

def data_structure_multicov(path_analysis_dir, path_info_dir, path_multi_pred_dir, data_continous_ratemaps,
                            data_binned_glm, dict_param, tot_time, cell_index, dict_objects=None,
                            file_predictors=None):
    """

    """
    date = datetime.date.today()
    path_plots_list = []
    tot_masked_indices = []
    best_predictors_name_df = pd.read_csv(path_info_dir + file_predictors)
    print(best_predictors_name_df)

    masked_values = []
    for i in range(dict_param['num_predictors']):
        test_mask = np.where(
            np.isnan(data_continous_ratemaps['possiblecovariates'][f"{best_predictors_name_df.iloc[i, 0]}"]) == True, 0,
            1)
        print(len(np.nonzero(test_mask)[0]))
        index_1_mask = np.where(test_mask == 1)[0]
        masked_values.append(list(index_1_mask))

    # TODO: remove the first if and put an assert to suggest the other function if 1 predictor
    if dict_param['num_predictors'] == 1:
        tot_mask_indices = masked_values[0]
    elif dict_param['num_predictors'] == 2:
        tot_mask_indices = np.intersect1d(masked_values[0], masked_values[1])
    else:
        tup_list_indices = [a for a in masked_values]
        tot_mask_indices = reduce(np.intersect1d, (tup_list_indices))
    T = len(tot_mask_indices)
    inpts = np.ones((dict_param['num_indep_neurons'], T, dict_param['num_predictors']), dtype=np.float64)
    for i in range(dict_param['num_predictors'] - 1):  # beacuse the last one should not be overwritten (bias=1)
        inpts[:, :, i] = data_continous_ratemaps['possiblecovariates'][f"{best_predictors_name_df.iloc[i, 0]}"][
            tot_mask_indices]
    inpts = list(inpts)  # correct format #in partiular if you have different length of time for spike trains

    name_upper_folder = f"{dict_param['animal_name']}_multipredictor_{date}_run/"
    if not os.path.exists(path_multi_pred_dir + name_upper_folder):
        os.makedirs(path_multi_pred_dir + name_upper_folder)
    plots_folder = path_multi_pred_dir + name_upper_folder

    # -------------------------------------------------------------------------------------------------------------------- #
    process_neur = []  # requested list structure for multiple neurons with sessions
    for i in range(len(dict_param['list_states'])):
        for j in range(dict_param['num_indep_neurons']):

            selected_neur_mat = data_binned_glm['spk_mat'][cell_index[j]]
            # binarization (possible presence of 2 spikes each bin)
            selected_neur_mat = np.where(selected_neur_mat == 0, selected_neur_mat, 1)
            # selecting the first neuron and taking only the shared indices to use the mask
            reduced_matrix = selected_neur_mat[tot_mask_indices].astype(int)
            process_neur.append(reduced_matrix.reshape((T, 1)))  # reshape in the correct format
            # checking how many points are lost due to nans  in the predictors (no nans in the neural response)
            miss_points_ratio = (tot_time - T) / tot_time
            # create a folder for the particular case (model, neuron and predictor) with .txt description###
            print(f"!Fraction missing points is {miss_points_ratio} for the model {i} neuron {j}")

            name_folder = f"multicov_{dict_param['animal_name']}_states={dict_param['list_states'][i]}_neur={data_continous_ratemaps['cell_names'][[cell_index[j]]]}" \
                          + f"_max_iters={dict_param['N_iters']}_tol={dict_param['tolerance']}" + \
                          f"_obs={dict_param['observation_type']}_trans={dict_param['transistion_type']}/"
            path_current_inference = plots_folder + name_folder
            if not os.path.exists(path_current_inference):
                os.makedirs(path_current_inference)

            text_content = f"Multiple predictors and neurons inference." "\n" \
                           f"{dict_param['animal_name']}_states={dict_param['list_states'][i]}" \
                           f"_neur={data_continous_ratemaps['cell_names'][[cell_index[j]]]}_max_iters={dict_param['N_iters']}" \
                           f"_tolerance={dict_param['tolerance']}" \
                           f"_tot_pred={dict_param['num_predictors']}_distal" "\n" \
                           f"obs={dict_param['observation_type']}_trans={dict_param['transistion_type']}" \
                           f"_method={dict_param['optim_method']}" "\n" \
                           f"fraction of missing points={miss_points_ratio}"
            with open(path_current_inference + "description_parameters_multipredictor.txt", "w") as file:
                file.write(text_content)
            path_plots_list.append(path_current_inference)
            # create the istances for inference
            ista_glmhmm = ssm.HMM(dict_param['list_states'][i], dict_param['num_dimen'], 2,
                                  observations=dict_param['observation_type'],
                                  observation_kwargs=dict(C=dict_param['num_categ_obs']),
                                  transitions=dict_param['transistion_type'])
            glmhmms_ista.append(ista_glmhmm)

    dict_processed_objects_multicov = {}
    dict_processed_objects_multicov["glmhmms_ista"] = glmhmms_ista
    dict_processed_objects_multicov["process_neur"] = process_neur
    dict_processed_objects_multicov["inputs_list"] = inpts
    dict_processed_objects_multicov["T_list"] = T
    dict_processed_objects_multicov["tot_masked_indices_list"] = tot_masked_indices
    dict_processed_objects_multicov["path_plots_list"] = path_plots_list
    data_file_name = 'dict_processed_objects_multicov.pkl'
    a_file = open(path_analysis_dir + data_file_name, "wb")
    pickle.dump(dict_processed_objects_multicov, a_file)
    a_file.close()

    return glmhmms_ista, process_neur, inpts, T, tot_masked_indices, path_plots_list, dict_processed_objects_multicov
