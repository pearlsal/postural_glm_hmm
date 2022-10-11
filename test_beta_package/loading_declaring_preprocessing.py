import pickle
import os
import csv
import numpy as np
import datetime
import ssm
import pandas as pd

# loading data and pre-inference processing#####

# TODO: use get... to download the files and ask Jingyi best way to handle her packge (like ssm, install and use?)
"""
path_raw_data == folder containing the "figshare" pickle files
path_output == folder where you have the files and where you want the additional files to be saved (!PLOT EXCEPTION!)
pickle_file == the above mentioned 
"""

# TODO: use Jingyi's pipeline and save the file with string rule (e.g. ...continous_predictors.pkl and
#  ...discretized_spikes.pkl)
"""
def source_data_preprocessing():    ##Jingyi pipeline statrting from figshare

path_data == folder containing pickle files preprocessed by Jingyi pipeline
file_ratemaps == it ends with ""
file_glm == it ends with ""

# return save the output. return path_output as well (so that it is saved in the memory)
"""


def load_data(file_ratemaps, file_glm, path_data):  # after Jingyi pipeline procesing
    """
    data_continous_ratemaps : it contains continous predictors with their names and cells' ID
    data_binned_glm : discretized and sorted spikes in a matricial form
    """
    with open(path_data + file_ratemaps, 'rb') as handle:
        data_continous_ratemaps = pickle.load(handle)

    with open(path_data + file_glm, 'rb') as handle:
        data_binned_glm = pickle.load(handle)

    # structure folders to include all outputs#
    top_folder = f"results_folder"  # _predict={predictors_name_list[j]}
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

    splitted_name = data_continous_ratemaps['output_file_prefix'].split("_")
    animal_name = splitted_name[0] + "_" + splitted_name[2] + "_" + splitted_name[3]

    # TODO: procesed_data folder to store the various dictionary with objects and quantities

    # TODO: create the path to data for actual data and csv and all useful files pre running
    # TODO: or even separate the folder function so that it creates the structure before

    return data_continous_ratemaps, data_binned_glm, path_top_folder, path_info_dir, path_analysis_dir, animal_name

# TODO: save txt-csv with all information
def get_data_information(data_continous_ratemaps, path_info_dir, data_binned_glm):
    """
    This function extract useful information of data and save them in a pickle
    """
    predictors_name_list = list(data_continous_ratemaps['possiblecovariates'].keys())

    # cells_id = data_continous_ratemaps['cell_names']

    return predictors_name_list  # , list(cells_id)


"""
	#with open('information_predictors_neurons.csv','w') as f:
	#w = csv.writer(f)
	#w.writerows(data_continous_ratemaps['possiblecovariates'])
"""

"""
!!! make interactive, printing the options for observation, transition etc....!!!
num_dimen == number of dimension (!!! in the input_driven and recurrent is possible only num_dimens=1. In the Bernoulli 
observation must be equal to the number of neuron in the population under study!!!)

num_indep_neurons == number of independent neurons in case the option selected for the observation (transistion too?) 

"""


def dict_parameters_hmm(path_info_dir,  animal_name, num_dimen, num_categ_obs, N_iters, tolerance, num_indep_neurons, num_predicotrs,
                        max_num_states, observation_type, transistion_type, optim_method):
    """
    This function includes all the parameters for the inference.
    The user has to insert manually the allowed value for the specific type of inference.
    The constraint description is in the file 'inference_constraint_explanation' in the package folder
    """
    dict_param = {}

    dict_param['num_dimen'] = num_dimen
    dict_param['num_categ_obs'] = num_categ_obs
    dict_param['N_iters'] = N_iters
    dict_param['tolerance'] = tolerance
    dict_param['num_indep_neurons'] = num_indep_neurons
    dict_param['num_predicotrs'] = num_predicotrs
    dict_param['max_num_states'] = max_num_states
    dict_param['num_states'] = max_num_states - 1
    dict_param['list_states'] = [a for a in range(2, max_num_states + 1)]  # !check the +1!##
    dict_param['observation_type'] = observation_type
    dict_param['transistion_type'] = transistion_type
    dict_param['optim_method'] = optim_method
    dict_param['animal_name'] = animal_name

    data_file_name = 'dictionary_parameters.pkl'
    with open(path_info_dir + data_file_name, "wb") as handle:
        pickle.dump(dict_param, handle)

    return dict_param

# TODO: use assert  to check if all the spike trains have the same length so that I can take ['spk_mat'][0] and..
def cells_selection_random(data_continous_ratemaps, data_binned_glm, dict_param, path_info_dir):
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


# TODO: discuss with Jonathan and Ben and create an automatic selection (N neurons chosen randomly or....?)
# TODO: combine with the automatic selection using a if condition on a variable (like extract_all) to decide weather manaual or automatic


def cells_selection_manual(data_continous_ratemaps, data_binned_glm, csv_file_cells, dict_param, path_info_dir):  # num_neu
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
        cells_index.append(np.where(cell_names_array == data_read.iloc[i,0])[0][0])
        # TODO: if error on the component [0], means the cells are not matching (one reason for the error). Use assert

    tot_time = len(data_binned_glm['spk_mat'][0])

    text_content = f"ID cells selected for this inference is {data_read} and the corresponding indices are {cells_index}"
    with open(path_info_dir + "cells_list_and_indices.txt", "w") as file:
        file.write(text_content)

    return cells_index, tot_time


# TODO: capital print to warn about the deleted time points of the mask and
def data_structure(dict_param, data_continous_ratemaps, data_binned_glm, path_top_folder,
                   path_analysis_dir,  predictors_name_list, tot_time, cell_index):
    """
    Main function to process the data (neurons' and predictors' time series)
    """

    glmhmms_ista = []  # save the istances for the hmm class
    T_list = []  # masked total time
    tot_masked_indices_list = []  # array of the masked indices
    path_plots_list = []  # to save single predictor inference plots
    inputs_list = []  # post-processed predictors list

    date = datetime.date.today()

    # TODO: automatize the name (kav_s2_interm) from the file!!##
    name_upper_folder = f"{dict_param['animal_name']}_1pred_{date}_run/"
    if not os.path.exists(path_top_folder + name_upper_folder):
        os.makedirs(path_top_folder + name_upper_folder)
    plots_folder = path_top_folder + name_upper_folder

    # TODO: should I use time.sleep() to let the message remain for longer? for what I understood it pause the run
    print(f"IMPORTANT: given the presence of nans (missing points of the camera) part of the data are deleted." +
          f"If the number of missing points is 'small enough', it should not interfere with the inference."
          f"Below the ratio of missing points")

    for k in range(dict_param['num_predicotrs']):

        # take only the common (shared, redundant) indices
        mask_raw = np.ma.masked_where(
            np.isnan(data_continous_ratemaps['possiblecovariates'][f"{predictors_name_list[k]}"]) == True,
            data_continous_ratemaps['possiblecovariates'][f"{predictors_name_list[k]}"],
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
            for k in range(dict_param['num_predicotrs']):
                selected_neur_mat = data_binned_glm['spk_mat'][
                    cell_index[j]]
                # brutal binarization (possible presence of 2 or more firing each bin)
                selected_neur_mat = np.where(selected_neur_mat == 0, selected_neur_mat, 1)
                # selecting the first neuron and taking only the shared indices to use the mask
                reduced_matrix = selected_neur_mat[tot_masked_indices_list[k]]
                process_neur.append(reduced_matrix.reshape((T_list[k], 1)))  # reshape in the correct format
                # checking how many points are lost due to nans  in the predictors (no nans in the neural response)
                miss_points_ratio = (tot_time - T_list[k]) / tot_time
                # create a folder for the particular case (model, neuron and predictor) with .txt description###
                print(f"!Fraction missing points is {miss_points_ratio} for the model {i} neuron {j} and predictor {k}")
                name_folder = f"{dict_param['animal_name']}_states={dict_param['list_states'][i]}_numsess={dict_param['num_indep_neurons']}" \
                              + f"_max_iters={dict_param['N_iters']}_tolerance={dict_param['tolerance']}" + \
                              f"_numpredict={1}" + \
                              f"_obs={dict_param['observation_type']}_trans={dict_param['transistion_type']}"
                if not os.path.exists(plots_folder + "/" + name_folder):
                    os.makedirs(plots_folder + "/" + name_folder)

                text_content = f"Single predictor and neuron inference. Predictor selection." "\n" \
                               f"{dict_param['animal_name']}_states={dict_param['list_states'][i]}" \
                               f"_numsess={dict_param['num_indep_neurons']}_max_iters={dict_param['N_iters']}" \
                               f"_tolerance={dict_param['tolerance']}_numpredict={1}" \
                               f"_tot_pred={dict_param['num_predicotrs']}_distal" "\n" \
                               f"obs={dict_param['observation_type']}_trans={dict_param['transistion_type']}" \
                               f"_method={dict_param['optim_method']}" "\n" \
                               f"fraction of missing points={miss_points_ratio}"

                with open(plots_folder + name_folder + "description_parameters.txt", "w") as file:
                    file.write(text_content)

                path_to_plots = plots_folder + name_folder  # save the folder for each inference for single plots
                path_plots_list.append(path_to_plots)

                # create the istances for inference
                test_glmhmm = ssm.HMM(dict_param['list_states'][i], dict_param['num_dimen'], 2,
                                      observations=dict_param['observation_type'],
                                      observation_kwargs=dict(C=dict_param['num_categ_obs']),
                                      transitions=dict_param['transistion_type'])
                glmhmms_ista.append(test_glmhmm)

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


