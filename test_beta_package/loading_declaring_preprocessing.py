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

    # top structure folder to include all outputs#
    top_folder = f"results_folder"  # _predict={predictors_name_list[j]}
    if not os.path.exists("./" + top_folder):
        os.makedirs("./" + top_folder)
    path_top_folder = os.path.dirname(os.path.abspath("./" + top_folder)) + "/results_folder/"
    print(f"top folder is {path_top_folder}")

    info_dir = f"info_dir/"
    if not os.path.exists(path_top_folder + info_dir):
        os.makedirs(path_top_folder + info_dir)
    path_info_dir = os.path.dirname(os.path.abspath(path_top_folder + info_dir)) + "/info_dir/"
    print(f"absolute path is {path_info_dir}")

    analysis_dir = f"analysis_dir/"
    if not os.path.exists(path_top_folder + analysis_dir):
        os.makedirs(path_top_folder + analysis_dir)
    path_analysis_dir = os.path.dirname(os.path.abspath(path_top_folder + analysis_dir)) + "/analysis_dir/"
    print(f"absolute path is {path_analysis_dir}")

    # TODO: procesed_data folder to store the various dictionary with objects and quantities

    # TODO: create the path to data for actual data and csv and all useful files pre running
    # TODO: or even separate the folder function so that it creates the structure before

    return data_continous_ratemaps, data_binned_glm, path_top_folder, path_info_dir, path_analysis_dir


"""

"""


# TODO: save txt-csv with all information
def get_data_information(data_continous_ratemaps, path_info_dir, data_binned_glm):  # finish!!!!  , data_binned_glm, path_top_folder
    """
    This function extract useful information of data and save them in a file.txt
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


def dict_parameters_hmm(path_info_dir, num_dimen, num_categ_obs, N_iters, tolerance, num_indep_neurons, num_predicotrs,
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

    # TODO: change to with open .....
    data_file_name = 'dictionary_parameters.pkl'
    with open(path_info_dir + data_file_name, "wb") as handle:
        pickle.dump(dict_param, handle)
    # data_file_name = 'dictionary_parameters.pkl'
    # a_file = open(path_info_dir + data_file_name, "wb")   # need to add /before the name?
    # pickle.dump(dict_param, a_file)
    # a_file.close()

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
    print(cells_to_infer)
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

    # with open(csv_file_cells) as fp:
    #     reader = csv.reader(fp, delimiter=",", quotechar='"')
    #     data_read = [row for row in reader]

    data_read = pd.read_csv(csv_file_cells)
    print(f"the neurons from csv are {data_read.iloc[0]}")
    cells_index = []
    for i in range(dict_param['num_indep_neurons']):
        cells_index.append(np.where(cell_names_array == data_read.iloc[i,0])[0][0])
        # if more than one probably == [0][0], if only one == [0]
    print(cells_index)

    tot_time = len(data_binned_glm['spk_mat'][0])
    print(tot_time)

    text_content = f"ID cells selected for this inference is {data_read} and the corresponding indices are {cells_index}"

    with open(path_info_dir + "cells_list_and_indices.txt", "w") as file:
        file.write(text_content)

    return cells_index, tot_time


# TODO: capital print to warn about the deleted time points of the mask and
def data_structure(dict_param, data_continous_ratemaps, data_binned_glm, path_top_folder,
                   path_analysis_dir,  predictors_name_list, tot_time, cell_index):
    # #!!remember first folder "path_plots", insert in the right function to have the .txt and other files in
    # the right place!!##
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
    name_upper_folder = f"KAV_s3_distal_1pred_{date}_run/"
    if not os.path.exists(path_top_folder + name_upper_folder):
        os.makedirs(path_top_folder + name_upper_folder)
    plots_folder = path_top_folder + name_upper_folder
    print(plots_folder)

    for k in range(dict_param['num_predicotrs']):
        # #take only the common (shared, redundant) indices##

        mask_raw = np.ma.masked_where(
            np.isnan(data_continous_ratemaps['possiblecovariates'][f"{predictors_name_list[k]}"]) == True,
            data_continous_ratemaps['possiblecovariates'][f"{predictors_name_list[k]}"],
            copy=True)  # condition, object to evaluate, create a separate copy
        tot_mask_indices = np.where(np.isfinite(mask_raw))[
            0]  # find the index for the finite elements(not nan)
        tot_masked_indices_list.append(tot_mask_indices)
        T_list.append(len(tot_mask_indices))

        # #assign the processed predictors containing only the common indices and adding the bias term initialized to 1
        # (!important, shouldn't be a zero)
        inpts = np.ones((dict_param['num_indep_neurons'], T_list[k], 2),
                        dtype=np.float64)  # maybe float32 considerably faster? # initialize inpts array
        # (the bias should be one to avoid problems. Important not zeros)
        inpts[:, :, 0] = data_continous_ratemaps['possiblecovariates'][f"{predictors_name_list[k]}"][tot_mask_indices]
        # it 0 becuase only one predictor in this module...
        inpts = list(inpts)  # convert inpts to correct format   # in partiular if you have different length of the time
        # the spike trains or for the predictors
        inputs_list.append(inpts)
    # print(states, dict_param['num_indep_neurons'], M-1)

    # #!create the structure for multiple neurons, thus a nested list for each first loop!##
    process_neur = []  # requested list structure for multiple neurons with sessions
    for i in range(len(dict_param['list_states'])):  # states
        for j in range(dict_param['num_indep_neurons']):  # neurons
            for k in range(dict_param['num_predicotrs']):  # predictors
                # #select the neuron in the list above and the valid (shared among the predictors) indices.
                # Then generate the right structure.##
                selected_neur_mat = data_binned_glm['spk_mat'][
                    cell_index[j]]  # double check if the neurons are correct

                selected_neur_mat = np.where(selected_neur_mat == 0, selected_neur_mat, 1)
                # brutal binarization (possible presence of 2 or more firing each bin) #?ok or better way to do it
                reduced_matrix = selected_neur_mat[
                    tot_masked_indices_list[k]]  # selectiong the first neuron and taking only the shared indices
                process_neur.append(reduced_matrix.reshape((T_list[k], 1)))  # reshape in the correct format
                # #if multiple neurons, generalize    #process_neur.append(reduced_matrix[m].reshape((T,1)))
                miss_points_ratio = (tot_time - T_list[k]) / tot_time  # checking how many points are lost due to nan
                # in the precictors (no nans in the neural response)
                # ##create a folder for the particular case with .txt description###

                name_folder = f"KAV_s3_states={dict_param['list_states'][i]}_numsess={dict_param['num_indep_neurons']}" \
                              + f"_max_iters={dict_param['N_iters']}_tolerance={dict_param['tolerance']}" + \
                              f"_numpredict={1}" + \
                              f"_obs={dict_param['observation_type']}_trans={dict_param['transistion_type']}" + \
                              f"_distal/"
                if not os.path.exists(plots_folder + "/" + name_folder):
                    os.makedirs(plots_folder + "/" + name_folder)

                text_content = f"Single predictor and neuron inference. Predictor selection." "\n" \
                               f"KAV_s3_states={dict_param['list_states'][i]}" \
                               f"_numsess={dict_param['num_indep_neurons']}_max_iters={dict_param['N_iters']}" \
                               f"_tolerance={dict_param['tolerance']}_numpredict={1}" \
                               f"_tot_pred={dict_param['num_predicotrs']}_distal" "\n" \
                               f"obs={dict_param['observation_type']}_trans={dict_param['transistion_type']}" \
                               f"_method={dict_param['optim_method']}" "\n" \
                               f"fraction of missing points={miss_points_ratio}"
                # f"predictors={predictors_name_list}" "\n" f"cells={list_string_cells}" "\n"

                with open(plots_folder + name_folder + "description_parameters.txt", "w") as file:
                    file.write(text_content)

                path_to_plots = plots_folder + name_folder  # save the folder for each inference for single plots
                path_plots_list.append(path_to_plots)

                test_glmhmm = ssm.HMM(dict_param['list_states'][i], dict_param['num_dimen'], 2,
                                      observations=dict_param['observation_type'],
                                      observation_kwargs=dict(C=dict_param['num_categ_obs']),
                                      transitions=dict_param['transistion_type'])
                glmhmms_ista.append(test_glmhmm)

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


