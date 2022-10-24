# import ssm
import pickle
import time
import numpy as np
# from test_beta_package.utils_glm_hmm import *

from test_beta_package import *

""" 
This module contains the function to run the inference and to further process the output
"""


def inference_section(path_analysis_dir, path_info_dir, dict_param, glmhmms_ista=None, process_neur=None,
                      inputs_list=None, dict_objects=None):
    """
    This function performs the fitting procedure.
    It creates a structure for the output quantities to be handled easier (per states)
    If you want to run the function separately, after you already did the inference, you should put dict_objects="a" or
    number or whatever other quantities acceptable as input (only a without quotes will not work)
    """
    startclock = time.time()
    time_states_comp = []
    fit_ll_states_list = []  # log-likelihood sorted by number of states model

    if dict_objects is not None:
        with open(path_analysis_dir + 'dict_objects.pkl', 'rb') as handle:
            dict_objects = pickle.load(handle)
        glmhmms_ista = dict_objects["glmhmms_ista"]
        process_neur = dict_objects["process_neur"]
        inputs_list = dict_objects["inputs_list"]

    # #!introduce the loop for neurons!##
    for i in range(len(dict_param['list_states'])):
        fit_ll_states_list.append([])  # nested list each iteration over the states
        time_states_comp.append([])
        for j in range(dict_param['num_predictors']):
            partial_clock = time.time()
            print(j)
            fit_ll = glmhmms_ista[(i * dict_param['num_predictors']) + j] \
                .fit(process_neur[(i * dict_param['num_predictors']) + j], inputs=inputs_list[j][0],
                     method=dict_param['optim_method'], num_iters=dict_param['N_iters'],
                     tolerance=dict_param['tolerance'])  # actual inference
            fit_ll_states_list[i].append(fit_ll)
            time_states_comp[i].append(time.time() - partial_clock)
            print(
                f"computation time loop {(i * (dict_param['num_predictors'])) + j} is {time.time() - partial_clock}")
    time_states_comp.append(time.time() - startclock)  # thus time_states_comp length is states*num_pred + 1
    print(f"The total time for the inference is {time.time() - startclock}")
    with open(path_info_dir + 'computation_time_inference.pkl', 'wb') as fp:
        pickle.dump(time_states_comp, fp)

    dict_processed_objects = {}
    dict_processed_objects["glmhmms_ista"] = glmhmms_ista
    dict_processed_objects["process_neur"] = process_neur
    dict_processed_objects["inputs_list"] = inputs_list
    dict_processed_objects["fit_ll_states_list"] = fit_ll_states_list
    dict_processed_objects["time_states_comp"] = time_states_comp

    data_file_name = 'dict_processed_objects.pkl'
    a_file = open(path_analysis_dir + data_file_name, "wb")
    pickle.dump(dict_processed_objects, a_file)
    a_file.close()

    print(f"the input list after inference is {inputs_list[0][0].shape}")

    return fit_ll_states_list, glmhmms_ista, time_states_comp


# -------------------------------------------------------------------------------------------------------------------- #



# TODO: check if it is better save the posterior in nested dict and not list
def posterior_prob_process(path_info_dir, path_analysis_dir, dict_param=None, glmhmms_ista=None, process_neur=None,
                           inputs_list=None, dict_processed_objects=None):
    """
    This function computes the posterior probability for each model, state and neuron (in this case single predictor as
    well)
    The objects are saved in a pickle file for further processing in case the inference was time consuming
    In addition it saved the computation time for each fitting procedure for computational time statistic
    """

    startclock = time.time()
    posterior_probs_list = []

    if dict_processed_objects is not None:
        with open(path_analysis_dir + 'dict_processed_objects.pkl', 'rb') as handle:
            dict_processed_objects = pickle.load(handle)
        glmhmms_ista = dict_processed_objects["glmhmms_ista"]
        process_neur = dict_processed_objects["process_neur"]
        inputs_list = dict_processed_objects["inputs_list"]
        with open(path_info_dir + 'dictionary_parameters.pkl', 'rb') as handle:
            dict_param = pickle.load(handle)

    for i in range(len(dict_param['list_states'])):
        posterior_probs_list.append([])  # structure for different states
        for k in range(dict_param['num_indep_neurons']):
            posterior_probs_list[i].append([])  # structure for different neurons
            for j in range(dict_param['num_predictors']):
                posterior_probs = [glmhmms_ista[i * ((dict_param['num_predictors']) * dict_param['num_indep_neurons']) +
                                                (k * dict_param['num_predictors']) + j].
                                       expected_states(data=data, input=inpt)[0]
                                   # way to get the posterior probability for each state
                                   for data, inpt
                                   in zip([process_neur[
                                               i * ((dict_param['num_predictors']) * dict_param['num_indep_neurons']) +
                                               (k * dict_param['num_predictors']) + j]], [inputs_list[j][0]])]
                posterior_probs_list[i][k].append(posterior_probs)
    comp_time_posterior = time.time() - startclock
    print(f"Total computation time for posterior probability is {comp_time_posterior}")

    dict_posterior = {"posterior_probs_list": posterior_probs_list}

    data_file_name = 'dict_posterior.pkl'
    a_file = open(path_analysis_dir + data_file_name, "wb")
    pickle.dump(dict_posterior, a_file)
    a_file.close()

    return posterior_probs_list


# -------------------------------------------------------------------------------------------------------------------- #

# TODO: decide if splitting the states_occupancies in 2 function (one for single and the other across all inferences)
def states_occupancies_computation(path_analysis_dir, posterior_probs_list=None, dict_posterior=None):
    """
    Obtain the state with maximum posterior probability at particular time point and to generate the histogram
    Is possible to concatenate the obtained quantities across different neurons (this can be useful if you are assuming
    the different neurons to be in the same hidden state)
    """

    if dict_posterior is not None:
        with open(path_analysis_dir + 'dict_posterior.pkl', 'rb') as handle:
            dict_posterior_loaded = pickle.load(handle)
        posterior_probs_list = dict_posterior_loaded["posterior_probs_list"]
        print(f"state of occupancy is {posterior_probs_list}")

    # for each time point, selecting which is the max of those states value
    state_max_posterior = np.argmax(posterior_probs_list[1][0][0][0], axis=1)

    # now obtain state fractional occupancies:
    _, states_occupancies = np.unique(state_max_posterior, return_counts=True)
    states_occupancies = states_occupancies / np.sum(states_occupancies)

    dict_states_occupancies = {"states_occupancies": states_occupancies}

    data_file_name = 'dict_states_occupancies.pkl'
    a_file = open(path_analysis_dir + data_file_name, "wb")
    pickle.dump(dict_states_occupancies, a_file)
    a_file.close()

    return states_occupancies


def predictors_selection_post_inference(path_analysis_dir, dict_param, glmhmms_ista, predictors_name_list, ):
    """

    """
    from postural_glm_hmm.test_beta_package import factorial_perm
    from postural_glm_hmm.test_beta_package import dict_transformed_inferred_weights

    inf_weight_dict = dict_transformed_inferred_weights(path_analysis_dir, dict_param, glmhmms_ista)

    # string for labels#
    state = "state"
    diff = "diff" # differences among the weights
    dict_diff = {}
    # TODO: assert dict_param['max_num_states']<6
    for i in range(dict_param['max_num_states']-1):
        print(f"iteration number {i}")
        print(f"the len of iteration should be 2 and it is {dict_param['max_num_states']-1}")

        for j in range(factorial_perm(dict_param['list_states'][i])):
            print(f"the factorial_perm number is {factorial_perm(dict_param['list_states'][i])}")
            dict_key = state + f"{dict_param['list_states'][i]}"
            dict_diff[f'{dict_diff}'] = []

            dict_diff[f'{dict_diff}'].append()




    diff_w_keys = ["diff_" + str(x) + "-" + str for x in
                   np.arange(factorial_perm(dict_param['list_states'][-1]))]  # made for easy recall of the dictionary items
    complete_names = [diff_w_keys[0], "diff 0-1", diff_w_keys[1], "diff 1-2", diff_w_keys[2],
                      "diff 0-2"]  ##!!!extend and automatize
    ticks_points = np.linspace(0.05, 0.95, 6)  ##!!generalize it as line just above

    #
    # TODO: make it automatic for n differences with factorial permutation
    diff_weights_dict[diff_w_keys[j]][0].append(abs(
        inf_weight_dict[key_states[j]][i][0, 0, 0] - inf_weight_dict[key_states[j]][i][
            1, 0, 0]))  # taking the difference between same weight but different states

    for i in range(dict_param['num_predictors']):
        diff_weights_dict[diff_w_keys[j]][0].append(
            abs(inf_weight_dict[key_states[j]][i][0, 0, 0] - inf_weight_dict[key_states[j]][i][1, 0, 0]))
        diff_weights_dict[diff_w_keys[j]][1].append(
            abs(inf_weight_dict[key_states[j]][i][1, 0, 0] - inf_weight_dict[key_states[j]][i][2, 0, 0]))
        diff_weights_dict[diff_w_keys[j]][2].append(
            abs(inf_weight_dict[key_states[j]][i][0, 0, 0] - inf_weight_dict[key_states[j]][i][2, 0, 0]))

        ###take the max for each predictor difference and then transform to an array and use where to find the indices
        ##do it better and generalize

    best_pred_w = {}
    for i in range(dict_param['num_states']):  # works for 2  #!improve
        best_pred_w[diff_w_keys[i]] = []
        for j in range(dict_param['num_predictors']):
            best_pred_w[diff_w_keys[i]].append(
                max([diff_weights_dict[diff_w_keys[i]][x][j] for x in
                     range(factorial_perm(dict_param['list_states'][i]))]))

    indices_best_pred = [[] for a in range(dict_param['num_states'])]
    for i in range(dict_param['num_states']):
        indices_best_pred[i].append(np.where(np.asarray(best_pred_w[diff_w_keys[i]]) > dict_param["threshold_diff_pred"])[0])

    arr_pred_names = np.asarray(predictors_name_list)
    best_pred_names_dict = {}
    for i in range(dict_param['num_states']):
        best_pred_names_dict[diff_w_keys[i]] = []
        best_pred_names_dict[diff_w_keys[i]].append(arr_pred_names[tuple(indices_best_pred[i])])
    print(best_pred_names_dict)
    # and actual one {predictors_name_list[j]}
    text_content = f"The best predictors are {best_pred_names_dict}" "\n" f"{dict_param['animal_name']}, " \
                   f"number states model ={dict_param['list_states'][i]}, number independent neurons " \
                   f"={dict_param['num_indep_neurons']}, max number iterations ={dict_param['N_iters']}, " \
                   f"tolorance ={dict_param['tolerance']}", f"single predictor inference selection, " \
                   f"total number of predictors ={dict_param['num_predictors']}" \
                   "\n" f"observation={dict_param['observation_type']}_, ransition={dict_param['transistion_type']}, " \
                   f"optimization method={dict_param['optim_method']}" "\n" f"predictors={predictors_name_list}" "\n" \
                   f"cells={list_string_cells}" "\n" f"fraction of missing points={miss_points_ratio}"

    with open(general_folder + "best_predictors.txt", "w") as file:
        file.write(text_content)


    return inf_weight_dict

