# import ssm
import pickle
import time
import numpy as np
""" 
This module contains the function to run the inference and to further process the output
"""

def inference_section(glmhmms_ista, process_neur, inputs_list, dict_param, path_info_dir):
    startclock = time.time()
    time_states_comp = []
    fit_ll_states_list = []  # log-likelihood sorted by number of states model

    # #!introduce the loop for neurons!##
    for i in range(len(dict_param['list_states'])):
        fit_ll_states_list.append([])  # nested list each iteration over the states
        time_states_comp.append([])
        for j in range(dict_param['num_predicotrs']):
            partial_clock = time.time()
            print(j)
            fit_ll = glmhmms_ista[(i * dict_param['num_predicotrs']) + j] \
                .fit(process_neur[(i * dict_param['num_predicotrs']) + j], inputs=inputs_list[j][0],
                     method=dict_param['optim_method'], num_iters=dict_param['N_iters'],
                     tolerance=dict_param['tolerance'])  # actual inference
            fit_ll_states_list[i].append(fit_ll)
            time_states_comp[i].append(time.time() - partial_clock)
            print(
                f"computation time loop {(i * (dict_param['num_predicotrs'])) + j} is {time.time() - partial_clock}")
    time_states_comp.append(time.time() - startclock)  # thus time_states_comp length is states*num_pred + 1
    print(f"The total time for the inference is {time.time() - startclock}")
    with open(path_info_dir + 'computation_time_inference.pkl', 'wb') as fp:
        pickle.dump(time_states_comp, fp)

    return fit_ll_states_list, glmhmms_ista, time_states_comp

# TODO: check if it is better save the posterior in nested dict and not list
def posterior_prob_process(dict_param, glmhmms_ista, process_neur, inputs_list, path_info_dir):
    startclock = time.time()
    posterior_probs_list = []
    for i in range(len(dict_param['list_states'])):
        posterior_probs_list.append([])  # structure for different states
        for k in range(dict_param['num_indep_neurons']):
            posterior_probs_list[i].append([])  # structure for different neurons
            for j in range(dict_param['num_predicotrs']):
                posterior_probs = [glmhmms_ista[i * ((dict_param['num_predicotrs']) * dict_param['num_indep_neurons']) +
                                                (k * dict_param['num_indep_neurons']) + j].
                                       expected_states(data=data, input=inpt)[0]
                                   # way to get the posterior probability for each state
                                   for data, inpt
                                   in zip([process_neur[
                                               i * ((dict_param['num_predicotrs']) * dict_param['num_indep_neurons']) +
                                               (k * dict_param['num_indep_neurons']) + j]], [inputs_list[j][0]])]
                posterior_probs_list[i][k].append(posterior_probs)
    comp_time_posterior = time.time() - startclock
    print(f"Total computation timefor posterior probability is {comp_time_posterior}")

    dict_posterior = {"posterior" : posterior_probs_list}
    print(dict_posterior)

    data_file_name = 'dict_posterior.pkl'
    a_file = open(path_info_dir + data_file_name, "wb")
    pickle.dump(dict_posterior, a_file)
    a_file.close()

    return posterior_probs_list


# TODO: decide if splitting the states_occupancies in 2 function (one for single and the other across all inferences)
def states_occupancies_computation(path_analysis_dir, posterior_probs_list):
    """
    Obtain the state with maximum posterior probability at particular time point and to generate the histogram
    Is possible to concatenate the obtained quantities across different neurons (this can be useful if you are assuming
    the different neurons to be in the same hidden state)
    """

    # for each time point, selecting which is the max of those states value
    state_max_posterior = np.argmax(posterior_probs_list[1][0][0][0], axis=1)

    # now obtain state fractional occupancies:
    _, states_occupancies = np.unique(state_max_posterior, return_counts=True)
    states_occupancies = states_occupancies / np.sum(states_occupancies)

    dict_states_occupancies = {"states_occupancies" : states_occupancies}

    data_file_name = 'dict_states_occupancies.pkl'
    a_file = open(path_analysis_dir + data_file_name, "wb")
    pickle.dump(dict_states_occupancies, a_file)
    a_file.close()

    return states_occupancies

