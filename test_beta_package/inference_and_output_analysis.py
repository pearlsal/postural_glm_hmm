# import ssm
import pickle
import time
""" 
This module contains the function to run the inference and to further process the output
"""

def inference_section(glmhmms_ista, process_neur, inputs_list, dict_param, path_info_dir):
    startclock = time.time()
    time_states_comp = []
    fit_ll_list = []  # redundant, just ravel the one below  #save for each inference,
    # but flat (no list for different states model)
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
            fit_ll_list.append(fit_ll)
            fit_ll_states_list[i].append(fit_ll)
            time_states_comp[i].append(time.time() - partial_clock)
            # TODO: save each comp time to make a statistic?
            print(
                f"computation time loop {(i * (dict_param['num_predicotrs'])) + j} is {time.time() - partial_clock}")
    time_states_comp.append(time.time() - startclock)  # thus time_states_comp length is states*num_pred + 1
    print(f"The total time for the inference is {time.time() - startclock}")
    with open(path_info_dir + 'computation_time_inference.pkl', 'wb') as fp:
        pickle.dump(time_states_comp, fp)

    return fit_ll_list, fit_ll_states_list, glmhmms_ista, time_states_comp


def posterior_prob_process(dict_param, glmhmms_ista, process_neur, inputs_list):
    startclock = time.time()
    posterior_probs_list = []
    print(inputs_list[0][0].shape)
    for i in range(len(dict_param['list_states'])):
        posterior_probs_list.append([])  # structure for different states
        for k in range(dict_param['num_indep_neurons']):
            posterior_probs_list[i].append([])  # structure for different neurons
            for j in range(dict_param['num_predicotrs']):
                posterior_probs = [glmhmms_ista[i * ((dict_param['num_predicotrs']) * dict_param['num_indep_neurons']) +
                                                (k * dict_param['num_indep_neurons']) + j].
                                       expected_states(data=data, input=inpt)[0]
                                   # way to get the posterior probability for each state #check the passage in the
                                   # ssm code
                                   for data, inpt
                                   in zip([process_neur[
                                               i * ((dict_param['num_predicotrs']) * dict_param['num_indep_neurons']) +
                                               (k * dict_param['num_indep_neurons']) + j]], [inputs_list[j][0]])]
                posterior_probs_list[i][k].append(posterior_probs)
    comp_time_posterior = time.time() - startclock
    print(f"Total computation timefor posterior probability is {comp_time_posterior}")

    dict_posterior = {"posterior":posterior_probs_list}
    print(dict_posterior)

    data_file_name = 'dict_posterior.pkl'
    a_file = open(data_file_name, "wb")
    pickle.dump(dict_posterior, a_file)
    a_file.close()

    return posterior_probs_list

