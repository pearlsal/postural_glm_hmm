# ####-----test to run the beta package-----#####
from test_beta_package import *

path_data_folder, path_raw_data_folder, path_process_data_folder, path_top_folder, path_info_dir, path_analysis_dir, \
path_plots_dir, path_single_pred_dir, path_multi_pred_dir = folder_structure()

data_continous_ratemaps, data_binned_glm = \
    processing_and_loading(path_raw_data_folder, path_process_data_folder, mat_file="diana_chasing2_notreheaded")


# obtain the information about predictors names, cells ID, animal name and session
predictors_name_list, cells_id, animal_name = get_data_information(path_info_dir, data_continous_ratemaps)
print(f"The animal name is {animal_name}")

# with the function below you assign the parameters for the inference.
# the constraint are described in the file "additional_details.txt"
dict_param = dict_parameters_hmm(path_info_dir,  animal_name, num_dimen=1, num_categ_obs=2, N_iters=100,
                                 tolerance=10 ** -5, num_indep_neurons=2, num_predictors=3, max_num_states=3,
                                 observation_type="input_driven_obs", transistion_type="inputdriven", optim_method="em")
print(f"parameters dictionary is {dict_param}")

# random selection of cells if you are exploring your data
cell_index, tot_time = cells_selection_random(path_info_dir, data_continous_ratemaps, data_binned_glm, dict_param)
print(f"the indices for the selected cells are {cell_index} and the total length of recording is  {tot_time}")

# processing of data and creation of istances for the inference
glmhmms_ista, process_neur, inputs_list, T_list, tot_masked_indices_list, path_plots_list, plots_folder = \
    data_structure(path_info_dir, path_analysis_dir, path_single_pred_dir, data_continous_ratemaps, data_binned_glm,
                   dict_param, tot_time, cell_index)
print(f"list of time length for different predictors is {T_list}")
# inference part # with previous inputs
fit_ll_states_list, glmhmms_ista, time_states_comp = inference_section(path_analysis_dir, path_info_dir, dict_param,
                                                                       glmhmms_ista=glmhmms_ista,
                                                                       process_neur=process_neur,
                                                                       inputs_list=inputs_list)
print(f"Example fit values is {fit_ll_states_list[0]}")

# with input from the code
posterior_probs_list = posterior_prob_process(path_info_dir, path_analysis_dir, dict_param=dict_param,
                                              glmhmms_ista=glmhmms_ista, process_neur=process_neur,
                                              inputs_list=inputs_list)

states_occupancies = states_occupancies_computation(path_analysis_dir, posterior_probs_list)
print(f"States occupancy is {states_occupancies}")

log_like_evolution_per_states(path_analysis_dir, path_info_dir, plots_dir=path_plots_list[0], dict_param=dict_param,
                              fit_ll_states_list=fit_ll_states_list)

posterior_prob_per_states_with_predictor(path_analysis_dir, path_info_dir, posterior_probs_list=posterior_probs_list,
                                             tot_masked_indices_list=tot_masked_indices_list,
                                             T_list=T_list, pred_index=0)

states_occupancies_histogram(path_analysis_dir, path_info_dir, dict_param=dict_param,
                             states_occupancies=states_occupancies)

transition_prob_matrix(path_analysis_dir, path_info_dir, glmhmms_ista=glmhmms_ista, dict_param=dict_param)

weights_distribution_histogram(path_analysis_dir, path_info_dir)
