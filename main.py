# ####-----test to run the beta package-----#####
from test_beta_package import *

data_continous_ratemaps, data_binned_glm, path_top_folder, path_info_dir, path_analysis_dir = \
    load_data("bruno_ready_for_ratemaps_or_GLM_processing_s3.pkl", "bruno_ready_for_GLMHMM_inference_s3.pkl",
              "./data/")
print(path_top_folder)

predictors_name_list = get_data_information(data_continous_ratemaps, path_info_dir, data_binned_glm)
print(predictors_name_list)

dict_param = dict_parameters_hmm(path_info_dir, num_dimen=1, num_categ_obs=2, N_iters=2, tolerance=10 ** -5, num_indep_neurons=2,
                                 num_predicotrs=2, max_num_states=3, observation_type="input_driven_obs",
                                 transistion_type="inputdriven", optim_method="em")
print(dict_param)

cell_index, tot_time = cells_selection_random(data_continous_ratemaps, data_binned_glm, dict_param, path_info_dir)
print(cell_index, tot_time)

#cell_index, tot_time = cells_selection_manual(data_continous_ratemaps, data_binned_glm, "cells_selection.csv", dict_param, path_info_dir)
#print(cell_index, tot_time)

glmhmms_ista, process_neur, inputs_list, T_list, tot_masked_indices_list, path_plots_list, plots_folder = \
    data_structure(dict_param, data_continous_ratemaps, data_binned_glm, path_top_folder, path_analysis_dir,
                   predictors_name_list, tot_time, cell_index)
print(plots_folder)

# inference part #
fit_ll_list, fit_ll_states_list, glmhmms_ista, time_states_comp = inference_section(glmhmms_ista, process_neur,
                                                                                    inputs_list, dict_param,
                                                                                    path_info_dir)
print(fit_ll_states_list, time_states_comp)

posterior_probs_list = posterior_prob_process(dict_param, glmhmms_ista, process_neur, inputs_list, path_info_dir)
print(posterior_probs_list)

states_occupancies = states_occupancies_computation(path_analysis_dir, posterior_probs_list)

log_like_evolution_per_states(fit_ll_states_list, dict_param, plots_folder)

# posterior_prob_per_states_with_predictor(posterior_probs_list, predictors_name_list, data_continous_ratemaps,
#                                         tot_masked_indices_list, T_list, dict_param, plots_folder)

states_occupancies_histogram(path_info_dir, dict_param, states_occupancies=states_occupancies)

transition_prob_matrix(plots_folder, dict_param, glmhmms_ista)

inf_weight_dict = dict_transformed_inferred_weights(path_analysis_dir, dict_param, glmhmms_ista)

weights_distribution_histogram(plots_folder, dict_param, inf_weight_dict)
