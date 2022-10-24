# ####-----test to run the beta package-----#####
from test_beta_package import *

path_data_folder, path_raw_data_folder, path_process_data_folder, path_top_folder, path_info_dir, path_analysis_dir, \
path_plots_dir, path_single_pred_dir, path_multi_pred_dir = folder_structure()

data_continous_ratemaps, data_binned_glm = \
    processing_and_loading(path_raw_data_folder, path_process_data_folder, # mat_file="diana_chasing2_notreheaded.mat")
                           data_continous_ratemaps_pickle="rm_pre_data_diana_chasing2_notreheaded_XYZeuler_notricks_eo.pkl",
                           data_binned_glm_pickle="binned_4_GLM_rm_pre_data_diana_chasing2_notreheaded_XYZeuler_notricks_eo.pkl")

# obtain the information about predictors names, cells ID, animal name and session
predictors_name_list, cells_id, animal_name = get_data_information(path_info_dir, data_continous_ratemaps)
print(predictors_name_list)

# with the function below you assign the parameters for the inference.
# the constraint are described in the file "additional_details.txt"
dict_param = dict_parameters_hmm(path_info_dir,  animal_name, num_dimen=1, num_categ_obs=2, N_iters=2,
                                 tolerance=10 ** -5, num_indep_neurons=2, num_predictors=2, max_num_states=3, observation_type="input_driven_obs",
                                 transistion_type="inputdriven", optim_method="em")
print(dict_param)

# random selection of cells if you are exploring your data
cell_index, tot_time = cells_selection_random(path_info_dir, data_continous_ratemaps, data_binned_glm, dict_param)
print(cell_index, tot_time)

# once you know which cells you want to analyse, you can give them as input in a csv file. Check inside the function
#cell_index, tot_time = cells_selection_manual(path_info_dir, data_continous_ratemaps, data_binned_glm, dict_param, csv_file_cells)
#print(cell_index, tot_time)

# processing of data and creation of istances for the inference
glmhmms_ista, process_neur, inputs_list, T_list, tot_masked_indices_list, path_plots_list, plots_folder = \
    data_structure(path_info_dir, path_analysis_dir, path_single_pred_dir, data_continous_ratemaps, data_binned_glm, dict_param,
                   tot_time, cell_index, predictors_name_list, predictor_file="best_predictors.csv")
#print(plots_folder)

# glmhmms_ista, process_neur, inpts, T, tot_masked_indices, path_plots_list = \
#    data_structure_multicov(path_analysis_dir, path_info_dir, dict_param, predictors_name_list=None, glmhmms_ista=None,
#                            process_neur=None, inputs_list=None, dict_objects=None,
#                            file_predictors= 'best_predictors.csv')

# inference part # with previous inputs
fit_ll_states_list, glmhmms_ista, time_states_comp = inference_section(path_analysis_dir, path_info_dir, dict_param,
                                                                       glmhmms_ista=glmhmms_ista,
                                                                       process_neur=process_neur,
                                                                       inputs_list=inputs_list) #, dict_objects=0)
"""
# loading the pickle
fit_ll_states_list, glmhmms_ista, time_states_comp = inference_section(path_analysis_dir, path_info_dir, dict_param,
                                                                       glmhmms_ista=glmhmms_ista,
                                                                       process_neur=process_neur,
                                                                       inputs_list=inputs_list)

print(fit_ll_states_list, time_states_comp)
"""
# with input from the code
posterior_probs_list = posterior_prob_process(path_info_dir, path_analysis_dir, dict_processed_objects=0)
# , dict_processed_objects="a")
# loading the pickle
# posterior_probs_list = posterior_prob_process(path_info_dir, path_analysis_dir, dict_param, dict_processed_objects="a") # glmhmms_ista=glmhmms_ista, process_neur=process_neur, inputs_list=inputs_list)



print(posterior_probs_list)

states_occupancies = states_occupancies_computation(path_analysis_dir, posterior_probs_list)
print(f"STATE OF OCCUPANCY IS {states_occupancies}")
log_like_evolution_per_states(path_analysis_dir, path_info_dir, dict_objects=0, dict_processed_objects=0, dictionary_information=0)

# posterior_prob_per_states_with_predictor(posterior_probs_list, predictors_name_list, data_continous_ratemaps,
#                                         tot_masked_indices_list, T_list, dict_param, plots_folder)

states_occupancies_histogram(path_analysis_dir, path_info_dir, dict_param=0, file_states_occup=0,states_occupancies=states_occupancies)

transition_prob_matrix(path_analysis_dir, path_info_dir, dict_param=0, dict_processed_objects=0)

# inf_weight_dict = dict_transformed_inferred_weights(path_analysis_dir, dict_param, glmhmms_ista)

weights_distribution_histogram(path_analysis_dir, path_info_dir)
