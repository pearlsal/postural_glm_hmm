import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
from test_beta_package.utils_glm_hmm import *

"""
Here you can find the implemented plot functions.
Most of them are compatible with all the inference options. 
For some function you need to specify the model and parameters accordingly. 
"""


def log_like_evolution_per_states(path_analysis_dir, path_info_dir, plots_dir=None, dict_param=None,
                                  fit_ll_states_list=None, dict_processed_objects=None,
                                  multipredictor=None):
    """
    All log-likelihoods time evolution grouped by color.
    """

    if dict_processed_objects is not None:
        with open(path_analysis_dir + 'dict_processed_objects.pkl', 'rb') as handle:
            dict_processed_objects = pickle.load(handle)
        fit_ll_states_list = dict_processed_objects["fit_ll_states_list"]

    if multipredictor is None:
        with open(path_analysis_dir + 'dict_objects.pkl', 'rb') as handle:
            dict_objects = pickle.load(handle)
        plots_dir = dict_objects["path_plots_list"][0]
    else:
        with open(path_analysis_dir + 'dict_objects_multicov.pkl', 'rb') as handle:
            dict_objects = pickle.load(handle)
        plots_dir = dict_objects["path_plots_list"][0]

    with open(path_info_dir + 'dictionary_parameters.pkl', 'rb') as handle:
        dict_param = pickle.load(handle)

    with open(path_info_dir + 'dictionary_information.pkl', 'rb') as handle:
        dict_info = pickle.load(handle)
        animal_name = dict_info['animal_name']

    dpi, fcc, ec, post_description_savefig = plot_parameter(dict_param, animal_name)

    if multipredictor is None:
        fig = plt.figure(figsize=(7, 5), dpi=dpi, facecolor=fcc, edgecolor=ec)
        for i in range(len(dict_param['list_states'])):
            colors_states = colors_number(colors_number=dict_param['list_states'][i])
            name_states = [str(x) + "_states" for x in dict_param['list_states']]
            for j in range(dict_param['num_predictors']):  # #!!cluster with lw or color the states and the neurons
                plt.plot(fit_ll_states_list[i][j], color=colors_states[i])  # ,color=CB_color_cycle[i]
            plt.plot(fit_ll_states_list[i][dict_param['num_predictors'] - 1], color=colors_states[i],
                     label=name_states[i])
        plt.legend(loc="best", fontsize=10)
        plt.xlabel("EM Iteration")
        plt.xlim(0, dict_param['N_iters'])
        plt.ylabel("Log-likelihood")
        plt.suptitle("Log-likelihood evolution (EM)")
        plt.tight_layout()
        plt.savefig(plots_dir + f"loglikelihood_time_evolution_" + post_description_savefig, bbox_inches="tight",
                    dpi=dpi)
        plt.show()
    else:
        fig = plt.figure(figsize=(7, 5), dpi=dpi, facecolor=fcc, edgecolor=ec)
        colors_states = colors_number(colors_number=dict_param['list_states'][-1])
        name_states = [str(x) + "_states" for x in dict_param['list_states']]
        for i in range(2):
            plt.plot(fit_ll_states_list[i][0], color=colors_states[i], label=name_states[i])

        plt.legend(loc="best", fontsize=10)
        plt.xlabel("EM Iteration")
        plt.xlim(0, dict_param['N_iters'])
        plt.ylabel("Log-likelihood")
        plt.suptitle("Log-likelihood evolution (EM)")
        plt.tight_layout()
        plt.savefig(plots_dir + f"loglikelihood_time_evolution_" + post_description_savefig, bbox_inches="tight",
                    dpi=dpi)
        plt.show()






# -------------------------------------------------------------------------------------------------------------------- #


def transition_prob_matrix(path_analysis_dir, path_info_dir, glmhmms_ista=None, dict_param=None,
                           dict_processed_objects=None, comp_istance=0, multipredictor=None):
    """
    Plot the probability transition matrix
    """

    if (dict_processed_objects and dict_param) is not None:
        with open(path_analysis_dir + 'dict_processed_objects.pkl', 'rb') as handle:
            dict_processed_objects = pickle.load(handle)
        glmhmms_ista = dict_processed_objects["glmhmms_ista"]

        with open(path_info_dir + 'dictionary_parameters.pkl', 'rb') as handle:
            dict_param = pickle.load(handle)

    if multipredictor is None:
        with open(path_analysis_dir + 'dict_objects.pkl', 'rb') as handle:
            dict_objects = pickle.load(handle)
            plots_dir = dict_objects['path_plots_list'][0]
        comp_istance = dict_param['num_predictors'] + 1
    else:
        with open(path_analysis_dir + 'dict_objects_multicov.pkl', 'rb') as handle:
            dict_objects = pickle.load(handle)
            plots_dir = dict_objects['path_plots_list'][0]


    with open(path_info_dir + 'dictionary_information.pkl', 'rb') as handle:
        dict_info = pickle.load(handle)
        animal_name = dict_info['animal_name']

    dpi, fcc, ec, post_description_savefig = plot_parameter(dict_param, animal_name)

    num_states = glmhmms_ista[comp_istance].transitions.log_Ps.shape[0]
    fig = plt.figure(figsize=(7, 4), dpi=dpi, facecolor=fcc, edgecolor=ec)
    fig.add_subplot(1, 1, 1)
    recovered_trans_mat = np.exp(glmhmms_ista[comp_istance].transitions.log_Ps)
    plt.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
    for i in range(recovered_trans_mat.shape[0]):
        for j in range(recovered_trans_mat.shape[1]):
            text = plt.text(j, i, str(np.around(recovered_trans_mat[i, j], decimals=2)), ha="center", va="center",
                            color="k", fontsize=12)
    plt.xlim(-0.5, num_states - 0.5)
    plt.xticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
    plt.yticks(range(0, num_states), ('1', '2', '3'), fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.title("recovered", fontsize=15)
    plt.tight_layout()
    plt.savefig(plots_dir + f"transition_prob_matrix" + post_description_savefig, bbox_inches="tight", dpi=dpi)
    plt.show()


# -------------------------------------------------------------------------------------------------------------------- #

def weights_distribution_histogram(path_analysis_dir, path_info_dir, multipredictor=None):
    """
    Histogram of parameters distribution. Divided by predictors and bias term, plus combined distribution
    """

    inf_weight_dict, plots_dir, dict_param, animal_name = \
        dict_transformed_inferred_weights(path_analysis_dir, path_info_dir, dict_param=0, dict_processed_objects=0,
                                          multipredictor=multipredictor)

    dpi, fcc, ec, post_description_savefig = plot_parameter(dict_param, animal_name)

    flat_bias_w = []
    flat_predictor_w = []

    if multipredictor is None:
        for key in inf_weight_dict.keys():
            for i in range(dict_param['num_predictors']):
                flat_predictor_w.append(list(inf_weight_dict[key][i][:, 0, 0]))
                flat_bias_w.append(inf_weight_dict[key][i][:, 0, 1])
        flat_predictor_w = list(itertools.chain.from_iterable(flat_predictor_w))
        flat_bias_w = list(itertools.chain.from_iterable(flat_bias_w))
        flat_all_weights = flat_predictor_w + flat_bias_w

    else:
        for key in inf_weight_dict.keys():
            for i in range(dict_param['num_predictors']):
                flat_predictor_w.append(list(inf_weight_dict[key][0][:, 0, i]))
            print(inf_weight_dict[key][0][:, 0, -1])
            print(inf_weight_dict[key][0][:, 0])
            flat_bias_w.append(inf_weight_dict[key][0][:, 0, -1])
        flat_predictor_w = list(itertools.chain.from_iterable(flat_predictor_w))
        flat_bias_w = list(itertools.chain.from_iterable(flat_bias_w))
        flat_all_weights = flat_predictor_w + flat_bias_w

    weights_lists = [flat_predictor_w, flat_bias_w, flat_all_weights]
    titles_histo = ["Predictor weight", "Bias weight", "Both weights"]

    fig = plt.figure(figsize=(15, 10), dpi=dpi, facecolor=fcc, edgecolor=ec)
    for i in range(3):
        fig.add_subplot(2, 2, i + 1)
        counts, edges = np.histogram(weights_lists[i], bins=30)
        plt.step(edges[:-1], counts, alpha=0.5)
        plt.xlabel(f"Weight amplitude", fontsize=10)
        plt.ylabel("Occurrences", fontsize=10)
        plt.title(titles_histo[i])
    plt.tight_layout()
    plt.savefig(plots_dir + f"weights_distribution_histogram" + post_description_savefig, bbox_inches="tight", dpi=dpi)
    plt.show()


# -------------------------------------------------------------------------------------------------------------------- #

def states_occupancies_histogram(path_analysis_dir, path_info_dir, dict_param=None, states_occupancies=None,
                                 file_states_occup=None, multipredictor=None):
    """
    Plot the histogram of cumulative occupancy for each state
    """
    if (file_states_occup and dict_param) is not None:
        with open(path_analysis_dir + 'dict_states_occupancies.pkl', 'rb') as handle:
            dict_states_occupancies = pickle.load(handle)
        states_occupancies = dict_states_occupancies["states_occupancies"]
        print(f"state of occupancy is {states_occupancies}")

        with open(path_info_dir + 'dictionary_parameters.pkl', 'rb') as handle:
            dict_param = pickle.load(handle)
    if multipredictor is None:
        with open(path_analysis_dir + 'dict_objects.pkl', 'rb') as handle:
            dict_objects = pickle.load(handle)
            plots_dir = dict_objects['path_plots_list'][0]

    else:
        with open(path_analysis_dir + 'dict_objects_multicov.pkl', 'rb') as handle:
            dict_objects = pickle.load(handle)
            plots_dir = dict_objects['path_plots_list'][0]

    with open(path_info_dir + 'dictionary_information.pkl', 'rb') as handle:
        dict_info = pickle.load(handle)
        animal_name = dict_info['animal_name']

    if multipredictor is None:
        dpi, fcc, ec, post_description_savefig = plot_parameter(dict_param, animal_name)
        colors_states = colors_number(colors_number=3)

        fig = plt.figure(figsize=(2, 2.5), dpi=dpi, facecolor=fcc, edgecolor=ec)
        for z, occ in enumerate(states_occupancies):
            plt.bar(z, occ, width=0.8, color=colors_states[z])
        plt.ylim((0, 1))
        plt.xticks([0, 1, 2], ['1', '2', '3'], fontsize=10)
        plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
        plt.xlabel('state', fontsize=15)
        plt.ylabel('frac. occupancy', fontsize=15)
        plt.tight_layout()
        plt.savefig(plots_dir + f"states_occupancies_histogram_" + post_description_savefig, bbox_inches="tight",
                    dpi=dpi)
        plt.show()

    else:
        dpi, fcc, ec, post_description_savefig = plot_parameter(dict_param, animal_name)
        colors_states = colors_number(colors_number=3)

        fig = plt.figure(figsize=(2, 2.5), dpi=dpi, facecolor=fcc, edgecolor=ec)
        for z, occ in enumerate(states_occupancies):
            plt.bar(z, occ, width=0.8, color=colors_states[z])
        plt.ylim((0, 1))
        plt.xticks([0, 1, 2], ['1', '2', '3'], fontsize=10)
        plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
        plt.xlabel('state', fontsize=15)
        plt.ylabel('frac. occupancy', fontsize=15)
        plt.tight_layout()
        plt.savefig(plots_dir + f"states_occupancies_histogram_" + post_description_savefig, bbox_inches="tight",
                    dpi=dpi)
        plt.show()





# -------------------------------------------------------------------------------------------------------------------- #

# TODO: generalize
def posterior_prob_per_states_with_predictor(path_analysis_dir, path_info_dir, data_continous_ratemaps,
                                             posterior_probs_list=None,
                                             tot_masked_indices_list=None, T_list=None,
                                             dict_posterior=None,
                                             instance_index=0, pred_index=0, multipredictor=None):
    """
    Posterior probabilities plot comparison with a predictor.
    """

    if (dict_posterior) is not None:
        with open(path_analysis_dir + 'dict_posterior.pkl', 'rb') as handle:
            dict_posterior_objects = pickle.load(handle)
            print(f"the keys of the posterior dict are {dict_posterior_objects.keys()}")
            posterior_probs_list = dict_posterior_objects['posterior_probs_list']

    if multipredictor is None:
        with open(path_analysis_dir + 'dict_objects.pkl', 'rb') as handle:
            dict_objects = pickle.load(handle)
            tot_masked_indices_list = dict_objects['tot_masked_indices_list']
            T_list = dict_objects['T_list']

    else:
        with open(path_analysis_dir + 'dict_objects_multicov.pkl', 'rb') as handle:
            dict_objects = pickle.load(handle)
            tot_masked_indices_list = dict_objects['tot_masked_indices_list']
            T_list = dict_objects['T_list']

    with open(path_analysis_dir + 'dict_objects.pkl', 'rb') as handle:
        dict_objects = pickle.load(handle)
    plots_dir = dict_objects["path_plots_list"][pred_index]
    predictors_name_list = dict_objects['predictors_name_list']

    with open(path_info_dir + 'dictionary_parameters.pkl', 'rb') as handle:
        dict_param = pickle.load(handle)

    with open(path_info_dir + 'dictionary_information.pkl', 'rb') as handle:
        dict_info = pickle.load(handle)
        animal_name = dict_info['animal_name']

    dpi, fcc, ec, post_description_savefig = plot_parameter(dict_param, animal_name)


    # TODO: use as input the index of the state you want to analyze
    model_id = 1  # number of state in the model taken from Klist ([0]==2 states)
    index_istance_struct = instance_index  # index wrt the structure of the glmhmm istances
    num_states = 3
    # TODO: use the name of the predictor and match it
    index_cov_check = pred_index
    name_check_covariate = predictors_name_list[index_cov_check]
    print(name_check_covariate)
    check_covariate = data_continous_ratemaps['possiblecovariates'][f"{name_check_covariate}"]
    norm_factor_check_cov = np.nanmax(abs(check_covariate[tot_masked_indices_list[index_cov_check]]))
    print(f"max value is {norm_factor_check_cov}")
    normalized_check_cov = np.divide(check_covariate[tot_masked_indices_list[index_cov_check]], norm_factor_check_cov)
    print(normalized_check_cov)
    colors = colors_number(num_states+1)

    if multipredictor is None:
        T = T_list[index_cov_check]
    else:
        print(T_list)
        T=T_list

    plotrows = 4
    plotcolumns = 4
    time_interval = 4000
    start_time_list = np.arange(0, T - time_interval, time_interval)
    end_time_list = np.arange(time_interval, T, time_interval)

    if np.amin(normalized_check_cov) < 0:
        ylimin = -1.01
        yticks = [-1, -0.5, 0, 0.5, 1]
    else:
        ylimin = -0.01
        yticks = [0, 0.5, 1]

    fig = plt.figure(figsize=(35, 20), dpi=dpi, facecolor=fcc, edgecolor=ec)

    for j in range(plotcolumns):
        for i in range(plotrows):
            fig.add_subplot(plotrows, plotcolumns, (j * plotrows) + i + 1)
            plt.plot(normalized_check_cov[start_time_list[(j * plotrows) + i]:end_time_list[(j * plotrows) + i]], c="k",
                     label=f"{name_check_covariate}")
            for k in range(num_states):
                plt.plot(posterior_probs_list[model_id][0][index_cov_check][0]
                         [start_time_list[(j * plotrows) + i]:end_time_list[(j * plotrows) + i], k],
                         label=f"State{k}", lw=1,
                         color=colors[k], linestyle="--")

            plt.ylim((ylimin, 1.01))
            plt.yticks(yticks, fontsize=10)
            plt.legend(loc="best", fontsize=8)
            plt.xlabel(f"Time steps {start_time_list[(j * plotrows) + i], end_time_list[(j * plotrows) + i]}",
                       fontsize=15)
            plt.ylabel("p(state)", fontsize=15)
            plt.title("Posterior probability states")
    plt.tight_layout()
    plt.savefig(plots_dir + f"posterior_probability_{name_check_covariate}_" + post_description_savefig, bbox_inches="tight", dpi=dpi)
    plt.show()

# posterior list struture
# first component is the model (how many states)
# second comp is neuron
# third is predictor
# fourth is just one component list thus only 0 (list of comprehension)
# fifth is the acutal array (2 components; (len,state))
# -------------------------------------------------------------------------------------------------------------------- #
