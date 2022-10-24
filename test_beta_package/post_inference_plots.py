import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import itertools
import pickle
from test_beta_package.utils_glm_hmm import *


# ####-----Plotting section, still with some part of inference output processing-----#####


def log_like_evolution_per_states(path_analysis_dir, path_info_dir, plots_dir=None, dict_param=None,
                                  fit_ll_states_list=None, dict_objects=None, dict_processed_objects=None,
                                  dictionary_information=None):
    """
    All log-likelihoods time evolution grouped by color.
    DONE
    """

    if (dict_processed_objects and dict_objects and dictionary_information) is not None:
        with open(path_analysis_dir + 'dict_processed_objects.pkl', 'rb') as handle:
            dict_processed_objects = pickle.load(handle)
        fit_ll_states_list = dict_processed_objects["fit_ll_states_list"]

        with open(path_analysis_dir + 'dict_objects.pkl', 'rb') as handle:
            dict_objects = pickle.load(handle)
        plots_dir = dict_objects["path_plots_list"][0]

        with open(path_info_dir + 'dictionary_parameters.pkl', 'rb') as handle:
            dict_param = pickle.load(handle)

        with open(path_info_dir + 'dictionary_information.pkl', 'rb') as handle:
            dict_info = pickle.load(handle)
            animal_name = dict_info['animal_name']


    dpi, fcc, ec, post_description_savefig = plot_parameter(dict_param, animal_name)

    print(len(fit_ll_states_list), len(fit_ll_states_list[0]))

    fig = plt.figure(figsize=(7, 5), dpi=dpi, facecolor=fcc, edgecolor=ec)
    for i in range(len(dict_param['list_states'])):
        colors_states = colors_number(colors_number=dict_param['list_states'][i])
        name_states = [str(x) + "_states" for x in dict_param['list_states']]
        for j in range(dict_param['num_predictors']):  # #!!cluster with lw or color the states and the neurons
            plt.plot(fit_ll_states_list[i][j], color=colors_states[i])  # ,color=CB_color_cycle[i]
        plt.plot(fit_ll_states_list[i][dict_param['num_predictors']-1], color=colors_states[i], label=name_states[i])
    plt.legend(loc="best", fontsize=10)
    plt.xlabel("EM Iteration")
    plt.xlim(0, dict_param['N_iters'])
    plt.ylabel("Log-likelihood")
    plt.suptitle("Log-likelihood evolution (EM)")
    plt.tight_layout()
    plt.savefig(plots_dir + f"loglikelihood_time_evolution_" + post_description_savefig, bbox_inches="tight",
                dpi=dpi)
    # plt.switch_backend('Qt5Agg')
    plt.show()
# -------------------------------------------------------------------------------------------------------------------- #

# TODO: generalize. This function shows all the time length thus condition on the product.
#TODO: create another function to visualize from start to end ([start:end])

def posterior_prob_per_states_with_predictor(path_analysis_dir, path_info_dir, data_continous_ratemaps,
                                             posterior_probs_list,
                                             predictors_name_list, tot_masked_indices_list, T_list,
                                             plots_dir=None,
                                             dict_posterior=None, dict_param=None,
                                  dict_processed_objects=None,
                                  dictionary_information=None):
    """
    Posterior probabilities plot comparison with a predictor
    """


    if (dict_processed_objects and dict_posterior and dictionary_information) is not None:
        with open(path_analysis_dir + 'dict_processed_objects.pkl', 'rb') as handle:
            dict_processed_objects = pickle.load(handle)

        with open(path_analysis_dir + 'dict_objects.pkl', 'rb') as handle:
            dict_objects = pickle.load(handle)
        plots_dir = dict_objects["path_plots_list"][0]

        with open(path_info_dir + 'dictionary_parameters.pkl', 'rb') as handle:
            dict_param = pickle.load(handle)

        with open(path_info_dir + 'dictionary_information.pkl', 'rb') as handle:
            dict_info = pickle.load(handle)
            animal_name = dict_info['animal_name']

    dpi, fcc, ec, post_description_savefig = plot_parameter(dict_param, animal_name)

    sess_id = 0  # session id; can choose any index between 0 and num_sess-1
    model_id = 1  # number of state in the model taken from Klist (0=>2 states)
    index_istance_struct = 0  # index wrt the structure of the glmhmm istances
    num_states = 3
    index_cov_check = 0
    name_check_covariate = predictors_name_list[index_cov_check]
    print(name_check_covariate)
    check_covariate = data_continous_ratemaps['possiblecovariates'][f"{name_check_covariate}"]
    norm_factor_check_cov = np.nanmax(abs(check_covariate[tot_masked_indices_list[index_cov_check]]))
    print(f"max value is {norm_factor_check_cov}")
    normalized_check_cov = np.divide(check_covariate[tot_masked_indices_list[index_cov_check]], norm_factor_check_cov)
    print(normalized_check_cov)
    colors = ['r', 'b', 'y', 'g']
    # ls = ["--", ":", "-."]
    lw = [1, 1, 1, 1]
    T = T_list[index_istance_struct]
    plotrows = 5
    plotcolumns = 5
    time_interval = 4500
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
                plt.plot(posterior_probs_list[model_id][sess_id][index_cov_check][0]
                         [start_time_list[(j * plotrows) + i]:end_time_list[(j * plotrows) + i], k],
                         label=f"State{k}", lw=lw[k],
                         color=colors[k], linestyle="--")

            plt.ylim((ylimin, 1.01))
            plt.yticks(yticks, fontsize=10)
            plt.legend(loc="best", fontsize=8)
            plt.xlabel(f"Time steps {start_time_list[(j * plotrows) + i], end_time_list[(j * plotrows) + i]}",
                       fontsize=15)
            plt.ylabel("p(state)", fontsize=15)
            plt.title("Posterior probability states")
    plt.tight_layout()
    plt.savefig(plots_dir + f"posterior_probability_" + post_description_savefig, bbox_inches="tight", dpi=dpi)
    plt.show()

# posterior list struture
# first component is the model (how many states)
# second comp is neuron
# third is predictor
# fourth is just one component list thus only 0 (list of comprehension)
# fifth is the acutal array (2 components; (len,state))
# -------------------------------------------------------------------------------------------------------------------- #

# TODO: generalize for different states
# TODO: impose
def states_occupancies_histogram(path_analysis_dir, path_info_dir, dict_param=None, file_states_occup=None,
                                 states_occupancies=None):
    """
    Plot the histogram of cumulative occupancy for each state
    DONE
    """
    if (file_states_occup and dict_param) is not None:
        with open(path_analysis_dir + 'dict_states_occupancies.pkl', 'rb') as handle:
            dict_states_occupancies = pickle.load(handle)
        states_occupancies = dict_states_occupancies["states_occupancies"]
        print(f"state of occupancy is {states_occupancies}")

        with open(path_info_dir + 'dictionary_parameters.pkl', 'rb') as handle:
            dict_param = pickle.load(handle)

        with open(path_analysis_dir + 'dict_objects.pkl', 'rb') as handle:
            dict_objects = pickle.load(handle)
            plots_dir = dict_objects['path_plots_list'][0]

        with open(path_info_dir + 'dictionary_information.pkl', 'rb') as handle:
            dict_info = pickle.load(handle)
            animal_name = dict_info['animal_name']


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
    plt.savefig(plots_dir + f"states_occupancies_histogram_" + post_description_savefig, bbox_inches="tight", dpi=dpi)
    plt.show()
# -------------------------------------------------------------------------------------------------------------------- #

def transition_prob_matrix(path_analysis_dir, path_info_dir, glmhmms_ista=None, dict_param=None,
                           dict_processed_objects=None):
    """
    Plot the probability transition matrix
    DONE
    """

    if (dict_processed_objects and dict_param) is not None:
        with open(path_analysis_dir + 'dict_processed_objects.pkl', 'rb') as handle:
            dict_processed_objects = pickle.load(handle)
        glmhmms_ista = dict_processed_objects["glmhmms_ista"]

        with open(path_info_dir + 'dictionary_parameters.pkl', 'rb') as handle:
            dict_param = pickle.load(handle)

        with open(path_analysis_dir + 'dict_objects.pkl', 'rb') as handle:
            dict_objects = pickle.load(handle)
            plots_dir = dict_objects['path_plots_list'][0]

        with open(path_info_dir + 'dictionary_information.pkl', 'rb') as handle:
            dict_info = pickle.load(handle)
            animal_name = dict_info['animal_name']

    dpi, fcc, ec, post_description_savefig = plot_parameter(dict_param, animal_name)

    comp_istance = 0
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
    plt.xticks(range(0, num_states), ('1', '2'), fontsize=10)
    plt.yticks(range(0, num_states), ('1', '2'), fontsize=10)
    plt.ylim(num_states - 0.5, -0.5)
    plt.title("recovered", fontsize=15)
    plt.tight_layout()
    plt.savefig(plots_dir + f"transition_prob_matrix" + post_description_savefig, bbox_inches="tight", dpi=dpi)
    plt.show()
# -------------------------------------------------------------------------------------------------------------------- #

def weights_distribution_histogram(path_analysis_dir, path_info_dir):
    """
    Histogram of parameters distribution. Divided by predictors and bias term, plus combined distribution
    """

    inf_weight_dict, plots_dir, dict_param = dict_transformed_inferred_weights(path_analysis_dir, path_info_dir, dict_param=0, dict_processed_objects=0)

    dpi, fcc, ec, post_description_savefig = plot_parameter(dict_param, animal_name)

    flat_bias_w = []
    flat_predictor_w = []
    for key in inf_weight_dict.keys():
        for i in range(dict_param['num_predictors']):
            flat_predictor_w.append(list(inf_weight_dict[key][i][:,0,0]))
            flat_bias_w.append(inf_weight_dict[key][i][:,0,1])
    flat_predictor_w = list(itertools.chain.from_iterable(flat_predictor_w))
    flat_bias_w = list(itertools.chain.from_iterable(flat_bias_w))
    flat_all_weights = flat_predictor_w + flat_bias_w


    weights_lists = [flat_predictor_w, flat_bias_w, flat_all_weights]
    titles_histo = ["Predictor weight", "Bias weight", "Both weights"]

    fig = plt.figure(figsize=(15, 10), dpi=dpi, facecolor=fcc, edgecolor=ec)
    for i in range(3):
        fig.add_subplot(2,2,i+1)
        counts, edges = np.histogram(weights_lists[i], bins=50)
        plt.stairs(counts, edges, fill=True, alpha=0.5)
        plt.xlabel(f"Weight amplitude", fontsize = 10)
        plt.ylabel("Occurrences", fontsize = 10)
        plt.title(titles_histo[i])
    plt.tight_layout()
    plt.savefig(plots_dir + f"weights_distribution_histogram" + post_description_savefig,bbox_inches="tight",dpi=dpi)
    plt.show()
# -------------------------------------------------------------------------------------------------------------------- #

# TODO: generalize the differences and plots for higher number of states
    """
    threshold_diff_pred == if the difference among pair of states is higher, those are the best candidate (other option mutual information)
    """

"""
    
def weights_distribution(dict_param, threshold_diff_pred=0.1):


    ###weights comparison and predictors selection###
    ##!!change all the diff_w_keys with key_states and check!!##

    colormap_size = np.linspace(0, 1, dict_param['num_predictors'] +1)  # color map to have the most distant colours of the colormap
    for i in range(len(colormap_size)):
        colors = cmx.jet(colormap_size)

    inf_weight_dict = {}
    key_states = [str(x) + "_states" for x in dict_param['list_states']]  # no?#!use it in the diff dictionary
    for i in range(dict_param['num_states']):
        inf_weight_dict[key_states[i]] = []
        for j in range(dict_param['num_predictors']):
            inf_weight_dict[key_states[i]].append(sigmoid(glmhmms_ista[(i * (dict_param['num_predictors'])) + j].observations.params))

            # string for labels#
    diff_w_keys = ["state_" + str(x) for x in
                   np.arange(factorial_perm(dict_param['list_states'][-1]))]  # made for easy recall of the dictionary items
    complete_names = [diff_w_keys[0], "diff 0-1", diff_w_keys[1], "diff 1-2", diff_w_keys[2],
                      "diff 0-2"]  ##!!!extend and automatize
    ticks_points = np.linspace(0.05, 0.95, 6)  ##!!generalize it as line just above

    ###!!imporve, not general!!###
    diff_weights_dict = {}  # dictionary to differentiate states and weights
    fig = plt.figure(figsize=(10, 10), dpi=dpi, facecolor=fcc, edgecolor=ec)
    for j in range(dict_param['num_states']):  ##!confusing becuase of the neurons, do it better!
        print(j)
        fig.add_subplot(2, 1, j + 1)
        diff_weights_dict[diff_w_keys[j]] = [[] for a in range(factorial_perm(dict_param['list_states'][
                                                                                  j]))]  # do it better with right string  # create the item of the dictionary with the right number of states to accept the corresponding weights
        if j == 0:  # make it automatic for the number of different model (num states)
            ticks_points = np.linspace(0.05, 0.95, factorial_perm(dict_param['list_states'][j]) + dict_param['list_states'][j])
            for i in range(dict_param['num_predictors']):
                diff_weights_dict[diff_w_keys[j]][0].append(abs(
                    inf_weight_dict[key_states[j]][i][0, 0, 0] - inf_weight_dict[key_states[j]][i][
                        1, 0, 0]))  # taking the difference between same weight but different states
                plt.plot(ticks_points[0:3:2], inf_weight_dict[key_states[j]][i][:, 0, 0], marker="o", color=colors[i],
                         ls="--", lw=0.5)  # plot weights
                plt.scatter(ticks_points[1], diff_weights_dict[diff_w_keys[j]][0][i], marker="D",
                            color=colors[i])  # plot difference of weights

            plt.plot(ticks_points[0:3:2], threshold_diff_pred * np.ones(len(ticks_points[0:3:2])), lw=1, color="k",
                     label="Threshold")
            plt.ylim((-0.05, 1.05))
            plt.xticks(ticks_points[:3], complete_names[:3], fontsize=10)
            plt.xlabel(f"States and differences", fontsize=10)
            plt.ylabel("Weight", fontsize=10)
            plt.gca().set_title('Two states')
            plt.legend(loc="best", fontsize=8)
        elif j == 1:
            ticks_points = np.linspace(0.05, 0.95, factorial_perm(dict_param['list_states'][j]) + dict_param['list_states'][j])
            for i in range(dict_param['num_predictors']):
                diff_weights_dict[diff_w_keys[j]][0].append(
                    abs(inf_weight_dict[key_states[j]][i][0, 0, 0] - inf_weight_dict[key_states[j]][i][1, 0, 0]))
                diff_weights_dict[diff_w_keys[j]][1].append(
                    abs(inf_weight_dict[key_states[j]][i][1, 0, 0] - inf_weight_dict[key_states[j]][i][2, 0, 0]))
                diff_weights_dict[diff_w_keys[j]][2].append(
                    abs(inf_weight_dict[key_states[j]][i][0, 0, 0] - inf_weight_dict[key_states[j]][i][2, 0, 0]))
                plt.plot(ticks_points[0:5:2], inf_weight_dict[key_states[j]][i][:, 0, 0], marker="o", color=colors[i],
                         ls="--", lw=0.5)
                plt.scatter(ticks_points[1:6:2],
                            [diff_weights_dict[diff_w_keys[j]][0][i], diff_weights_dict[diff_w_keys[j]][1][i],
                             diff_weights_dict[diff_w_keys[j]][2][i]], marker="D", color=colors[i])

            plt.plot(ticks_points[0:6:1], threshold_diff_pred * np.ones(len(ticks_points[0:6:1])), lw=1, color="k",
                     label="Threshold")
            plt.ylim((-0.05, 1.05))
            plt.xticks(ticks_points[:6], complete_names[:6], fontsize=10)
            plt.xlabel(f"States and differences", fontsize=10)
            plt.ylabel("Weight", fontsize=10)
            plt.gca().set_title('Three states')
            plt.legend(loc="best", fontsize=8)
        else:
            pass
    plt.suptitle("Predictors' weights", fontsize=15)
    plt.tight_layout()
    plt.savefig(
        general_folder + f'weights_with_differences_states={dict_param['list_states']}_num_sessions={num_sess}_max_iters={N_iters}_tolerance={tolerance}_num_predictors={M - 1}_obs={obslist[0]}_trans={translist[comp_trans]}_method={methodlist[0]}_KAV_s2.pdf',
        bbox_inches="tight", dpi=120)
    plt.show()
    plt.close()

    ###take the max for each predictor difference and then transform to an array and use where to find the indices
    ##do it better and generalize

    best_pred_w = {}
    for i in range(dict_param['num_states']):  # works for 2  #!improve
        best_pred_w[diff_w_keys[i]] = []
        for j in range(dict_param['num_predictors']):
            best_pred_w[diff_w_keys[i]].append(
                max([diff_weights_dict[diff_w_keys[i]][x][j] for x in range(factorial_perm(dict_param['list_states'][i]))]))

    indices_best_pred = [[] for a in range(dict_param['num_states'])]
    for i in range(dict_param['num_states']):
        indices_best_pred[i].append(np.where(np.asarray(best_pred_w[diff_w_keys[i]]) > threshold_diff_pred)[0])

    arr_pred_names = np.asarray(predictors_name_list)
    best_pred_names_dict = {}
    for i in range(dict_param['num_states']):
        best_pred_names_dict[diff_w_keys[i]] = []
        best_pred_names_dict[diff_w_keys[i]].append(arr_pred_names[tuple(indices_best_pred[i])])
    print(best_pred_names_dict)
    # and actual one {predictors_name_list[j]}
    text_content = f"The best predictors are {best_pred_names_dict}" "\n"f"KAV_s2_states={dict_param['list_states'][i]}_numsess={num_sess}_iters={N_iters}_tol={tolerance}_numpredict={1}_tot_pred={M - 1}" "\n" f"observation={obslist[0]}_transition={translist[comp_trans]}_method={methodlist[0]}" "\n" f"predictors={predictors_name_list}" "\n" f"cells={list_string_cells}" "\n" f"fraction of missing points={miss_points_ratio}"
    with open(general_folder + "best_predictors.txt", "w") as file:
        file.write(text_content)

"""