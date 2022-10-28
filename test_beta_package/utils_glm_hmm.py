import scipy.special as scsp
import numpy as np
import pickle
import matplotlib.cm as cmx
from test_beta_package import *

""" 
Here are coded the simple and recurrent functions.
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -------------------------------------------------------------------------------------------------------------------- #

# remember you should give the permutation factor as well but in this case is always 2 (commutative property)
def factorial_perm(x):
    return np.int32(scsp.factorial(x) / (scsp.factorial(x - 2) * scsp.factorial(2)))


# -------------------------------------------------------------------------------------------------------------------- #


def dict_transformed_inferred_weights(path_analysis_dir, path_info_dir, glmhmms_ista=None, dict_param=None,
                                      dict_processed_objects=None, multipredictor=None):
    """
    Create a dictionary with the transformed weights
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

    else:
        with open(path_analysis_dir + 'dict_objects_multicov.pkl', 'rb') as handle:
            dict_objects = pickle.load(handle)
            plots_dir = dict_objects['path_plots_list'][0]

    with open(path_info_dir + 'dictionary_information.pkl', 'rb') as handle:
        dict_info = pickle.load(handle)
        animal_name = dict_info['animal_name']


    if multipredictor is None:
        inf_weight_dict = {}
        key_states = [str(x) + "_states" for x in dict_param['list_states']]
        for i in range(dict_param['num_states']):
            inf_weight_dict[key_states[i]] = []
            for j in range(dict_param['num_predictors']):
                inf_weight_dict[key_states[i]].append(
                    sigmoid(glmhmms_ista[(i * (dict_param['num_predictors'])) + j].observations.params))
        print(f"inferred and transformed weights are {inf_weight_dict}")
        data_file_name = 'dict_transformed_inferred_weights.pkl'
        a_file = open(path_analysis_dir + data_file_name, "wb")
        pickle.dump(inf_weight_dict, a_file)
        a_file.close()

    else:
        inf_weight_dict = {}
        key_states = [str(x) + "_states" for x in dict_param['list_states']]
        for i in range(dict_param['num_states']):
            inf_weight_dict[key_states[i]] = []
            inf_weight_dict[key_states[i]].append(sigmoid(glmhmms_ista[i].observations.params))
        print(f"inferred and transformed weights are {inf_weight_dict}")
        data_file_name = 'dict_transformed_inferred_weights_multi_predictor.pkl'
        a_file = open(path_analysis_dir + data_file_name, "wb")
        pickle.dump(inf_weight_dict, a_file)
        a_file.close()

    return inf_weight_dict, plots_dir, dict_param, animal_name


# -------------------------------------------------------------------------------------------------------------------- #

def colors_number(colors_number):
    """
    Create array of colors for plotting
    """

    colormap_size = np.linspace(0, 1, colors_number)
    for i in range(len(colormap_size)):
        colors_states = cmx.jet(colormap_size)

    return colors_states

# -------------------------------------------------------------------------------------------------------------------- #

def plot_parameter(dict_param, animal_name):
    """
    Simple function to get parameters for plots
    """
    dpi = 120  # dots per inch give the size of the picture
    fcc = 'w'  # white background
    ec = 'k'  # black frame

    post_description_savefig = f"neur={dict_param['num_indep_neurons']}_max_iters={dict_param['N_iters']}" \
                               f"_tolerance={dict_param['tolerance']}" \
                               f"_tot_pred={dict_param['num_predictors']}_obs={dict_param['observation_type']}" \
                               f"_trans={dict_param['transistion_type']}_method={dict_param['optim_method']}" \
                               f"_{animal_name}.pdf"

    return dpi, fcc, ec, post_description_savefig