import scipy.special as scsp
import numpy as npn
import pickle


# ####Including all the simple functions and utilities#####

def sigmoid(x):
    return 1 / (1 + npn.exp(-x))
# -------------------------------------------------------------------------------------------------------------------- #

# remember you should give the permutation factor as well but in this case is always 2 (commutative property)

def factorial_perm(x):
    return npn.int32(scsp.factorial(x) / (scsp.factorial(x - 2) * scsp.factorial(2)))
# -------------------------------------------------------------------------------------------------------------------- #

def dict_transformed_inferred_weights(path_analysis_dir, dict_param, glmhmms_ista):

    inf_weight_dict = {}
    key_states = [str(x) + "_states" for x in dict_param['list_states']]
    for i in range(dict_param['num_states']):
        inf_weight_dict[key_states[i]] = []
        for j in range(dict_param['num_predicotrs']):
            inf_weight_dict[key_states[i]].append(sigmoid(glmhmms_ista[(i*(dict_param['num_predicotrs']))+j].observations.params))

    data_file_name = 'dict_transformed_inferred_weights.pkl'
    a_file = open(path_analysis_dir + data_file_name, "wb")
    pickle.dump(inf_weight_dict, a_file)
    a_file.close()

    return inf_weight_dict


"""
# TODO: to avoid writing in each function 
def colors_number_and_info_plot(colors_number, dict_param):
    
    colormap_size = np.linspace(0, 1, colors_number)
    for i in range(len(colormap_size)):
        colors_states = cmx.jet(colormap_size)
    # name_states = [str(x) + "_states" for x in np.arange(1, state_occupancies.shape[0] +1)
    # use name states in the ticks when automatized?

    dpi = 120  # dots per inch give the size of the picture #! check the one in the papers
    fcc = 'w'  # white background
    ec = 'k'  # black frame
    
    post_description_savefig = f"numsess={dict_param['num_indep_neurons']}_max_iters={dict_param['N_iters']}" \
                               f"_tolerance={dict_param['tolerance']}_numpredict={1}" \
                               f"_tot_pred={dict_param['num_predicotrs']}_obs={dict_param['observation_type']}" \
                               f"_trans={dict_param['transistion_type']}_method={dict_param['optim_method']}" \
                               f"_KAV_s3_distal.pdf"
    
    return colors_states, dpi, fcc, ec, post_description_savefig  
"""