The code is in its first beta version. 

This package is aimed at bridging the postural pipeline of the Whitlock group with a state space model (ssm-> https://github.com/lindermanlab/ssm ). 

The state space model is widely applicable in various sets of systems and contexts. One of its main features is being able to link neural activity with time-dependent predictors and highlight correlation among them. 
Among other applications, this model can extend the Generalized Linear Model (GLM) introducing an arbitrary number of states, each representing a different set of GLM weights.
The procedure of inference is carried out with a Bayesian approach (see "additional_details" document). This technique implies the computation of the posterior probability per each time point. This quantity can be very useful in case the phenomenon we are interested in, is time-dependent.

COMMANDS TO RUN:
First, you have to install the environment using the command: "conda env create -f environment.yml" and activate it.
Secondly, go to "https://github.com/lindermanlab/ssm" and follow the steps to install the ssm package (suggested to download in the top-level folder of the project, outside "postural_glm_hmm"). 
Assuming you have the pickle file obtained from "prepare4ratemap()" and from "prepare_data4glms()" functions of Jingyi's pipeline and you store them in the folder named "data", change the name of the file to the corresponding animal and you are ready to run the "main.py".
The options of the package are multiple, but the combinations are strongly constrained in this version. At the current state of development, the "main.py" is an example containing one of the allowed parameters' structures.

NOTEBOOKS:
The folder "notebooks" contains examples of different types of use of this package. 
These examples cover the tested application of the model with different datasets.
 

The complexity of the model could raise misunderstandings about the variables' roles. The description below should help to clarify the role of each parameter.
NOTATION: 
- num_dimen == number of possibly dependent dimensions. In our frame can be only 1 if you want to have a time-dependent predictor and use input_driven observation.
- num_categorical_obs == when you are using input_driven observation you have to specify the finite possible outcome of your neural response time series. In this frame is equal to 2 (binary, Bernoulli)
- N_iters == maximum number of iterations before interrupting the procedure, regardless of the optimization function value (log-likelihood)
- tolerance == threshold on the optimization function value. Once is reached, the optimization is stopped
- num_indep_neurons == if you want to consider neurons as independent, this is the number of cells you are including (the final likelihood is the product of the probability of each neuron)
- num_predicotrs == number of predictors you want to include in the inference 
- max_num_states == if you want to test the model for a different number of states, you can declare here the maximum number of states you want and the code will test models including 2 states up to "max_num_states" (this is done by "list_states")
- observation_type == type of distribution you want to study. In our frame, the simplest approach would be Bernoulli. Nonetheless, the ssm package allows the use of an input_driven observation to allow any distribution given the constraint on the categorical variable 
- transistion_type == it select the type of transition. In practice, it selects the method and the constraint on the transition probabilities
- optim_method == optimization method to fit the log-likelihood and transition probability (in addition to the initial probabilities on the states)


