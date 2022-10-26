The code is in beta version. 

This package is aimed at bridging the postural pipeline of the Whitlock group with a state space model (ssm-> https://github.com/lindermanlab/ssm). It is under testing on Windows 10.

The state space model is widely applicable in various sets of systems and contexts. One of its main features is being able to link neural activity with time-dependent predictors and highlight correlation between them. 
Among other applications, this model can extend the Generalized Linear Model (GLM) introducing an arbitrary number of states, each representing a different set of GLM weights.
The procedure of inference is carried out with a Bayesian approach. This technique implies the computation of the posterior probability per each time point. This quantity can be particularly useful in case the phenomenon we want to study is time-dependent.

COMMANDS TO RUN:
Install git on your machine using the link https://git-scm.com/download/win and following the instructions. Remember to install the second option in the Git setup ("Use Git from the Windows Command Prompt").
Create the folder of your project (e.g. glmhmm_project). Open the powershell and type "git clone https://github.com/teo-fantacci/postural_glm_hmm.git". Type "cd postural_glm_hmm".
First, you have to install the environment using the command: "conda env create -f environment.yml" and then activate it typing "conda activate glmhmmbeta".
Second, type "cd ssm", then "pip install numpy cython" and finally "pip install -e ." to install completely the ssm package. Type "cd .." to move to the parent folder. 
Choose the ".mat" from the GUI you want to analyze and copy it in the folder "/data/raw_data/" and copy paste the name of the file in the "main.py" in the function "processing_and_loading" in "mat_file=" argument. 
Now you can run the "main.py" to obtain the first test run. Type "python main.py" and enter.
The options of the package are multiple, but the combinations of the parameters are constrained in the ssm. At the current state of development, the "main.py" is an example containing one of the allowed parameters' structures.

NOTEBOOKS:
In the main folder there are 3 Jupyter notebooks.
1) "neurons_selection_notebook.ipynb" allows you to select the neurons you are interested in and avoid running the postural pipeline from "file.mat".
2) "multi_predictor_notebook.ipynb" run the inference with multiple predictors. In this case as well you load the already processed "file.pickle".
3) "plots_loading_notebook.ipynb" is a shortcut to load all the processed quantities and have a fast code to run the plots.
 

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


ADDITIONAL DETAILS:
You can find more information in the "additional_details.txt" which includes reference about the theory as well. 

