The code is in its first beta version. 

This package is aimed at bridging the postural pipeline of the Whitlock group with a state space model (ssm-> https://github.com/lindermanlab/ssm ). 

The state space model is widely applicable in various sets of systems and contexts. One of its main features is being able to link neural activity with time-dependent predictors and highlight correlation among them. 
Among other applications, this model can extend the Generalized Linear Model (GLM) introducing an arbitrary number of states, each representing a different set of GLM weights.
The procedure of inference is carried out with a Bayesian approach (see "additional_details" document). This technique implies the computation of the posterior probability per each time point. This quantity can be very useful in case the phenomenon we are interested in, is time-dependent.

COMMANDS TO RUN:
First, you have to install the environment using the command: "conda env create -f environment.yml" and activate it.
Secondly, go to "https://github.com/lindermanlab/ssm" and follow the steps to install the ssm package (suggested to download in the top level folder of the project, outside "postural_glm_hmm"). 
Assuming you have the pickle file obtained from "prepare4ratemap()" and from "prepare_data4glms()" functions of Jingyi's pipeline and you store them in the folder named "data", you are ready to run the "main.py".
The options of the package are multiple, but the combinations are strongly constrained in this version. At the current state of development, the "main.py" is an example containing one of the allowed parameters' structures.

NOTEBOOKS:
The folder "notebooks" contains examples of different types of use of this package. 
These examples cover the tested application of the model with different datasets.
 

The complexity of the model could raise misunderstandings in the variables role. The description below should help to clarify the role of each parameter.
NOTATION: 
- 
.
.


