# ADHSMM
The MATLAB codes show simple examples for trajectory generation and control of a robot manipulator, which are built on an adaptive duration hidden semi-Markov model (ADHSMM). The models, algorithms and results given in these codes are part of a project aimed at learning proactive and reactive collaborative robot behaviors.

### Description

	- demo_ADHSMM_logDuration01
		This code implements an adaptive duration hidden semi-Markov model whose duration probabilities are 
		represented by conditional log-normal distributions. The user can:

		1. Define the number of states of the model
		2. Set if the state sequence reconstruction considers the observations or only the duration information
		3. Choose different patterns of external input that condition the duration probabilities
		

	- demo_trajADHSMMlog_online01
		This code implements an online trajectory retrieval that is built on an adaptive duration hidden semi-Markov
		model and the dynamic features of the demonstrated trajectories (inspired by Trajectory HMM). The user can:

		1. Define the number of states of the model
		2. Define the number of dynamic features to be considered into the observation vector of the model
		3. Switch between centered and non-centered time windows for the online trajectory reconstruction
		4. Set the time window length
		5. Set if the state sequence reconstruction considers the observations or only the duration information
		6. Choose different patterns of external input that condition the duration probabilities

		
	- demo_trajADHSMMlogLQR_online01
		This code implements an LQR-based reproduction for an online trajectory retrieval method built on an adaptive
		duration hidden semi-Markov model whose duration probabilities are represented by conditional log-normal 
		distributions. The user can:

		1. Define the number of states of the model
		2. Define the number of dynamic features to be considered into the observation vector of the model
		3. Switch between centered and non-centered time windows for the online trajectory reconstruction
		4. Set the time window length	
		5. Set how much high control inputs are "punished" into the LQR formulation
		6. Set if the state sequence reconstruction considers the observations or only the duration information
		7. Choose different patterns of external input that condition the duration probabilities
		

### References  
	
	[1] Rozo, L., Silvério, J. Calinon, S. and Caldwell, D. (2016). Learning Controllers for Reactive and Proactive 
	Behaviors in Human-Robot Collaboration. Frontiers in Robotics and AI, 3:30, pp. 1-11.

	[2] Rozo, L., Silvério, J. Calinon, S. and Caldwell, D. (2016). Exploiting Interaction Dynamics for Learning 
	Collaborative Robot Behaviors. International Joint Conference on Artificial Intelligence (IJCAI), Workshop on 
	Interactive Machine Learning, New York - USA, pp. 1-7.

### Authors

	Leonel Rozo, Joao Silverio, and Sylvain Calinon
	http://leonelrozo.weebly.com/
	http://programming-by-demonstration.org/
		
	This source code is given for free! In exchange, we would be grateful if you cite the following reference in any 
	academic publication that uses this code or part of it:

	@article{Rozo16Frontiers, 
	  author = "Rozo, L. and Silv\'erio, J. and Calinon, S. and Caldwell, D. G.",
	  title  = "Learning Controllers for Reactive and Proactive Behaviors in Human-Robot Collaboration",
	  journal= "Frontiers in Robotics and {AI}",
	  year   = "2016",
	  month  = "June",
	  volume = "3",
	  number = "30",
	  pages  = "1--11",
	  doi 	 = "10.3389/frobt.2016.00030",
	  note	 = "Specialty Section Robotic Control Systems"
	}
