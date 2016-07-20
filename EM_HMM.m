function [model, GAMMA, s] = EM_HMM(s, model)
%Estimation of HMM parameters with an EM algorithm
%Sylvain Calinon, 2015

%Parameters of the EM algorithm
nbMinSteps = 5; %Minimum number of iterations allowed
nbMaxSteps = 75; %Maximum number of iterations allowed
maxDiffLL = 1E-4; %Likelihood increase threshold to stop the algorithm

diagRegularizationFactor = 1E-6; %Regularization term is optional, see Eq. (2.1.2) in doc/TechnicalReport.pdf

%Initialization of the parameters
nbSamples = length(s);
Data=[];
for n=1:nbSamples
	Data = [Data s(n).Data];
	s(n).nbData = size(s(n).Data,2);
end
[nbVar, nbData] = size(Data);
nbStates = size(model.Sigma,3);

for nbIter=1:nbMaxSteps
	fprintf('.');
	
	%E-step
	for n=1:nbSamples
		
		%Emission probabilities, see Eq. (2.0.5) in doc/TechnicalReport.pdf
		for i=1:nbStates
			s(n).B(i,:) = model.Priors(i) * gaussPDF(s(n).Data, model.Mu(:,i), model.Sigma(:,:,i));
		end
		
		%Forward variable ALPHA, see Eq. (2.5.2) in doc/TechnicalReport.pdf
		s(n).ALPHA(:,1) = model.StatesPriors .* s(n).B(:,1);
		%Scaling to avoid underflow issues
		s(n).c(1) = 1 / sum(s(n).ALPHA(:,1)+realmin);
		s(n).ALPHA(:,1) = s(n).ALPHA(:,1) * s(n).c(1);
		for t=2:s(n).nbData
			s(n).ALPHA(:,t) = (s(n).ALPHA(:,t-1)'*model.Trans)' .* s(n).B(:,t); 
			%Scaling to avoid underflow issues
			s(n).c(t) = 1 / sum(s(n).ALPHA(:,t)+realmin);
			s(n).ALPHA(:,t) = s(n).ALPHA(:,t) * s(n).c(t);
		end
		
		%Backward variable BETA, see Eq. (2.5.3) in doc/TechnicalReport.pdf
		s(n).BETA(:,s(n).nbData) = ones(nbStates,1) * s(n).c(end); %Rescaling
		for t=s(n).nbData-1:-1:1
			s(n).BETA(:,t) = model.Trans * (s(n).BETA(:,t+1) .* s(n).B(:,t+1));
			s(n).BETA(:,t) = min(s(n).BETA(:,t) * s(n).c(t), realmax); %Rescaling
		end
		
		%Intermediate variable GAMMA, see Eq. (2.5.4) in doc/TechnicalReport.pdf
		s(n).GAMMA = (s(n).ALPHA.*s(n).BETA) ./ repmat(sum(s(n).ALPHA.*s(n).BETA)+realmin, nbStates, 1); 
		
		%Intermediate variable XI (fast version, by considering scaling factor), see Eq. (2.5.5) in doc/TechnicalReport.pdf
		for i=1:nbStates
			for j=1:nbStates
				s(n).XI(i,j,:) = model.Trans(i,j) * (s(n).ALPHA(i,1:end-1) .* s(n).B(j,2:end) .* s(n).BETA(j,2:end)); 
			end
		end
	end
	
	%Concatenation of HMM intermediary variables
	GAMMA=[]; GAMMA_TRK=[]; GAMMA_INIT=[]; XI=[];
	for n=1:nbSamples
		GAMMA = [GAMMA s(n).GAMMA];
		GAMMA_INIT = [GAMMA_INIT s(n).GAMMA(:,1)];
		GAMMA_TRK = [GAMMA_TRK s(n).GAMMA(:,1:end-1)];
		XI = cat(3,XI,s(n).XI);
	end
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2)+realmin, 1, size(GAMMA,2));
	
	%M-step
	for i=1:nbStates
		
		%Update the centers, see Eq. (2.5.8) in doc/TechnicalReport.pdf
		model.Mu(:,i) = Data * GAMMA2(i,:)'; 
		
		%Update the covariance matrices, see Eq. (2.5.9) in doc/TechnicalReport.pdf
		Data_tmp = Data - repmat(model.Mu(:,i),1,nbData);
		model.Sigma(:,:,i) = Data_tmp * diag(GAMMA2(i,:)) * Data_tmp'; %Eq. (54) Rabiner
		
		%Regularization term is optional, see Eq. (2.1.2) in doc/TechnicalReport.pdf
		model.Sigma(:,:,i) = model.Sigma(:,:,i) + eye(nbVar) * diagRegularizationFactor;
	end
	
	%Update initial state probability vector, see Eq. (2.5.6) in doc/TechnicalReport.pdf 
	model.StatesPriors = mean(GAMMA_INIT,2); 
	
	%Update transition probabilities, see Eq. (2.5.7) in doc/TechnicalReport.pdf
	model.Trans = sum(XI,3)./ repmat(sum(GAMMA_TRK,2)+realmin, 1, nbStates); 
	
	%Compute the average log-likelihood through the ALPHA scaling factors
	LL(nbIter)=0;
	for n=1:nbSamples
		LL(nbIter) = LL(nbIter) - sum(log(s(n).c));
	end
	LL(nbIter) = LL(nbIter)/nbSamples;
	%Stop the algorithm if EM converged
	if nbIter>nbMinSteps
		if LL(nbIter)-LL(nbIter-1)<maxDiffLL
			disp(['EM converged after ' num2str(nbIter) ' iterations.']);
			return;
		end
	end
end

disp(['The maximum number of ' num2str(nbMaxSteps) ' EM iterations has been reached.']);


