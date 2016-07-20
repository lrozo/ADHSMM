function ADHSMMtoText(ADHSMM, name)
%
% Leonel Rozo, August 2015
%
% Function writes the data from ADHSMM to the following files:
% - ADHSMM_<name>_priors.txt
% - ADHSMM_<name>_trans.txt
% - ADHSMM_<name>_durMu.txt
% - ADHSMM_<name>_durSigma.txt
% - ADHSMM_<name>_mu.txt
% - ADHSMM_<name>_sigma.txt
% - ADHSMM_<name>_varnames.txt
% The supplied HSMM should have the following fields:
% - Sigma         [nbVar x nbVar x nbStates] covariance matrices
% - Mu            [nbVar x nbStates] means
% - StatesPriors  [nbStates x 1]  State priors
% - Trans         [nbStates x nbStates] Transition matrix
% - Mu_Pd         [1 x nbStates]        Matrix of duration means
% - Sigma_Pd      [1 x 1 x nbStates]    Matrix of duration covariances
% - varnames      {1 x nbStates} Cell of variable names

if isdir('./textModels/')==0
	mkdir('./textModels/');
end

% Saving ADHSMM components
dlmwrite(['./textModels/ADHSMM_', name, '_sigma.txt'], ADHSMM.Sigma, ...
	'delimiter', ' ', 'precision','%.6f');
dlmwrite(['./textModels/ADHSMM_', name, '_mu.txt'], ADHSMM.Mu, ...
	'delimiter', ' ', 'precision','%.6f');	
dlmwrite(['./textModels/ADHSMM_', name, '_priors.txt'], ...
	ADHSMM.StatesPriors', 'delimiter', ' ', 'precision','%.6f');
dlmwrite(['./textModels/ADHSMM_', name, '_trans.txt'], ADHSMM.Trans, ...
	'delimiter', ' ', 'precision','%.6f');

% Saving duration GMMs for every ADHSMM component
durPriors = [];
durMu     = [];
durSigma  = [];
for k = 1 : ADHSMM.nbStates
	durPriors = [durPriors ADHSMM.gmm_Pd(k).Priors];
	durMu			= [durMu ADHSMM.gmm_Pd(k).Mu];
	[r,c,w]   = size(ADHSMM.gmm_Pd(k).Sigma);
	durSigma	= [durSigma reshape(ADHSMM.gmm_Pd(k).Sigma,r,c*w)]; 
end
dlmwrite(['./textModels/ADHSMM_', name, '_durPriors.txt'], ...
	durPriors, 'delimiter', ' ', 'precision','%.6f');
dlmwrite(['./textModels/ADHSMM_', name, '_durMu.txt'], durMu, ...
	'delimiter', ' ', 'precision','%.6f');
dlmwrite(['./textModels/ADHSMM_', name, '_durSigma.txt'], ...
	durSigma, 'delimiter', ' ', 'precision','%.6f');

% Write Varnames to file:
varnames = [];
for i = 1:length(ADHSMM.varnames)-1;
	varnames = [varnames, ADHSMM.varnames{i}, ' '];
end
fileID = fopen(['./textModels/ADHSMM_', name, '_varnames.txt'],'w');
varnames = [varnames, ADHSMM.varnames{end}, ' '];
fprintf(fileID,varnames);
fclose(fileID);

end