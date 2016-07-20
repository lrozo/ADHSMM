function demo_ADHSMM_logDuration01

% Leonel Rozo, 2016
%
% Example of an adaptive duration model implemented as a hidden semi-Markov 
% model with adaptive duration probabilities.
% 
% - This model adapts the log-normal state duration probability according 
%		to an external input "u". 
% - Every model state has a duration probability represented by a mixture of 
%   Gaussians. 
% - A conditional Gaussian distribution is obtained at each time step by
%   applying GMR given the external input "u".
%
% This code:
%		1. Sets the variable values for the ADHSMM, and a linear quadratic
%		   regulator in charge of following a step-wise trajectory obtained 
%			 from the forward variable of the model.
%		2. Loads synthetic data to be used for training the model. The data
%			 correspond to several G-shape trajectories in 2D.
%		3. Trains the model in two phases: (i) an HSMM is trained, (ii)
%			 duration probabilities (GMM) for each model state are set manually
%			 (these can be easily learned from EM for a Gaussian mixture model).
%		4. Reconstructs a state sequence where the state duration depends on
%			 the given external input.
%		5. Retrieves a reproduction trajectory by implementing a linear
%			 quadratic regulator that follows the step-wise reference obtained
%			 from the state sequence previously computed.
%		6. Plots the results in dynamic graphs.
%


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -----> Model variables
model.nbStates = 7; % Number of states
model.dt = 1;				% Time step duration
nbData  = 200;      % Number of datapoints
onlyDur = 0;        % 0 -> Standard computation, 1 -> Only duration

% -----> LQR variables
rFactor = 5.5E0;    % Acceleration cost in LQR

% -----> Other variables
saveTxt = 0;				% Flag to save .txt files for pbdlib


%% Load AMARSI data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/GShape.mat');
%nbSamples = length(demos);
nbSamples = 3;
Data=[];
for n = 1 : nbSamples
  s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, ...
    linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
  Data = [Data s(n).Data]; 
end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -----> K-means initialization
%model = init_GMM_kmeans(Data, model);
model = init_GMM_timeBased([repmat(1:nbData,1,nbSamples); Data], model);
model.Mu = model.Mu(2:end,:);
model.Sigma = model.Sigma(2:end,2:end,:);

% -----> Transition matrix initialization
% %Random initialization
% model.Trans = rand(model.nbStates,model.nbStates);
% model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);
% model.StatesPriors = rand(model.nbStates,1);
% model.StatesPriors = model.StatesPriors/sum(model.StatesPriors);

%Left-right model initialization
model.Trans = zeros(model.nbStates);
for i=1:model.nbStates-1
	model.Trans(i,i) = 1-(model.nbStates/nbData);
	model.Trans(i,i+1) = model.nbStates/nbData;
end
model.Trans(model.nbStates,model.nbStates) = 1.0;
model.StatesPriors = zeros(model.nbStates,1);
model.StatesPriors(1) = 1;
model.Priors = ones(model.nbStates,1);

% -----> EM learning
[model, H] = EM_HMM(s, model);
%Removal of self-transition (for HSMM representation) and normalization
model.Trans = model.Trans - diag(diag(model.Trans)) + eye(model.nbStates)*realmin;
model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);

% -----> Learning duration probabilities
% % Set state duration manually (for HSMM representation)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -> Discrete case:
%   -Assumptions: 
%       + u = 0 -> Short duration,    u = 1 -> Long duration
%       + All the HSMM states have the same duration prob. distribution
for i = 1 : model.nbStates
  % 2 Gaussians composing the state duration probability.  
  model.gmm_Pd(i).nbStates = 2;
  model.gmm_Pd(i).Priors = ones(model.gmm_Pd(i).nbStates,1);
  model.gmm_Pd(i).Mu = [0.0 1.0 ; 3 4.6];
  model.gmm_Pd(i).Sigma(:,:,1) = [1E-2 0.0 ; 0.0 0.11];
  model.gmm_Pd(i).Sigma(:,:,2) = [1E-2 0.0 ; 0.0 0.28];
end


%% Reconstruction of states probability sequence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -----> Designing external input
% Artificial external (discrete) input
% u = zeros(nbData,1);
% u = ones(nbData,1);
u = [zeros(45,1) ; ones(45,1) ; zeros(110,1)];
% u = [zeros(45,1) ; ones(10,1) ; zeros(145,1)];
% u = [ones(40,1) ; zeros(80, 1); ones(80,1)];
% u = [zeros(50,1) ; ones(50,1) ; zeros(45,1) ; ones(35,1) ; zeros(20,1)];

% -----> Initial conditions for reconstruction and reproduction
currPos = s(1).Data(:,1);	% Initial position
repData = [];							% Reproduction data
nbDataRep = nbData + 20;  % Number of time steps for reproduction

% -----> Reconstruction of state sequence
% Slow reconstruction of states sequence based on standard computation of 
% the alpha variable for HSMM. 
h = zeros(model.nbStates,nbDataRep);
c = zeros(nbDataRep,1); %scaling factor to avoid numerical issues
c(1)=1; %Initialization of scaling factor

for t = 1 : nbDataRep   
  repData = [repData currPos];	% Reproduction data
  model.Pd = [];								% Temporary duration prob. model 
  
  for i = 1 : model.nbStates
    % Conditional Gaussian distribution given the external input "u"  
    [model.Mu_Pd(:,i), model.Sigma_Pd(:,:,i), ~] = GMR(model.gmm_Pd(i),...
      u(min(t,nbData)), 1, 2);
    % Maximum duration for HSMM state i
    nbD = exp(model.Mu_Pd(:,i) + 2*model.Sigma_Pd(:,:,i));
    % Pre-computation of duration probabilities 
    model.Pd(i,:) = gaussPDF(log(0.0001:nbData), model.Mu_Pd(:,i), ...
			model.Sigma_Pd(:,:,i)); 
    % The rescaling formula below can be used to guarantee that the cumulated 
    % sum is one (to avoid the numerical issues)
    model.Pd(i,:) = model.Pd(i,:) / sum(model.Pd(i,:));
      
    if t <= nbD
      if(onlyDur)  
        oTmp = 1; %Observation probability for "duration-only HSMM"
      else
	    oTmp = prod(c(1:t) .* gaussPDF(repData(:,1:t), model.Mu(:,i), ...
          model.Sigma(:,:,i))); %Observation probability for standard HSMM
      end
	  h(i,t) = model.StatesPriors(i) * model.Pd(i,t) * oTmp;
      U(t).o(i,1) = oTmp;
    end
    for d=1:min(t-1,nbD)
      if(onlyDur)  
	    oTmp = 1; %Observation probability for "duration-only HSMM"
      else
        oTmp = prod(c(t-d+1:t) .* gaussPDF(repData(:,t-d+1:t), ...
          model.Mu(:,i), model.Sigma(:,:,i))); %Observation prob. for HSMM
      end
	  h(i,t) = h(i,t) + h(:,t-d)' * model.Trans(:,i) * model.Pd(i,d) * oTmp;
      U(t).o(i,d) = oTmp;
    end
  end
  sdP(t).Pd = model.Pd;   % Keep track of the temporary state duration prob.
  c(t+1) = 1/sum(h(:,t)); % Update of scaling factor
  
  % LQR tracking for stepwise reference 
  [~,qList(t)] = max(h(:,t),[],1);
  a.currTar = model.Mu(:,qList(t));        % Reference
  a.currSigma = model.Sigma(:,:,qList(t)); % Current covariance    
  a = reproduction_LQR_infiniteHorizon(model, a, currPos, rFactor);
  
  r(1).refLQR(:,t) = a.currTar;	% Saving step-wise reference for plots
	r(1).ddx(:,t) = a.ddx;				% Saving accelaration computed from LQR
	
	% Emulating that the system stays at the same position when perturbed if
	% the external input is equal to 1
  if(~u(min(t,nbData))) 
    currPos = a.Data;  
  end
end
h = h ./ repmat(sum(h,1),model.nbStates,1);

% Saving .txt files
if(saveTxt)
	model.varnames{1} = 'x1';
	model.varnames{2} = 'x2';
	ADHSMMtoText(model, 'test');
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 18 4.5],'position',[10,10,1300,450],'color',[1 1 1]); 

% Time series plot of the data
for i = 1 : 2
	limAxes = [1, nbData, min(Data(i,:))-1E0, max(Data(i,:))+1E0];
  subplot(2,2,i); hold on; box on;
		msh=[]; x0=[];
		for t=1:nbData-1
			if size(msh,2)==0
				msh(:,1) = [t; model.Mu(i,qList(t))];
			end
			if t==nbData-1 || qList(t+1)~=qList(t)
				% Reference
				msh(:,2) = [t+1; model.Mu(i,qList(t))];
				% Variance
				sTmp = model.Sigma(i,i,qList(t))^.5;
				% Mesh for patch (variance)
				msh2 = [msh(:,1)+[0;sTmp], msh(:,2)+[0;sTmp], msh(:,2)-[0;sTmp],...
					msh(:,1)-[0;sTmp], msh(:,1)+[0;sTmp]];
				patch(msh2(1,:), msh2(2,:), [.85 .85 .85],'edgecolor',[.7 .7 .7]);
				plot(msh(1,:), msh(2,:), '-','linewidth',3,'color',[.7 .7 .7]);
				plot([msh(1,1) msh(1,1)], limAxes(3:4), ':','linewidth',1,...
					'color',[.7 .7 .7]);
				x0 = [x0 msh];
				msh=[];
			end
		end
		% Demonstrations
% 		for n=1:nbSamples
% 		  plot(1:nbData, Data(i,(n-1)*nbData+1:n*nbData), '-', ...
% 				'color', [.6 .6 .6]);
% 		end		
		plot(r(1).refLQR(i,:), '.','lineWidth', 2.5, 'color', [0.5 0.5 0.5]);
		plot(repData(i,:), '-','lineWidth', 2.5, 'color', [0.3 0.3 0.3]);
		
		xlabel(['$t$'],'interpreter','latex','fontsize',18);
		ylabel(['$x_' num2str(i) '$'],'interpreter','latex','fontsize',18);
		axis(limAxes);
end


figure('PaperPosition',[0 0 18 4.5],'position',[10,10,1300,450],'color',[1 1 1]); 
clrmap = lines(model.nbStates);

%Spatial plot of the data
subplot(3,4,[1,5]); hold on;
for i=1:model.nbStates
	plotGMM(model.Mu(1:2,i), model.Sigma(1:2,1:2,i), clrmap(i,:), .6);
end
plot(Data(1,:), Data(2,:), '.', 'color', [.6 .6 .6]);
plot(r(1).refLQR(1,:), r(1).refLQR(2,:), '--','lineWidth', 2.5, 'color', ...
  [0.7 0.3 0.3]);
plot(repData(1,:), repData(2,:), '-','lineWidth', 2.5, 'color', ...
  [0.3 0.3 0.3]);
%xlabel('$x_1$','fontsize',14,'interpreter','latex'); ylabel('$x_2$','fontsize',14,'interpreter','latex');
%axis equal; axis square;
axis tight; axis off;

%Timeline plot of the duration probabilities
h1=[];
for t = 1 : nbData
  delete(h1);    
  %Spatial plot of the data
  subplot(3,4,[1,5]); hold on;
    h1 = plot(r(1).refLQR(1,t), r(1).refLQR(2,t), 'x','lineWidth', 2.5, 'color', ...
      [0.2 0.9 0.2]);
    h1 = [h1 plot(repData(1,t), repData(2,t), 'o','lineWidth', 2.5, 'color', ...
      [0.9 0.6 0.3])];
		h1 = [h1 quiver(repData(1,t), repData(2,t), r(1).ddx(1,t), r(1).ddx(2,t),...
      25, 'LineWidth', 2)];
  
  subplot(3,4,2:4); hold on;    
    for i=1:model.nbStates
			yTmp = sdP(t).Pd(i,:) / max(sdP(t).Pd(i,:));
			h1 = [h1 patch([1, 1:size(yTmp,2), size(yTmp,2)], [0, yTmp, 0],...
				clrmap(i,:), 'EdgeColor', 'none', 'facealpha', .6)];
			h1 = [h1 plot(1:size(yTmp,2), yTmp, 'linewidth', 2, 'color', clrmap(i,:))];
    end
    %set(gca,'xtick',[10:10:nbD],'fontsize',8); 
    axis([1 nbData 0 1]);
    ylabel('$Pd$','fontsize',16,'interpreter','latex');

  %Timeline plot of the state sequence probabilities
  subplot(3,4,6:8); hold on;
    for i=1:model.nbStates
      h1 = [h1 patch([1, 1:t, t], [0, h(i,1:t), 0], clrmap(i,:),...
        'EdgeColor', 'none', 'facealpha', .6)];
      h1 = [h1 plot(1:t, h(i,1:t), 'linewidth', 2, 'color', clrmap(i,:))];
    end
%     h1 = [h1 plot(1:t, u(1:t), 'linewidth', 2, 'color', [0.5 0.5 0.5])];
    set(gca,'xtick',[10:10:nbData],'fontsize',8); axis([1 nbData -0.01 1.01]);
%     xlabel('$t$','fontsize',16,'interpreter','latex'); 
    ylabel('$h$','fontsize',16,'interpreter','latex');
  
  subplot(3,4,10:12); hold on;  
    h1 = [h1 plot(1:t, u(1:t), 'linewidth', 2, 'color', [0.5 0.5 0.5])];
		set(gca,'xtick',[10:10:nbData],'fontsize',8); axis([1 nbData -0.01 1.01]);
    xlabel('$t$','fontsize',16,'interpreter','latex'); 
    ylabel('$u$','fontsize',16,'interpreter','latex');
    
    pause(0.1);
%     pause;  
end
