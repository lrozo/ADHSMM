function demo_trajADHSMMlogLQR_online01

% Leonel Rozo, 2016
%
% Example of an LQR-based reproduction for an online trajectory retrieval 
% method built on an ADHSMM and trajectory GMM.
% 
% - A time window is defined in order to carry out the optimization process
%		of the trajectory GMM for a specific number of steps, given a state
%		sequence.
% - The ADHSMM model adapts the log-normal state duration probability
%		according to an external input "u". 
% - Every ADHSMM state has a duration probability represented by a mixture
%		of Gaussians. 
% - A conditional Gaussian distribution is obtained at each time step by
%   applying GMR given the external input "u".
% - Infinite LQR (with position and velocity references) is used, under the
%   assumption that there is not a specific finite horizon for the task.  
%
% This code:
%		1. Sets the variable values for the ADHSMM , the trajectory retrieval 
%			 model (trajGMM), and the linear quadratic regulator (LQR). 
%		2. Loads synthetic data to be used for training the model. The data
%			 correspond to several G-shape trajectories in 2D.
%		3. Trains the model in two phases: (i) an HSMM is trained, (ii)
%			 duration probabilities (GMM) for each model state are set manually
%			 (these can be easily learned from EM for a Gaussian mixture model).
%		4. Reconstructs, in an online fashion, a state sequence where the state 
%			 duration depends on the given external input.
%		5. Retrieves a reference trajectory from a weighted least squares
%		   approach that uses the means and covariance matrices from the state 
%			 sequence previously computed.
%		6. Plots the results in dynamic graphs.
%


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -----> Model variables (TrajectoryADHSMM)
model.nbStates = 7; % Number of components in the model
model.nbVarPos = 2; % Dimension of position data (here: x1,x2)
model.nbDeriv  = 2; % Number of static&dynamic features 
                    % (D=2 for [x,dx], D=3 for [x,dx,ddx], etc.)
model.dt       = 1; % Time step (large values such as 1 will tend to create 
                    % clusers by following position information)
model.nbVar    = model.nbVarPos * model.nbDeriv;
nbSamples      = 4; % Number of trajectory samples
nbD            = 200; % Number of datapoints in a trajectory
ctrdWndw       = 1;   % Uses a centered window for the online implementation
Tw             = 15;  % Time length for centered window implementation

% -----> LQR variables
% rFactor        = 10E1; % R factor for infinite LQR
rFactor        = 2E1; % R factor for infinite LQR

% Construct operator PHI (big sparse matrix), 
% see Eq. (2.4.5) in doc/TechnicalReport.pdf
[~,PHI1,PHI0] = constructPHI(model, nbD, nbSamples); 
T1 = nbD;


%% Load dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/GShape.mat');

Data = [];
for n = 1 : nbSamples
  % Resampling  
  s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, ...
    linspace(1,size(demos{n}.pos,2),nbD)); 
      
  % Re-arrange data in vector form for computing derivatives
  x = reshape(s(n).Data, numel(s(n).Data), 1);
  zeta = PHI1 * x;
  dataTmp = reshape(zeta, model.nbVarPos*model.nbDeriv, nbD);
  s(n).Data = dataTmp;
  Data = [Data dataTmp];  
end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Learning');
% -----> K-means initialization
% model = init_GMM_kmeans(Data, model);
model = init_GMM_timeBased([repmat(1:nbD,1,nbSamples); Data], model);
% Recovering model without artificial time dimension
model.Mu = model.Mu(2:end,:);
model.Sigma = model.Sigma(2:end,2:end,:);

% -----> Transition matrix initialization
% Random initialization for transition matrix
model.Trans = rand(model.nbStates,model.nbStates);
model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);
model.StatesPriors = rand(model.nbStates,1);
model.StatesPriors = model.StatesPriors/sum(model.StatesPriors);

% %Left-right model initialization
% model.Trans = zeros(model.nbStates);
% for i=1:model.nbStates-1
% 	model.Trans(i,i) = 1-(model.nbStates/nbD);
% 	model.Trans(i,i+1) = model.nbStates/nbD;
% end
% model.Trans(model.nbStates,model.nbStates) = 1.0;
% model.StatesPriors = zeros(model.nbStates,1);
% model.StatesPriors(1) = 1;
% model.Priors = ones(model.nbStates,1);

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


%% Online reproduction with a time window 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Iterative reproduction');

% -----> Initial conditions for reference trajectory retrieval
currPos = s(2).Data(1:model.nbVarPos,1);	% Initial position
currVel = zeros(2,1);											% Initial velocity
repData = [];															% Reproduction data
nbDataRep = T1 + 40;											% Time steps for reproduction

% -----> Designing external input
% % Artificial external (discrete) input
% r(1).u = zeros(nbD,1);
% r(1).u = ones(nbD,1);
r(1).u = [zeros(50,1) ; ones(45,1) ; zeros(105,1)];
% r(1).u = [zeros(45,1) ; ones(10,1) ; zeros(145,1)];
% r(1).u = [ones(40,1) ; zeros(80, 1); ones(80,1)];
% r(1).u = [zeros(50,1) ; ones(50,1) ; zeros(45,1) ; ones(35,1) ; zeros(20,1)];

% -----> Initializing variables for reproduction
h = zeros(model.nbStates,nbDataRep);
c = zeros(nbDataRep,1); %scaling factor to avoid numerical issues
c(1) = 1; % Initialization of scaling factor
% Compute PHI operator for current time window size
if(ctrdWndw)
	PHIw = kron(PHI0(1:(2*Tw+1)*model.nbDeriv, 1:(2*Tw+1)),...
		eye(model.nbVarPos));
else
%     PHIw = kron(PHI0(1:(T1-t+1)*model.nbDeriv, 1:T1-t+1), ...
%       eye(model.nbVarPos)); %Operator for the time window [t,T1]
	PHIw = kron(PHI0(1:(2*Tw)*model.nbDeriv, 1:2*Tw), ...
		eye(model.nbVarPos)); %Operator for the time window [t,T1]
end
	
% -----> Online Reproduction loop
tic
for t = 1 : nbDataRep
  fprintf('.');
  
	% -----> Reconstruction of state sequence from the forward variable
  model.Pd = []; % Reset temporary duration prob. model 
  repData = [repData currPos]; % Reproduction position data
	   
  % % Compute current weight (influenced by position data)
  for i=1:model.nbStates
    % Conditional Gaussian distribution given the external input "u"  
    [model.Mu_Pd(:,i), model.Sigma_Pd(:,:,i), ~] = GMR(model.gmm_Pd(i),...
      r(1).u(min(t,T1)), 1, 2);
    % Setting maximum duration
    % Three standard deviations from the mean account for 99% of the data
    model.nbPD(i) = exp(model.Mu_Pd(:,i) + 3*sqrt(model.Sigma_Pd(:,:,i)));
    % Computation of duration probabilities 
    model.Pd(i,:) = gaussPDF(log(0.0001:model.nbPD(i)), model.Mu_Pd(:,i),...
			model.Sigma_Pd(:,:,i)); 
    % The rescaling formula below can be used to guarantee that the cumulated sum is one (to avoid the numerical issues)
%     model.Pd(i,:) = model.Pd(i,:) / sum(model.Pd(i,:));

    if t<=model.nbPD(i)
% 	  oTmp = 1; %Observation probability for "duration-only HSMM"
	  oTmp = prod(c(1:t) .* gaussPDF(repData(:,1:t), ...
        model.Mu(1:model.nbVarPos,i),...
        model.Sigma(1:model.nbVarPos,1:model.nbVarPos,i))); % Observation probability for standard HSMM
	  h(i,t) = model.StatesPriors(i) * model.Pd(i,t) * oTmp;
		end
		for d=1:min(t-1,model.nbPD(i))
% 	  oTmp = 1; %Observation probability for "duration-only HSMM"
			oTmp = prod(c(t-d+1:t) .* gaussPDF(repData(:,t-d+1:t), ...
        model.Mu(1:model.nbVarPos,i), ...
        model.Sigma(1:model.nbVarPos,1:model.nbVarPos,i))); %Observation probability for standard HSMM	
			h(i,t) = h(i,t) + h(:,t-d)' * model.Trans(:,i) * model.Pd(i,d) * oTmp;
		end
  end
  c(t+1) = 1 / sum(h(:,t)); %Update of scaling factor
  sdP(t).Pd = model.Pd; % Keep track of the temporary state duration prob.
	
  
	% -----> Trajectory retrieval for time window
  if(ctrdWndw) % Centered time window [t-Tw,t+Tw]
    % Saving "Tw" weights based on previous observations
    H = zeros(model.nbStates, 2*Tw+1);
    cnt = 0;
    for tt = min(t, Tw+1) : -1 : 1
      H(:,tt) = h(:,t-cnt);  
      cnt = cnt+1;
    end
    % % Predict future weights (not influenced by position data) 
    for tt = min(t, Tw+1)+1 : 2*Tw+1
      for i = 1 : model.nbStates
%         if tt <= model.nbPD(i)
%           H(i,tt) = model.StatesPriors(i) * model.Pd(i,tt);
%         end
        for d = 1 : min(tt-1, model.nbPD(i))
		  H(i,tt) = H(i,tt) + H(:,tt-d)' * model.Trans(:,i) * model.Pd(i,d);
        end
      end
    end  
      
  else % Time-window (either [t,T1] or [t,2*Tw])
	% Predict future weights (not influenced by position data)
% 	H = zeros(model.nbStates, T1-t+1);
    H = zeros(model.nbStates, 2*Tw);
	H(:,1) = h(:,t);
% 	for tt = 2 : T1-t+1
    for tt = 2 : 2*Tw
  	  for i=1:model.nbStates
%         if tt<=nbPD
%     	  H(i,tt) = model.StatesPriors(i) * model.Pd(i,tt);
%         end
				for d=1:min(tt-1,model.nbPD(i))
					H(i,tt) = H(i,tt) + H(:,tt-d)' * model.Trans(:,i) * model.Pd(i,d);
				end
      end
    end	  
  end
  H = H ./ repmat(sum(H,1),model.nbStates,1);
  r(1).s(t).H = H;
	
  
  % % Compute state path
  [~,q] = max(H,[],1); %works also for nbStates=1
  % Saving vector of centers \mu and matrix of covariances \Sigma
  MuQ = zeros(length(q)*model.nbVar, 1); 
  SigmaQ = zeros(length(q)*model.nbVar, length(q)*model.nbVar); 
  for tt = 1 : length(q)
	id1 = (tt-1)*model.nbVar+1:tt*model.nbVar;
	MuQ(id1,1) = model.Mu(:,q(tt));
	SigmaQ(id1,id1) = model.Sigma(:,:,q(tt));
  end
  
  
  % % Reconstruction for the time window 
  % Retrieval of data with weighted least squares solution
  [zeta, ~, ~, Scov] = lscov(PHIw, MuQ, SigmaQ, 'chol'); 
  
  if(ctrdWndw)
    r(1).s(t).Data = reshape(zeta, model.nbVarPos, 2*Tw+1);
    r(1).Data(:,t) = r(1).s(t).Data(:, min(t, Tw+1));
    
    % Rebuild covariance by reshaping Scov, Eq. (2.4.12) in doc/TechnicalReport.pdf
    for tt = 1 : 2*Tw+1
      id = (tt-1)*model.nbVarPos+1 : tt*model.nbVarPos;
      r(1).s(t).expSigma(:,:,tt) = Scov(id,id) * (2*Tw+1);
    end
    r(1).expSigma(:,:,t) = r(1).s(t).expSigma(:,:,min(t, Tw+1));
  else
%     r(1).s(t).Data = reshape(zeta, model.nbVarPos, T1-t+1);
    r(1).s(t).Data = reshape(zeta, model.nbVarPos, 2*Tw);    
    r(1).Data(:,t) = r(1).s(t).Data(:,1);  
    
    % Rebuild covariance by reshaping Scov, Eq. (2.4.12) in doc/TechnicalReport.pdf
%     for tt = 1 : T1-t+1
    for tt = 1 : 2*Tw
      id = (tt-1)*model.nbVarPos+1 : tt*model.nbVarPos;
      r(1).s(t).expSigma(:,:,tt) = Scov(id,id) * (2*Tw);
    end
    r(1).expSigma(:,:,t) = r(1).s(t).expSigma(:,:,1);
  end      
  
  
  % -----> Infinite LQR-based reproduction
  % % Using only position data
%   rr.currTar = r(1).Data(:,t);
%   rr.currSigma = r(1).expSigma(:,:,t);
%   [rr,~] = reproduction_LQR_infiniteHorizon(model, rr, currPos, rFactor);

  % % Using position and velocity data (NOT desired velocity)
  rr.currTar = [r(1).Data(:,t) ; zeros(model.nbVarPos,1)];
  rr.currSigma = [r(1).expSigma(:,:,t) zeros(model.nbVarPos,model.nbVarPos);...
    zeros(model.nbVarPos,model.nbVarPos) 100*eye(model.nbVarPos)];
  [rr,~] = reproduction_LQR_infiniteHorizon_withVel(model, rr, ...
    [currPos ; currVel], rFactor);
  
  if(~r(1).u(min(t,T1))) % Emulating that the system stays at the same position when perturbed.
    currPos = rr.Data;
    currVel = rr.dx;
  else
    currVel = zeros(2,1);  
  end
end
toc
h = h ./ repmat(sum(h,1),model.nbStates,1);
fprintf('\n');


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 18 4.5],'position',[10,10,1300,450],'color',[1 1 1]); 

% Time series plot of the data
for i = 1 : 2
	limAxes = [1, nbDataRep, min(Data(i,:))-1E0, max(Data(i,:))+1E0];
  subplot(2,2,i); hold on; box on;
		msh = [1:nbDataRep, nbDataRep:-1:1; ...
			r(1).Data(i,:)-squeeze(r(1).expSigma(i,i,:).^.5)'*1, ...
			fliplr(r(1).Data(i,:)+squeeze(r(1).expSigma(i,i,:).^.5)'*1)];
		patch(msh(1,:), msh(2,:), [.85 .85 .85],'edgecolor',[.7 .7 .7],...
			'edgealpha',.8,'facealpha',.5);
		plot(r(1).Data(i,:), '-','lineWidth', 2.5, 'color', [0.5 0.5 0.5]);
		plot(repData(i,:), '-','lineWidth', 2.5, 'color', [0.3 0.3 0.3]);
		
		ylabel(['$x_' num2str(i) '$'],'interpreter','latex','fontsize',18);
		axis(limAxes);
end


figure('PaperPosition',[0 0 18 4.5],'position',[10,10,1300,450],'color',[1 1 1]); 
%xx = round(linspace(1,64,model.nbStates));
%clrmap = colormap('jet')*0.5;
%clrmap = min(clrmap(xx,:),.9);
clrmap = lines(model.nbStates);

%Spatial plot of the data
subplot(3,4,[1,5]); hold on;
for i=1:model.nbStates
	plotGMM(model.Mu(1:2,i), model.Sigma(1:2,1:2,i), clrmap(i,:), .6);
end
plot(Data(1,:), Data(2,:), '.', 'color', [.6 .6 .6]);
plot(r(1).Data(1,:), r(1).Data(2,:), '--','lineWidth', 2.5, 'color', ...
  [0.3 0.3 0.3]);
plot(repData(1,:), repData(2,:), '-','lineWidth', 2.5, 'color', ...
  [0.2 0.2 0.8]);
% plot(r(1).xLQR(1,:), r(1).xLQR(2,:), '-','lineWidth', 2.5, 'color', ...
%   [0.3 0.3 0.3]);
%xlabel('$x_1$','fontsize',14,'interpreter','latex'); ylabel('$x_2$','fontsize',14,'interpreter','latex');
%axis equal; axis square;
axis tight; axis off;

%Timeline plot of the duration probabilities
h1=[];
for t = 1 : nbDataRep
  delete(h1);    
  %Spatial plot of the data
  subplot(3,4,[1,5]); hold on;
%     h1 = plot(r(1).xLQR(1,t), r(1).xLQR(2,t), 'o','lineWidth', 2.5, 'color', ...
%       [0.9 0.6 0.3]);
    h1 = plot(r(1).Data(1,t), r(1).Data(2,t), 'x','lineWidth', 2.5, 'color', ...
      [0.2 0.9 0.2]);
    h1 = [h1 plot(repData(1,t), repData(2,t), 'o','lineWidth', 2.5, 'color', ...
      [0.9 0.6 0.3])];
    h1 = [h1 plotGMM(r(1).Data(1:2,t), r(1).expSigma(:,:,t), [0.2 0.9 0.2], .5)];
    h1 = [h1 plot(r(1).s(t).Data(1,:), r(1).s(t).Data(2,:),...
      '-','lineWidth',2.5,'color',[.8 0 0])];
  
  subplot(3,4,2:4); hold on;    
    for i=1:model.nbStates
	  yTmp = sdP(t).Pd(i,:) / max(sdP(t).Pd(i,:));
	  h1 = [h1 patch([1, 1:size(yTmp,2), size(yTmp,2)], ...
        [0, yTmp, 0], clrmap(i,:), 'EdgeColor',...
        'none', 'facealpha', .6)];
	  h1 = [h1 plot(1:size(yTmp,2), yTmp, 'linewidth', 2, 'color', clrmap(i,:))];
    end
    %set(gca,'xtick',[10:10:nbD],'fontsize',8); 
    axis([1 nbDataRep 0 1]);
    ylabel('$Pd$','fontsize',16,'interpreter','latex');

  %Timeline plot of the state sequence probabilities
  subplot(3,4,6:8); hold on;
    if(ctrdWndw || t < nbDataRep)
      for i=1:model.nbStates
        h1 = [h1 patch([1, 1:size(r(1).s(t).H,2), size(r(1).s(t).H,2)], ...
          [0, r(1).s(t).H(i,:), 0], clrmap(i,:), 'EdgeColor', ...
          'none', 'facealpha', .6)];
        h1 = [h1 plot(1:size(r(1).s(t).H,2), r(1).s(t).H(i,:), ...
          'linewidth', 2, 'color', clrmap(i,:))];
      end
      if(ctrdWndw)
        h1 = [h1 plot([Tw+1 Tw+1], [-1 1], '--', 'linewidth', 1, ...
          'color', [0.5 0.5 0.5])];    
      else
        h1 = [h1 plot([1 1], [-1 1], '--', 'linewidth', 1, ...
          'color', [0.5 0.5 0.5])];  
      end
%       set(gca,'xtick',[10:10:size(r(1).s(t).H,2)],'fontsize',8); 
      axis([1 size(r(1).s(t).H,2) -0.01 1.01]);
%     xlabel('$t$','fontsize',16,'interpreter','latex'); 
      ylabel('$H$','fontsize',16,'interpreter','latex');
    end
  
  subplot(3,4,10:12); hold on;  
    for i=1:model.nbStates
      h1 = [h1 patch([1, 1:t, t], [0, h(i,1:t), 0], clrmap(i,:),...
        'EdgeColor', 'none', 'facealpha', .6)];
      h1 = [h1 plot(1:t, h(i,1:t), 'linewidth', 2, 'color', clrmap(i,:))];
    end
    h1 = [h1 plot(1:min(t,T1), r(1).u(1:min(t,T1)), 'linewidth', 2,...
			'color', [0.5 0.5 0.5])];
		set(gca,'xtick',[10:10:nbDataRep],'fontsize',8); axis([1 nbDataRep -0.01 1.01]);
    xlabel('$t$','fontsize',16,'interpreter','latex'); 
    ylabel('$h$','fontsize',16,'interpreter','latex');
    
  pause(0.1);
%   pause;  
end
