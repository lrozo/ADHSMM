function [PHI,PHI1,PHI0] = constructPHI(model,nbData,nbSamples)
%Construct PHI operator (big sparse matrix) used in trajectory-GMM, see Eq. (2.4.5) in doc/TechnicalReport.pdf
%Sylvain Calinon, 2015

op1D = zeros(model.nbDeriv);
op1D(1,end) = 1;
for i=2:model.nbDeriv
	op1D(i,:) = (op1D(i-1,:) - circshift(op1D(i-1,:),[0,-1])) / model.dt;
end
op = zeros(nbData*model.nbDeriv, nbData);
op((model.nbDeriv-1)*model.nbDeriv+1:model.nbDeriv*model.nbDeriv, 1:model.nbDeriv) = op1D;
PHI0 = zeros(nbData*model.nbDeriv, nbData);
for t=0:nbData-model.nbDeriv
	PHI0 = PHI0 + circshift(op, [model.nbDeriv*t,t]);
end
%Handling of borders
for i=1:model.nbDeriv-1
	op(model.nbDeriv*model.nbDeriv+1-i,:)=0; op(:,i)=0;
	PHI0 = PHI0 + circshift(op, [-i*model.nbDeriv,-i]);
end
%Application to multiple dimensions and multiple demonstrations
PHI1 = kron(PHI0, eye(model.nbVarPos));
PHI = kron(eye(nbSamples), PHI1);

% PHI0
% pcolor([PHI0 zeros(size(PHI0,1),1); zeros(1,size(PHI0,2)+1)]); %dummy values for correct display
% shading flat; axis ij; axis equal tight;
% pause;
% close all;
% return