% Fit Interaction predictors to Facebook wall post data as
% by leveraging the firnedship data
% Author: Ruthwik Junuthula & Kevin S. Xu
includeCc = true;
directed = true;
% Parameters for unknown classes
kPost = 4;                    %no : of classes
pPost = kPost^2;              %size of the probability matrix
initCovPost = 100*eye(pPost); %initialize the covariance matrix
stateCovInPost = 0.1;   
stateCovOutPost = 0.02;
% Optional parameters
Opt.directed = directed;     
% Number of random initializations for k-means step of spectral clustering
Opt.nKmeansReps = 5;
Opt.maxIter = 150;           % Maximum number of local search iterations
Opt.output = 1;              % Level of output to display in console
%% Load data
load('FacebookFilteredAdj_90Days_120OutInDegfrnd_0OutInDeg.mat')

if directed == false
	str1 = 'Undirected';
else
	str1 = 'Directed';
end
%% Pre-processing
[n,~,tMax] = size(adj);
logistic = @(x) 1./(1+exp(-x));	% Logistic function, where 'x' is the state
%% Estimate states using EKF with classes estimated by local search
disp('Estimating states using EKF with a posteriori class estimates')
stateTransPost = eye(pPost);    %state trasition matrix (F)
stateCovPost = generateStateCov(kPost,stateCovInPost,stateCovOutPost,...
               directed);       % covariance matrix of the process noise
OptPostEkf = Opt;
OptPostEkf.nClasses = kPost;
[classPostEkf,psiPostEkf,psiCovPostEkf,~,OptPostEkf] = ekfDsbmLocalSearch ...
   (adj,frndadj,kPost,stateTransPost,stateCovPost,[],[],initCovPost,OptPostEkf);
 
thetaPostEkf = logistic(psiPostEkf);
blockDensPost = calcBlockDens(adj,classPostEkf,OptPostEkf);
psiPredPostEkf = stateTransPost*psiPostEkf;
thetaPredPostEkf = [zeros(pPost,1) logistic(psiPredPostEkf(:,1:tMax-1))];
msePredPostEkf = sum((thetaPredPostEkf(:,2:tMax) - blockDensPost(:,2:tMax)).^2);
%% Calculate forecast error of links
disp('Calculating ROC of link forecast')

% Exponentially-weighted moving average (EWMA) predictor
predMatEwma = zeros(n,n,tMax);
ff = 0.5;   % Forgetting factor for EWMA
ccWt = 0.01;   % Weight of EKF predictor in convex combination

%% Katz and AA on Firendship network

frndProbMatKatz = zeros(n,n,tMax);
frndProbMatAA = zeros(n,n,tMax);
parfor i=1:tMax-1
    frndProbMatKatz(:,:,i+1) = predictLinksKatz(frndadj(:,:,i),0.005,10);
end

parfor i=1:tMax-1
    frndProbMatAA(:,:,i+1) = predictLinksAA(frndadj(:,:,i));
end

%% Katz and AA on interaction network
 predMatKatz = zeros(n,n,tMax);
 predMatAA   = zeros(n,n,tMax);
parfor i=1:tMax-1
    predMatKatz(:,:,i+1) = predictLinksKatz(adj(:,:,i),0.1,7);
	predMatAA(:,:,i+1) = predictLinksAA(adj(:,:,i));
	predMatAA(:,:,i+1) = predictLinksAA(adj(:,:,i))/max(max(predMatAA(:,:,i+1)));
	predMatKatz(:,:,i+1) = predMatKatz(:,:,i+1)/max(max(predMatKatz(:,:,i+1)));
end
%% Link forecast based on a posteriori class estimates
predMatPostEkf = predAdjMatDsbm(adj,blockvec2mat(thetaPredPostEkf, ...
                         directed),classPostEkf);

predMatEwma(:,:,2) = adj(:,:,2);
for t = 3:tMax
    predMatEwma(:,:,t) = ff*predMatEwma(:,:,t-1) + (1-ff)*adj(:,:,t-1);
end

%%interaction predictors (TS-AA & TS-Katz)
predMatTsKatz=0.5*predMatKatz+0.5*predMatEwma;
predMatTsAA = 0.5*predMatAA+0.5*predMatEwma;




%% Friendship predictor
% Katz
predMatEwmaKatz = 0.5*predMatEwma + 0.5*frndProbMatKatz./max(max(frndProbMatKatz));%past only
predMatKatz = frndProbMatKatz./max(max(frndProbMatKatz)); %No
predMatTsKatzKatz = 0.5*frndProbMatKatz./max(max(frndProbMatKatz))+0.5*predMatTsKatz;%predicted
predMatTsAAKatz = 0.5*frndProbMatKatz./max(max(frndProbMatKatz))+0.5*predMatTsAA;%predicted

%% AA
predMatEwmaAA = 0.5*predMatEwma + 0.5*frndProbMatAA./max(max(frndProbMatAA));%past only
predMatPostEkfAA = 0.2*frndProbMatAA./max(max(frndProbMatAA))+0.8*predMatPostEkf; 
predMatPostEkfEwmaAA = 0.7*predMatPostEkfEwma + 0.3*frndProbMatAA./max(max(frndProbMatAA));
predMatAA = frndProbMatAA./max(max(frndProbMatAA)); %No
predMatTsAAAA =  0.5*frndProbMatAA./max(max(frndProbMatAA))+ 0.5*predMatTsAA;%predicted
predMatTsKatzAA =  0.5*frndProbMatAA./max(max(frndProbMatAA))+0.5*predMatTsKatz; %predicted
%% 

[recEwma,precEwma,praucEwma,~,~,maxF1Ewma] = dlpPRCurve(frndadj(:,:,2:end), ...
                                    predMatEwma(:,:,2:end),'new',true);
[recPostEkfEwma,precPostEkfEwma,praucPostEkfEwma,~,~,maxF1PostEkfEwma] = ...
             dlpPRCurve(frndadj(:,:,2:end),predMatPostEkfEwma(:,:,2:end),'new',true);
                                                        
%% Pr new friendships
% Pr new Katz                                                       
[recEwmaKatz,precEwmaKatz,praucEwmaKatz] = ...
       dlpPRCurve(frndadj(:,:,2:end),predMatEwmaKatz(:,:,2:end),'new',true);

[recTsKatzKatz,precTsKatzKatz,praucTsKatzKatz] = ...
      dlpPRCurve(frndadj(:,:,2:end),predMatTsKatzKatz(:,:,2:end),'new',true);
  
[recTsAAKatz,precTsAAKatz,praucTsAAKatz] = ...
      dlpPRCurve(frndadj(:,:,2:end),predMatTsAAKatz(:,:,2:end),'new',true);
  
[recEwmaKatz,precEwmaKatz,praucKatz] = ...
          dlpPRCurve(frndadj(:,:,2:end),predMatKatz(:,:,2:end),'new',true);
% AUC new katz
[fprEwmaKatz,tprEwmaKatz,~,aucEwmaKatz] = ...
       dlpROCCurve(frndadj(:,:,2:end),predMatEwmaKatz(:,:,2:end),'new',true);

[fprTsKatzKatz,tprTsKatzKatz,~,aucTsKatzKatz] = ...
      dlpROCCurve(frndadj(:,:,2:end),predMatTsKatzKatz(:,:,2:end),'new',true);
  
[fprPostEkfEwmaKatz,tprTsAAKatz,~,aucTsAAKatz] = ...
      dlpROCCurve(frndadj(:,:,2:end),predMatTsAAKatz(:,:,2:end),'new',true);
  
[fprEwmaKatz,tprEwmaKatz,~,aucKatz] = ...
          dlpROCCurve(frndadj(:,:,2:end),predMatKatz(:,:,2:end),'new',true);
														
%% Pr new AA
[recEwmaAA,precEwmaAA,praucEwmaAA] = dlpPRCurve(frndadj(:,:,2:end),predMatEwmaAA(:,:,2:end),'new',true);
[recTsAAAA,precTsAAAA,praucTsAAAA] = dlpPRCurve(frndadj(:,:,2:end),predMatTsAAAA(:,:,2:end),'new',true);
[recTsKatzAA,precTsKatzAA,praucTsKatzAA] = ...
                          dlpPRCurve(frndadj(:,:,2:end),predMatTsKatzAA(:,:,2:end),'new',true);
[recEwmaAA,precEwmaAA,praucAA] = ...
                          dlpPRCurve(frndadj(:,:,2:end),predMatAA(:,:,2:end),'new',true);
%% AUC new AA                                                    
[fprEwmaAA,tprEwmaAA,~,aucEwmaAA] = dlpROCCurve(frndadj(:,:,2:end),predMatEwmaAA(:,:,2:end),'new',true);
[fprTsAAAA,tprTsAAAA,~,aucTsAAAA] = ...
                 dlpROCCurve(frndadj(:,:,2:end),predMatTsAAAA(:,:,2:end),'new',true);
[fprTsKatzAA,tprTsKatzAA,~,aucTsKatzAA] = ...
                 dlpROCCurve(frndadj(:,:,2:end),predMatTsKatzAA(:,:,2:end),'new',true);
[fprEwmaAA,tprEwmaAA,~,aucAA] = dlpROCCurve(frndadj(:,:,2:end),predMatAA(:,:,2:end),'new',true);

% Pr new interactions                                                    
[recEwmaAll,precEwmaAll,praucEwmaAll,~,~,maxF1EwmaAll] = dlpPRCurve(adj(:,:,2:end),predMatEwma(:,:,2:end),'all',true);
[recPostEkfAll,precPostEkfAll,praucPostEkfAll,~,~,maxF1PostEkfAll] = ... 
                                                                dlpPRCurve(adj(:,:,2:end),predMatEwmaKatz(:,:,2:end),'all',true);
[recPostEkfEwmaAll,precPostEkfEwmaAll,praucPostEkfEwmaAll,~,~,maxF1PostEkfEwmaAll] = ...
                                                            dlpPRCurve(adj(:,:,2:end),predMatPostEkfEwma(:,:,2:end),'all',true);
[recPostEkffrndAll,precPostEkffrndAll,praucPostEkffrndAll,~,~,maxF1PostEkffrndAll] = ...
                                                            dlpPRCurve(adj(:,:,2:end),predMatPostEkffrnd(:,:,2:end),'all',true);
[recPostEkfEwmafrndAll,precPostEkfEwmafrndAll,praucPostEkfEwmafrndAll,~,~,maxF1PostEkfEwmafrndAll] = ...
                                                        dlpPRCurve(adj(:,:,2:end),predMatfrnd(:,:,2:end),'all',true);
[recEwmafrndAll,precEwmafrndAll,praucEwmafrndAll,~,~,maxF1EwmafrndAll] = ...
                                                        dlpPRCurve(adj(:,:,2:end),predMatEwmafrnd(:,:,2:end),'all',true);
														
[fprEwma,tprEwma,~,aucEwma] = dlpROCCurve(adj(:,:,2:end),predMatEwma(:,:,2:end),'existing',true);
[fprPostEkf,tprPostEkf,~,aucPostEkf] = dlpROCCurve(adj(:,:,2:end),predMatEwmaKatz(:,:,2:end),'existing',true);
[fprPostEkfEwma,tprPostEkfEwma,~,aucPostEkfEwma] = dlpROCCurve(adj(:,:,2:end),predMatPostEkfEwma(:,:,2:end),'existing',true);
[fprPostEkffrnd,tprPostEkffrnd,~,aucPostEkffrnd] = dlpROCCurve(adj(:,:,2:end),predMatPostEkffrnd(:,:,2:end),'existing',true);
[fprPostEkfEwmafrnd,tprPostEkfEwmafrnd,~,aucPostEkfEwmafrnd] = ...
                                                        dlpROCCurve(adj(:,:,2:end),predMatfrnd(:,:,2:end),'existing',true);
[fprEwmafrnd,tprEwmafrnd,~,aucEwmafrnd] = dlpROCCurve(adj(:,:,2:end),predMatEwmafrnd(:,:,2:end),'existing',true);													
														
														
[fprEwmaAll,tprEwmaAll,~,aucEwmaAll] = dlpROCCurve(adj(:,:,2:end),predMatEwma(:,:,2:end),'all',true);
[fprPostEkfAll,tprPostEkfAll,~,aucPostEkfAll] = dlpROCCurve(adj(:,:,2:end),predMatEwmaKatz(:,:,2:end),'all',true);
[fprPostEkfEwmaAll,tprPostEkfEwmaAll,~,aucPostEkfEwmaAll]= dlpROCCurve(adj(:,:,2:end),predMatPostEkfEwma(:,:,2:end),'all',true);
[fprPostEkffrndAll,tprPostEkffrndAll,~,aucPostEkffrndAll]= dlpROCCurve(adj(:,:,2:end),predMatPostEkffrnd(:,:,2:end),'all',true);
[fprPostEkfEwmafrndAll,tprPostEkfEwmafrndAll,~,aucPostEkfEwmafrndAll] = ...
                                                        dlpROCCurve(adj(:,:,2:end),predMatfrnd(:,:,2:end),'all',true);
[fprEwmafrndAll,tprEwmafrndAll,~,aucEwmafrndAll] = ...
                                                        dlpROCCurve(adj(:,:,2:end),predMatEwmafrnd(:,:,2:end),'all',true);														
														
unifiedMetricEwma = unifiedDlpMetric(praucEwma,aucEwma,adj(:,:,2:end),true)
unifiedMetricPostEkf = unifiedDlpMetric(praucPostEkf,aucPostEkf,adj(:,:,2:end),true)
unifiedMetricPostEkfEwma = unifiedDlpMetric(praucPostEkfEwma,aucPostEkfEwma,adj(:,:,2:end),true)
unifiedMetricPostEKffrnd = unifiedDlpMetric(praucPostEkffrnd,aucPostEkffrnd,adj(:,:,2:end),true)
unifiedMetricPostEkfEwmafrnd = unifiedDlpMetric(praucPostEkfEwmafrnd,aucPostEkfEwmafrnd,adj(:,:,2:end),true)	


% Plot ROC curves
figure
figure(1)
plot(fprPostEkf,tprPostEkf,fprEwma,tprEwma,fprPostEkfEwma,tprPostEkfEwma,fprPostEkffrnd,tprPostEkffrnd,fprPostEkfEwmafrnd,tprPostEkfEwmafrnd)
legend('A post Ekf','EWMA','A posteriori EKF + EWMA','PostEkf+frnd','postEkf+Ewma+frnd','Location','Best')
xlabel('Rec')
ylabel('prec')
title('ROC curves for link forecasts')
figure(2)
plot(recPostEkf,precPostEkf,recEwma,precEwma,recPostEkfEwma,precPostEkfEwma,recPostEkffrnd,precPostEkffrnd,recPostEkfEwmafrnd,precPostEkfEwmafrnd)
legend('A ','EWMA','TS-Katz','PostEkf+frnd','postEkf+Ewma+frnd','EWMA+frnd','Location','Best')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC curves for link forecasts')
