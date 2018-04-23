% Fit interaction predictors to Facebook wall post data as
% by leveraging the friendship data
% Author: Ruthwik Junuthula & Kevin S. Xu

directed = true;
minDegfrnd = 120;   % Minimum degree in original friendship network
minOutInDeg = 0;    % Minimum in- or out-degree in original interaction network
datapath = ''; 
ResultsPath = '';

% Parameters for dynamic stochastic block model
kPost = 4;                    %no : of classes
pPost = kPost^2;              %size of the probability matrix
initCovPost = 100*eye(pPost); %initialize the covariance matrix
stateCovInPost = 0.1;   
stateCovOutPost = 0.02;

% Optional parameters for dynamic stochastic block model
Opt.directed = directed; 

Opt.nKmeansReps = 5;       % Number of random initializations for k-means
                           % step of spectral clustering
Opt.maxIter = 1000;        % Maximum number of local search iterations
Opt.output = 2;            % Level of output to display in console

%% Load data
load([datapath 'FacebookFilteredAdj_90Days_' int2str(minDegfrnd) ...
            'Degfrnd_' int2str(minOutInDeg) 'OutInDeg.mat']);

%% Pre-processing
[n,~,tMax] = size(adj);
logistic = @(x) 1./(1+exp(-x));	% Logistic function, where 'x' is the state
% Estimate states using EKF with classes estimated by local search
disp('Estimating states using EKF with a posteriori class estimates')
stateTransPost = eye(pPost);    %state trasition matrix (F)
stateCovPost = generateStateCov(kPost,stateCovInPost,stateCovOutPost,...
               directed);       % covariance matrix of the process noise
OptPostEkf = Opt;
OptPostEkf.nClasses = kPost;
[classPostEkf,psiPostEkf,psiCovPostEkf,~,OptPostEkf] = ekfDsbmLocalSearch ...
  (adj,kPost,stateTransPost,stateCovPost,[],[],initCovPost,OptPostEkf);
 
thetaPostEkf = logistic(psiPostEkf);
blockDensPost = calcBlockDens(adj,classPostEkf,OptPostEkf);
psiPredPostEkf = stateTransPost*psiPostEkf;
thetaPredPostEkf = [zeros(pPost,1) logistic(psiPredPostEkf(:,1:tMax-1))];

%% friendship predictors
frndProbMatKatz = zeros(n,n,tMax);
frndProbMatAA = zeros(n,n,tMax);

for i=1:tMax-1
    temp = predictLinksKatz(frndadj(:,:,i),0.005,5);
    temp = temp*(1-frndadj(:,:,i));
    frndProbMatKatz(:,:,i+1) = temp/max(max(temp));
end
for i=1:tMax-1
    temp = predictLinksAA(frndadj(:,:,i));
    temp = temp*(1-frndadj(:,:,i));
    frndProbMatAA(:,:,i+1) = temp/max(max(temp));
end

%Combining prediction of friendships at t+1 with current friendships for
%prediction of interactions at t+1 
predMatFriendshipAA(:,:,2:tMax)  = frndProbMatAA(:,:,2:tMax) ...
    + frndadj(:,:,1:tMax-1);
predMatFriendshipKatz(:,:,2:tMax) = frndProbMatKatz(:,:,2:tMax) ...
    + frndadj(:,:,1:tMax-1);

%% interaction predictors
predMatPostEkf = predAdjMatDsbm(adj,blockvec2mat(thetaPredPostEkf, ...
                   directed),classPostEkf);
predMatEwma = zeros(n,n,tMax);
ff = 0.5;   % Forgetting factor for EWMA
ccWt = 0.01;   % Weight of EKF predictor in convex combination

predMatEwma(:,:,2) = adj(:,:,1);
for t = 3:tMax
    predMatEwma(:,:,t) = ff*predMatEwma(:,:,t-1) + (1-ff)*adj(:,:,t-1);
end
predMatPostEkf = ccWt*predMatPostEkf + (1-ccWt)*predMatEwma;

predMatKatz = zeros(n,n,tMax);
for i=1:tMax-1
    predMat_temp = predictLinksKatz(adj(:,:,i),0.005,5);
	predMatKatz(:,:,i+1) = predMat_temp/max(max(predMat_temp));
end

predMatAA = zeros(n,n,tMax);
for i=1:tMax-1
    predMat_temp = predictLinksAA(adj(:,:,i));
	predMatAA(:,:,i+1) = predMat_temp/max(max(predMat_temp));
end
predMatTSAA = zeros(n,n,tMax);
predMatTSKatz = zeros(n,n,tMax);
predMatTSAA(:,:,2) = predMatAA(:,:,2);
predMatTSKatz(:,:,2) = predMatKatz(:,:,2);
for t = 3:tMax
   predMatTSAA(:,:,t)  = ff*predMatTSAA(:,:,t-1) ...
                               + (1-ff)*predMatAA(:,:,t);
   predMatTSKatz(:,:,t)  = ff*predMatTSKatz(:,:,t-1) ...
                               + (1-ff)*predMatKatz(:,:,t);
end

%% Initializing convex combination weights and matrices to store results
a=0;
predMatPostEkffrnd = zeros(n,n,tMax);
predMatEwmafrnd = zeros(n,n,tMax);
predMatTSAAfrnd =  zeros(n,n,tMax);
predMatTSKatzfrnd= zeros(n,n,tMax);
for i=1:9
    a = a + 0.1;
    b = 1-a;
    %% Combining Interaction and friendship predictors
    predMatPostEkffrnd(:,:,2:end) = a*predMatPostEkf(:,:,2:end) ...
        + b*frndadj(:,:,1:end-1);
    predMatPostEkfKatz = a*predMatPostEkf + b*predMatFriendshipKatz;
    predMatPostEkfAA   = a*predMatPostEkf + b*predMatFriendshipAA;

    predMatEwmafrnd(:,:,2:end) = a*predMatEwma(:,:,2:end) ...
        + b*frndadj(:,:,1:end-1);
    predMatEwmaKatz = a*predMatEwma + b*predMatFriendshipKatz;               
    predMatEwmaAA   =  a*predMatEwma + b*predMatFriendshipAA;

    predMatTSAAfrnd(:,:,2:end) = a*predMatTSAA(:,:,2:end) ...
        + b*frndadj(:,:,1:end-1);
    predMatTSAA_Katz =a*predMatTSAA + b*predMatFriendshipKatz;                 
    predMatTSAA_AA   = a*predMatTSAA + b*predMatFriendshipAA;

    predMatTSKatzfrnd(:,:,2:end) = a*predMatTSKatz(:,:,2:end) ...
        + b*frndadj(:,:,1:end-1);
    predMatTSKatz_Katz =a*predMatTSKatz + b*predMatFriendshipKatz;                 
    predMatTSKatz_AA   = a*predMatTSKatz+ b*predMatFriendshipAA;

    %% Evaluating Interactions
    %PR_new
    %DSBM 
    [recPostEkf,precPostEkf,praucPostEkf(i),~,~,maxF1PostEkf] = ...
                dlpPRCurve(adj,predMatPostEkf,'new',true); %#ok<*SAGROW>
    [recPostEkffrnd,precPostEkffrnd,praucPostEkffrnd(i),~,~, ...
            maxF1PostEkffrnd] = dlpPRCurve(adj,predMatPostEkffrnd,'new',true);
    [recPostEkfAA,precPostEkfAA,praucPostEkfAA(i),~,~,maxF1PostEkfAA] = ...
                                 dlpPRCurve(adj, predMatPostEkfAA,'new',true);
    [recPostEkfKatz,precPostEkfKatz,praucPostEkfKatz(i),~,~, ...
            maxF1PostEkfKatz] = dlpPRCurve(adj,predMatPostEkfKatz,'new',true);


    %EWMA
    [recEwma,precEwma,praucEwma(i),~,~,maxF1Ewma] = dlpPRCurve(adj, ...
                       predMatEwma,'new',true);
    [recEwmafrnd,precEwmafrnd,praucEwmafrnd(i),~,~,maxF1Ewmafrnd] = ...
                       dlpPRCurve(adj,predMatEwmafrnd,'new',true);
    [recEwmaKatz,precEwmaKatz,praucEwmaKatz(i),~,~,maxF1EwmaKatz] = ...
                       dlpPRCurve(adj,predMatEwmaKatz,'new',true);
    [recEwmaAA,precEwmaAA,praucEwmaAA(i),~,~,maxF1EwmaAA] = dlpPRCurve(adj, ...
                       predMatEwmaAA,'new',true);

    % TS-AA      
    [recTSAA,precTSAA,praucTSAA(i),~,~,maxF1TSAA] = ...
               dlpPRCurve(adj,predMatTSAA,'new',true);
    [recTSAAfrnd,precTSAAfrnd,praucTSAAfrnd(i),~,~,maxF1TSAAfrnd] = ...
               dlpPRCurve(adj,predMatTSAAfrnd,'new',true);
    [recTSAA_Katz,precTSAA_Katz,praucTSAA_Katz(i),~,~,maxF1TSAA_Katz] = ...
               dlpPRCurve(adj,predMatTSAA_Katz,'new',true);
    [recTSAA_AA,precTSAA_AA,praucTSAA_AA(i),~,~,maxF1TSAA_AA] = ...
               dlpPRCurve(adj,predMatTSAA_AA,'new',true);

    %TS-Katz      
    [recTSKatz,precTSKatz,praucTSKatz(i),~,~,maxF1TSKatz] = ...
               dlpPRCurve(adj,predMatTSKatz,'new',true);
    [recTSKatzfrnd,precKatzAAfrnd,praucTSKatzfrnd(i),~,~,maxF1TSKatzfrnd] = ...
               dlpPRCurve(adj,predMatTSKatzfrnd,'new',true);
    [recTSKatz_Katz,precTSKatz_Katz,praucTSKatz_Katz(i),~,~, ...
        maxF1TSKatz_Katz] = dlpPRCurve(adj,predMatTSKatz_Katz,'new',true);
    [recTSKatz_AA,precTSKatz_AA,praucTSKatz_AA(i),~,~,maxF1TSKatz_AA] = ...
               dlpPRCurve(adj,predMatTSKatz_AA,'new',true);	

    %AUC_existing
    %DSBM 
    [fprPostEkf,tprPostEkf,~,aucPostEkf(i)] = ...
               dlpROCCurve(adj,predMatPostEkf,'existing',true);
    [fprPostEkffrnd,tprPostEkffrnd,~,aucPostEkffrnd(i)] = ...
               dlpROCCurve(adj,predMatPostEkffrnd,'existing',true);
    [fprPostEkfAA,tprPostEkfAA,~,aucPostEkfAA(i)] = dlpROCCurve(adj, ...
               predMatPostEkfAA,'existing',true);
    [fprPostEkfKatz,tprPostEkfKatz,~,aucPostEkfKatz(i)] = ...
               dlpROCCurve(adj,predMatPostEkfKatz,'existing',true);

    %EWMA
    [fprEwma,tprEwma,~,aucEwma(i)] = ...
            dlpROCCurve(adj,predMatEwma,'existing',true);
    [fprEwmafrnd,tprEwmafrnd,~,aucEwmafrnd(i)] = ...
            dlpROCCurve(adj,predMatEwmafrnd,'existing',true);
    [fprEwmaKatz,tprEwmaKatz,~,aucEwmaKatz(i)] = ...
            dlpROCCurve(adj,predMatEwmaKatz,'existing',true);
    [fprEwmaAA,tprEwmaAA,~,aucEwmaAA(i)] = dlpROCCurve(adj, ...
            predMatEwmaAA,'existing',true);


     %TS-AA      
    [fprTSAA,tprTSAA,~,aucTSAA(i)] = ...
               dlpROCCurve(adj,predMatTSAA,'existing',true);
    [fprTSAAfrnd,tprTSAAfrnd,~,aucTSAAfrnd(i)] = ...
               dlpROCCurve(adj,predMatTSAAfrnd,'existing',true);
    [fprTSAA_Katz,tprTSAA_Katz,~,aucTSAA_Katz(i)] = ...
               dlpROCCurve(adj,predMatTSAA_Katz,'existing',true);
    [fprTSAA_AA,tprTSAA_AA,~,aucTSAA_AA(i)] = ...
               dlpROCCurve(adj,predMatTSAA_AA,'existing',true);

    %TS-Katz      
    [fprTSKatz,tprTSKatz,~,aucTSKatz(i)] = ...
               dlpROCCurve(adj,predMatTSKatz,'existing',true);
    [fprTSKatzfrnd,tprKatzAAfrnd,~,aucTSKatzfrnd(i)] = ...
              dlpROCCurve(adj,predMatTSKatzfrnd,'existing',true);
    [fprTSKatz_Katz,tprTSKatz_Katz,~,aucTSKatz_Katz(i)] = ...
               dlpROCCurve(adj,predMatTSKatz_Katz,'existing',true);
    [fprTSKatz_AA,tprTSKatz_AA,~,aucTSKatz_AA(i)] = ...
               dlpROCCurve(adj,predMatTSKatz_AA,'existing',true);	


    %% unified metrics	


    unifiedMetricPostEkffrnd(i) = unifiedDlpMetric(praucPostEkffrnd(i), ...
                                                  aucPostEkffrnd(i),adj,true);
    unifiedMetricPostEkfAA(i) = unifiedDlpMetric(praucPostEkfAA(i), ...
                                             aucPostEkfAA(i),adj,true);
    unifiedMetricPostEkfKatz(i) = unifiedDlpMetric(praucPostEkfKatz(i), ...
                                           aucPostEkfKatz(i),adj,true);

    unifiedMetricEwmafrnd(i) = unifiedDlpMetric(praucEwmafrnd(i), ...
                                              aucEwmafrnd(i),adj,true);
    unifiedMetricEwmaAA(i) = unifiedDlpMetric(praucEwmaAA(i),aucEwmaAA(i), ...
                                                             adj,true);
    unifiedMetricEwmaKatz(i) = unifiedDlpMetric(praucEwmaKatz(i), ...
                                              aucEwmaKatz(i),adj,true);

    unifiedMetricTSAAfrnd(i) = unifiedDlpMetric(praucTSAAfrnd(i), ...
                                              aucTSAAfrnd(i),adj,true);
    unifiedMetricTSAA_AA(i) = unifiedDlpMetric(praucTSAA_AA(i), ...
                                               aucTSAA_AA(i),adj,true);
    unifiedMetricTSAA_Katz(i) = unifiedDlpMetric(praucTSAA_Katz(i), ...
                                             aucTSAA_Katz(i),adj,true);

    unifiedMetricTSKatzfrnd(i) = unifiedDlpMetric(praucTSKatzfrnd(i), ...
                                             aucTSKatzfrnd(i),adj,true);
    unifiedMetricTSKatz_AA(i) = unifiedDlpMetric(praucTSKatz_AA(i), ...
                                              aucTSKatz_AA(i),adj,true);
    unifiedMetricTSKatz_Katz(i) = unifiedDlpMetric(praucTSKatz_Katz(i), ...
                                            aucTSKatz_Katz(i),adj,true);
end

unifiedMetricPostEkf = unifiedDlpMetric(praucPostEkf(1),aucPostEkf(1),adj,true);
unifiedMetricEwma    = unifiedDlpMetric(praucEwma(1),aucEwma(1),adj,true);
unifiedMetricTSAA    = unifiedDlpMetric(praucTSAA(1),aucTSAA(1),adj,true);
unifiedMetricTSKatz  = unifiedDlpMetric(praucTSKatz(1),aucTSKatz(1),adj,true);

%% Compute weights that maximize GMAUC for each predictor
[~,wtEwmafrnd] = max(unifiedMetricEwmafrnd) %#ok<*NOPTS>
[~,wtTSAAfrnd] = max(unifiedMetricTSAAfrnd)
[~,wtTSKatzfrnd] = max(unifiedMetricTSKatzfrnd)
[~,wtPostEkffrnd] = max(unifiedMetricPostEkffrnd)
[~,wtEwmaAA] = max(unifiedMetricEwmaAA)
[~,wtTSAA_AA] = max(unifiedMetricTSAA_AA)
[~,wtTSKatz_AA] = max(unifiedMetricTSKatz_AA)
[~,wtPostEkfAA] = max(unifiedMetricPostEkfAA)
[~,wtEwmaKatz] = max(unifiedMetricEwmaKatz)
[~,wtTSAA_Katz] = max(unifiedMetricTSAA_Katz)
[~,wtTSKatz_Katz] = max(unifiedMetricTSKatz_Katz)
[~,wtPostEkfKatz] = max(unifiedMetricPostEkfKatz)

%% Display summary statistics
fprintf('\\hline \n')
fprintf('Predictor \t & PRAUC (new) \t & AUC (prev) \t & GMAUC \\\\ \n')
fprintf('\\hline \n')
fprintf('EWMA \t\t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucEwma(1),aucEwma(1),unifiedMetricEwma)
fprintf('TS-AA \t\t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucTSAA(1),aucTSAA(1),unifiedMetricTSAA)
fprintf('TS-Katz \t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucTSKatz(1),aucTSKatz(1),unifiedMetricTSKatz)
fprintf('DSBM \t\t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucPostEkf(1),aucPostEkf(1),unifiedMetricPostEkf)
fprintf('\\hline \n')
fprintf('EWMA + FR \t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucEwmafrnd(wtEwmafrnd),aucEwmafrnd(wtEwmafrnd), ...
    unifiedMetricEwmafrnd(wtEwmafrnd))
fprintf('TS-AA + FR \t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucTSAAfrnd(wtTSAAfrnd),aucTSAAfrnd(wtTSAAfrnd), ...
    unifiedMetricTSAAfrnd(wtTSAAfrnd))
fprintf('TS-Katz + FR \t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucTSKatzfrnd(wtTSKatzfrnd),aucTSKatzfrnd(wtTSKatzfrnd), ...
    unifiedMetricTSKatzfrnd(wtTSKatzfrnd))
fprintf('DSBM + FR \t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucPostEkffrnd(wtPostEkffrnd),aucPostEkffrnd(wtPostEkffrnd), ...
    unifiedMetricPostEkffrnd(wtPostEkffrnd))
fprintf('\\hline \n')
fprintf('EWMA + AA \t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucEwmaAA(wtEwmaAA),aucEwmaAA(wtEwmaAA), ...
    unifiedMetricEwmaAA(wtEwmaAA))
fprintf('EWMA + Katz \t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucEwmaKatz(wtEwmaKatz),aucEwmaKatz(wtEwmaKatz), ...
    unifiedMetricEwmaKatz(wtEwmaKatz))
fprintf('TS-AA + AA \t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucTSAA_AA(wtTSAA_AA),aucTSAA_AA(wtTSAA_AA), ...
    unifiedMetricTSAA_AA(wtTSAA_AA))
fprintf('TS-AA + Katz \t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucTSAA_Katz(wtTSAA_Katz),aucTSAA_Katz(wtTSAA_Katz), ...
    unifiedMetricTSAA_Katz(wtTSAA_Katz))
fprintf('TS-Katz + AA \t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucTSKatz_AA(wtTSKatz_AA),aucTSKatz_AA(wtTSKatz_AA), ...
    unifiedMetricTSKatz_AA(wtTSKatz_AA))
fprintf('TS-Katz + Katz \t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucTSKatz_Katz(wtTSKatz_Katz),aucTSKatz_Katz(wtTSKatz_Katz), ...
    unifiedMetricTSKatz_Katz(wtTSKatz_Katz))
fprintf('DSBM + AA \t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucPostEkfAA(wtPostEkfAA),aucPostEkfAA(wtPostEkfAA), ...
    unifiedMetricPostEkfAA(wtPostEkfAA))
fprintf('DSBM + Katz \t & %5.3f \t & %5.3f \t & %5.3f \\\\ \n', ...
    praucPostEkfKatz(wtPostEkfKatz),aucPostEkfKatz(wtPostEkfKatz), ...
    unifiedMetricPostEkfKatz(wtPostEkfKatz))
fprintf('\\hline \n')

%% Save results
save([ResultsPath 'ResultsFBFriendshipInteractions_' ... 
     int2str(minDegfrnd) 'Degfrnd_' int2str(minOutInDeg) 'OutInDeg.mat'])
