% Script to filter out inactive nodes and nodes with low degrees from
% Viswanath et al. (2009)'s Facebook data
% Author: Ruthwik Junuthula & Kevin S. Xu

binSize = 90;   % Size of time bins in days
minDegfrnd = 120;    % Minimum degree over the entire friendship adj matrix
minOutInDeg = 0; % Minimum out or in-degree over the entire interactions adj matrix
matFilePath = '';
dataFile = [matFilePath 'FacebookBothAdj' int2str(binSize) 'Days.mat'];

%% Load adjacency matrices
load(dataFile)
tMax = length(adj);

%% Compute minimum degree over friendships
adjAllTimes = frndadj{1};
for t = 2:tMax
    adjAllTimes = adjAllTimes + frndadj{t};
end
adjAllTimes(adjAllTimes>0) = 1;

% Identify nodes with at least the minimum degree and remove all other nodes
degfrnd = sum(adjAllTimes)';
minDegNodesfrnd = (degfrnd>=minDegfrnd);

%% Compute minimum degree over interactions (wall posts)
adjAllTimes = adj{1};
for t = 2:tMax
    adjAllTimes = adjAllTimes + adj{t};
end
adjAllTimes(adjAllTimes>0) = 1;

% Identify nodes with at least the minimum out or in-degree and remove all
% other nodes
outDeg = sum(adjAllTimes,2);
inDeg = sum(adjAllTimes)';
minDegNodes = (outDeg>=minOutInDeg) | (inDeg>=minOutInDeg);
minDegNodes = minDegNodes & minDegNodesfrnd;
for t = 1:tMax
    adj{t} = adj{t}(minDegNodes,minDegNodes); %#ok<SAGROW>
    frndadj{t}=frndadj{t}(minDegNodes,minDegNodes); %#ok<SAGROW>
end

%% Convert to full 3-D adjacency matrix form
adj = full(cell2mat(adj));
n = size(adj,1);
adj = reshape(adj,[n n tMax]);
frndadj=full(cell2mat(frndadj));
n=size(adj,1);
frndadj=reshape(frndadj,[n n tMax]);

%% Save filtered and reshaped adjacency matrix
filtFile = [matFilePath 'FacebookFilteredAdj_' int2str(binSize) 'Days_' ...
    int2str(minDegfrnd) 'Degfrnd_' int2str(minOutInDeg) 'OutInDeg.mat'];
save(filtFile,'binSize','minDegfrnd','minOutInDeg','frndadj','adj', ...
    'endDate','startDate','traceEndDate','traceStartDate')
