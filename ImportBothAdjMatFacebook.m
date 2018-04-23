% Script to import Facebook data from Viswanath et al. (2009)
% Author: Ruthwik Junuthula and Kevin S. Xu

binSize = 90;   % Size of time bins in days
rawDataPath = '';
wallFile = [rawDataPath 'facebook-wall.txt'];
friendsFile = [rawDataPath 'facebook-links.txt'];
matFilePath = '';

startDate = datenum(2006,9,1);
endDate = datenum(2009,1,1);

%% Load raw data
f = fopen(wallFile,'r');
wallData = textscan(f,'%f %f %f');
fclose(f);
nRowsWall = length(wallData{3});

f = fopen(friendsFile,'r');
friendsData = textscan(f,'%f %f %s');
fclose(f);
nRowsFriends = length(friendsData{3});

n = max(max(wallData{1}),max(wallData{2}));   % Number of nodes

traceStartDate = floor(unixtime2serial(min(wallData{3})));
traceEndDate = ceil(unixtime2serial(max(wallData{3})));
if isempty(startDate)
    startDate = traceStartDate;
end
if isempty(endDate)
    endDate = traceEndDate;
end
traceLen = endDate - startDate;
tMax = ceil(traceLen/binSize)-1;    % Number of time steps
endDate = startDate + tMax*binSize;

%% Create adjacency matrices
fromNodeArray = cell(1,tMax);
toNodeArray = cell(1,tMax);
numEdges = zeros(1,tMax);
for t = 1:tMax
    fromNodeArray{t} = zeros(nRowsWall,1);
    toNodeArray{t} = zeros(nRowsWall,1);
end

disp('Importing wall posts')
tic
for k = 1:nRowsWall
    if mod(k,100000) == 0
        disp(['Processing entry ' int2str(k)])
    end
    fromNode = wallData{2}(k);
    toNode = wallData{1}(k);
    rowDate = wallData{3}(k);
    t = ceil((unixtime2serial(rowDate) - startDate) / binSize);
    if (t > 0) && (t <= tMax) && (fromNode ~= toNode)
        numEdges(t) = numEdges(t)+1;
        fromNodeArray{t}(numEdges(t)) = fromNode;
        toNodeArray{t}(numEdges(t)) = toNode;
    end
end

adj = cell(1,tMax);
for t = 1:tMax
    adj{t} = sparse(fromNodeArray{t}(1:numEdges(t)), ...
        toNodeArray{t}(1:numEdges(t)),ones(numEdges(t),1),n,n);
    adj{t}(adj{t} > 0) = 1;
end
toc

disp('Importing friendship edges')
fromNodeArray = cell(1,tMax);
toNodeArray = cell(1,tMax);
numEdges = zeros(1,tMax);
for t = 1:tMax
    fromNodeArray{t} = zeros(2*nRowsFriends,1);
    toNodeArray{t} = zeros(2*nRowsFriends,1);
end

tic
for k = 1:nRowsFriends
    if mod(k,100000) == 0
        disp(['Processing entry ' int2str(k)])
    end
    fromNode = friendsData{2}(k);
    toNode = friendsData{1}(k);
    rowDate = friendsData{3}{k};
    
    if (fromNode > n) || (toNode > n)
        continue
    end
    
    if strcmp(rowDate,'\N')
        % If friendship formation time is unknown, assume it was there
        % since the start time
        tStart = 1;
    else
        rowDateSerial = unixtime2serial(str2double(rowDate));
        tStart = ceil((rowDateSerial - startDate) / binSize);
    end
    
    % Place friendship edge in all time steps after formation. Friendship
    % network is undirected so it is addition of 2 reciprocated edges
    % corresponding to 2 adjacency matrix entries.
    if (tStart > 0) && (tStart <= tMax)
        numEdges(tStart:tMax) = numEdges(tStart:tMax)+2;
        for t = tStart:tMax
            fromNodeArray{t}(numEdges(t)-1) = fromNode;
            toNodeArray{t}(numEdges(t)-1) = toNode;
            fromNodeArray{t}(numEdges(t)) = toNode;
            toNodeArray{t}(numEdges(t)) = fromNode;
        end
    end
end

frndadj = cell(1,tMax);
for t = 1:tMax
    frndadj{t} = sparse(fromNodeArray{t}(1:numEdges(t)), ...
        toNodeArray{t}(1:numEdges(t)),ones(numEdges(t),1),n,n);
    frndadj{t}(frndadj{t} > 0) = 1;
end
toc

%% Save processed adjacency matrices
save([matFilePath 'FacebookBothAdj' int2str(binSize) 'Days.mat'],'binSize', ...
    'adj','frndadj','endDate','startDate','traceEndDate','traceStartDate')
