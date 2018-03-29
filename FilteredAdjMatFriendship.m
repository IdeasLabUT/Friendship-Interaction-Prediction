function [FriendAdjMat]=FilteredAdjMatFriendship(NodeNum,n)
rawData = adjFriends;
Maxnode=max(NodeNum);
FriendAdjMat = sparse(Maxnode,Maxnode);
Nodes=adjFriends;
Friends=rawData{2};
FilteredNodes=false(Maxnode,1);
for i=1:n
    FilteredNodes(NodeNum(i))=1;
end
for i=1:Maxnode
    FriendAdjMat(Nodes(i),Friends(i))=1;%Making the friendship network 
    FriendAdjMat(Friends(i),Nodes(i))=1; ...directed
end 
FriendAdjMat=FriendAdjMat(FilteredNodes,FilteredNodes);
end
