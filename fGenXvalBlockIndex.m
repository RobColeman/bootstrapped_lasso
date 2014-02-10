function [RandBlockIndex,NBlocks,OrderedBlockIndex,RandomOrder] = fGenXvalBlockIndex(N,NBlocks)
%
%
%   Usage: [RandBlockIndex,OrderedBlockIndex,RandomOrder] = fGenXvalBlockIndex(N,NBlocks)
%
%   Inputs:
%       N: Number of data cases
%       NBlocks: Number of Cross-Validation Blocks
%   Outputs:
%       RandBlockIndex: Random Ordered sequence of Xvalidation Block indices, numbers 1:NBlocks
%       OrderedBlockIndex: Xvalidation blocks in order
%       RandomOrder:    Random Sequence of intigers from 1:N for shuffling data order
%
%
%
%   Usage in Xvalidation
%   [RandBlockIndex,NBlocks] = fGenXvalBlockIndex(N,NBlocks);
%   For j = 1:NBlocks   
%         TrIDX   = RandBlockIndex~=j;
%         TeIDX   = RandBlockIndex==j;
%         Xtr     = X(TrIDX,:);
%         Xte     = X(TeIDX,:);
%         Ytr     = Y(TrIDX);
%         Yte     = Y(TeIDX);
%
%         Train and Test here
%         Store Results and Model Information
%   end % Over Xvalidation Blocks
%
%   For Leave-One-Out Crossvalidation, use NBlocks == N
%
%
%
%
%   Created by: 
%
%   Robert Coleman - rbrt.coleman@gmail.com
%       September 2012
%%


if  mod(N,NBlocks-1) == 0
    NBlocks = NBlocks-1;
end
RandomOrder = randperm(N);
NperBlock   = ceil(N/NBlocks);

OrderedBlockIndex = [];
for j = 1:(NBlocks-1)
   BlockNum = j;
   OrderedBlockIndex = [OrderedBlockIndex; BlockNum*ones(NperBlock,1) ];
end
BlockNum = NBlocks;
% Last Block
OrderedBlockIndex = [OrderedBlockIndex; BlockNum*ones(N-length(OrderedBlockIndex),1)];

RandBlockIndex = OrderedBlockIndex(RandomOrder);
end % function