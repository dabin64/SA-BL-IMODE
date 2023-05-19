function [F,C] = ulEvaluate(UPop,LPOP,fn)
[F,~,C] = ulTestProblem(UPop, LPOP, fn);
if isempty(C)
    C = zeros(size(UPop,1),1);
end
end