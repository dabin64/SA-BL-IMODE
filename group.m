function [groups,groupNum] = group(BI)
global ulFunctionEvaluations
global llFunctionEvaluations
ulFunctionEvaluations = 0;
llFunctionEvaluations = 0;
epsilon = 1e-4;
xL = BI.xrange(1,:)+rand(1,BI.dim).*(BI.xrange(2,:)-BI.xrange(1,:));
num = max(2,BI.l_ieqcon_num+BI.l_eqcon_num+1);
archiveL = -1*ones(BI.dim,num);
lambdaObjL = -1*ones(BI.dim,BI.dim);
lambdaConL = -1*ones(BI.dim,BI.dim);

x1 = BI.xrange(1,:)+1e-1;
[obj,con]= llEvaluate(x1(1:BI.u_dim),x1(BI.u_dim+1:BI.dim),BI.fn);
llFunctionEvaluations = llFunctionEvaluations + 1;
fit1 =[obj,con];

for i = 1:BI.dim-1   
    if ((archiveL(i,:))~=-1)
        fit2 = archiveL(i,:);
    else
        x2 = x1;
        x2(i) = xL(i);
        [obj,con] = llEvaluate(x2(1:BI.u_dim),x2(BI.u_dim+1:BI.dim),BI.fn);
        llFunctionEvaluations = llFunctionEvaluations + 1;
        fit2 = [obj,con];
        archiveL(i,:) = fit2;
    end
    
    for j = BI.u_dim+1:BI.dim
        if j>i
            if ((archiveL(j,:))~=-1)
                fit3 = archiveL(j,:);
            else
                x3 = x1;
                x3(j) = xL(j);
                [obj,con] = llEvaluate(x3(1:BI.u_dim),x3(BI.u_dim+1:BI.dim),BI.fn);
                llFunctionEvaluations = llFunctionEvaluations + 1;
                fit3 = [obj,con];
                archiveL(j,:) = fit3;
            end
            x4 = x1;
            x4(i) = xL(i);
            x4(j) = xL(j);
            [obj,con] = llEvaluate(x4(1:BI.u_dim),x4(BI.u_dim+1:BI.dim),BI.fn);
            llFunctionEvaluations = llFunctionEvaluations + 1;
            fit4 = [obj,con];
            d1 = fit2(1)-fit1(1);
            d2 = fit4(1)-fit3(1);
            d3 = abs(fit2(2:end)-fit1(2:end));
            d4 = abs(fit4(2:end)-fit2(2:end));
            
            lambdaObjL(i,j) = abs(d1-d2);
            lambdaConL(i,j) = sum(d4.*d3)>0;
        end
    end
end
adjL = lambdaObjL>epsilon | lambdaConL>0;

xU = BI.xrange(1,:)+rand(1,BI.dim).*(BI.xrange(2,:)-BI.xrange(1,:));
num = max(2,BI.u_ieqcon_num+BI.u_eqcon_num+1);
archiveU = -1*ones(BI.dim,num);
lambdaObjU = -1*ones(BI.dim,BI.dim);
lambdaConU = -1*ones(BI.dim,BI.dim);

x1 = BI.xrange(1,:)+1e-1;
[f,c] = ulEvaluate(x1(1:BI.u_dim),x1(BI.u_dim+1:BI.dim),BI.fn);
ulFunctionEvaluations = ulFunctionEvaluations + 1;
fit1 = [f,c];

for i = 1:BI.u_dim  
    if ((archiveU(i,:))~=-1)
        fit2 = archiveU(i,:);
    else
        x2 = x1;
        x2(i) = xU(i);
        [f,c] = ulEvaluate(x2(1:BI.u_dim),x2(BI.u_dim+1:BI.dim),BI.fn);
        ulFunctionEvaluations = ulFunctionEvaluations + 1;
        fit2 = [f,c];
        archiveU(i,:) = fit2;
    end
    
    for j = 1:BI.u_dim
        if j>i
            if adjL(i,j)~=1
                if ((archiveU(j,:))~=-1)
                    fit3 = archiveU(j,:);
                else
                    x3 = x1;
                    x3(j) = xU(j);
                    [f,c] = ulEvaluate(x3(1:BI.u_dim),x3(BI.u_dim+1:BI.dim),BI.fn);
                    ulFunctionEvaluations = ulFunctionEvaluations + 1;
                    fit3 = [f,c];
                    archiveU(j,:) = fit3;
                end
                
                x4 = x1;
                x4(i) = xU(i);
                x4(j) = xU(j);
                [f,c] = ulEvaluate(x4(1:BI.u_dim),x4(BI.u_dim+1:BI.dim),BI.fn);
                ulFunctionEvaluations = ulFunctionEvaluations + 1;
                fit4 = [f,c];
                d1 = fit2(1)-fit1(1);
                d2 = fit4(1)-fit3(1);
                d3 = abs(fit2(2:end)-fit1(2:end));
                d4 = abs(fit4(2:end)-fit2(2:end));
                
                lambdaObjU(i,j) = abs(d1-d2);
                lambdaConU(i,j) = sum(d4.*d3)>0;
            end
        end
    end
end
adjU = lambdaObjU>epsilon | lambdaConU>0;
adj = adjL|adjU;
adj(logical(eye(BI.dim))) = 1;
adj = adj|adj';
labels = findConnComp(adj);

groupNum = max(labels);
groups = cell(groupNum,1);

for i = 1:groupNum
    index = find(labels == i);
    if all(index <= BI.u_dim) || all(index > BI.u_dim)
        groups{i}{1} = index;
    else
        indexU = index(index <= BI.u_dim);
        groups{i}{1} = indexU;
        indexL = index(index > BI.u_dim);
        adjL1 = adjL(indexL,indexL);
        labels1 = findConnComp(adjL1);
        groupNum1 = max(labels1);
        for k = 1:groupNum1
            groups{i}{k+1} = [indexL(labels1==k)];
        end
    end
end
end

function labels = findConnComp(adj)

L = size(adj,1);

labels = zeros(1,L);
rts = [];
ccc = 0;

while true
    ind = find(labels==0);
    if ~isempty(ind)
        fue = ind(1);
        rts = [rts fue];
        list = [fue];
        ccc = ccc+1;
        labels(fue) = ccc;
        while true
            list_new = [];
            for lc = 1:length(list)
                p = list(lc);
                cp = find(adj(p,:));
                cx1 = cp(labels(cp)==0);
                labels(cx1)=ccc;
                list_new = [list_new cx1];
            end
            list = list_new;
            if isempty(list)
                break;
            end
        end
    else
        break;
    end
end
end
