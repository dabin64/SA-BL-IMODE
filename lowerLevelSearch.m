function [bestLX,bestLF,bestLC] = lowerLevelSearch(xu,level_Archive,BI,groups,groupNum)
global   llFE
%% Parameter setting
Archive = [];
record = [];
minN     = 4;
aRate    = 10;
MCR = zeros(60*BI.l_dim,1) + 0.2;
MF  = zeros(60*BI.l_dim,1) + 0.2;
k   = 1;
MOP = ones(1,5)/5;
maxIter = ceil(BI.LmaxFEs / BI.l_N);
ImprIter = ceil(BI.LmaxImprFEs / BI.l_N);
llFE     = 0;
best.stop = 0;
sqp.stop = 0;
tt       = 0;
cycle    = 0;
%% Generate random population
[B_Init,Population] = Level_Initialization(xu,level_Archive,BI,groups,groupNum);
%% init dataset
train_data             = [];
Incre_learning         = [];
[~,init_rank]         = sort(FitnessSingle(Population));
init_Population       = Population(init_rank(1:BI.l_N));
for i = 1:BI.l_N
    train_data{1,i}    = init_Population(i);
end
for iter = 1 : maxIter
    if(best.stop == 0 )
        % Reduce the population size
        N          = ceil((minN-BI.l_N)*llFE/BI.LmaxFEs) + BI.l_N;
        [~,rank]   = sort(FitnessSingle(Population));
        Population = Population(rank(1:N));
        Archive    = Archive(randperm(end,min(end,ceil(aRate*N))));
        % Eigencoordinate System
        [~,I]  = sort(Population.objs,'ascend');
        num    = min(N,BI.l_dim);
        TopDec = Population(I(1:num)).decs;
        B      = eig(cov(TopDec));
        % Eigencoordinate System
        if isempty(B_Init)
            [~,I]  = sort(Population.objs,'ascend');
            num    = min(N,BI.l_dim);
            TopDec = Population(I(1:num)).decs;
            B1      = eig(cov(TopDec));
        else
            B1    =  B_Init;
        end
        R1= deal(zeros(size(B,2)));
        R1(logical(eye(size(B,2)))) = rand(1,size(B,2));
        R2= deal(zeros(size(B1,2)));
        R2(logical(eye(size(B1,2)))) = rand(1,size(B1,2));
        %% RCES
        % construct train set
        RCPS_Xte             = [];
        train_set_uni        = [];
        train_set            = [];
        train_set_tem        = [];
        Y_lable              = [];
        Xtr                  = [];
        ind                  = [];
        train_rank           = [];
        for i = 1:N
            train_set_tem   = [train_data{1,i},train_set_tem];
        end
        train_set_tem       = [Population,train_set_tem];
        [~,ind]             = unique(train_set_tem.objs);
        trainN              = size(ind,1);
        train_set_uni       = train_set_tem(ind(1:trainN));
        [~,train_rank]      = sort(FitnessSingle(train_set_uni));
        train_set           = train_set_uni(train_rank(1:trainN));
        if(trainN<=3)
            train_set        = Population;
            trainN           = size(Population,2);
        end
        % increment learning
        incre_lable         = ones(size(Incre_learning,1),1);
        incre_Xtr           = Incre_learning;
        % Generate Lable
        for i=1:trainN
            for j =1:trainN-1
                Y_lable((i-1)*(trainN-1)+j,1)=1;
            end
            if(i>1)
                for kk=1:i-1
                    Y_lable((i-1)*(trainN-1)+kk,1)=2;
                end
            end
        end
        % tr data
        t=0;
        for n=1:trainN
            for  m=1:trainN
                if(n==m)
                else
                    t=t+1;
                    Xtr(t,:) = [train_set(n).dec,train_set(m).dec];
                end
            end
        end
          % Generate parents
    for i=1:N
        for j= 1:3                              % 预选子代数量为3
         Population1(3*(i-1)+j) = Population(i);
        end
    end
  % Generate parents, CR, F, and operator for each offspring
    Xp1 = Population1(ceil(rand(1,3*N).*max(1,0.25*3*N))).decs;
    Xp2 = Population1(ceil(rand(1,3*N).*max(2,0.5*3*N))).decs;
    Xr1 = Population1(randi(end,1,3*N)).decs;
    Xr3 = Population1(randi(end,1,3*N)).decs;
    P   = [Population1,Archive];
    Xr2 = P(randi(end,1,3*N)).decs;
    CR  = randn(3*N,1).*sqrt(0.1) + MCR(randi(end,3*N,1));
    CR  = sort(CR);
    CR  = repmat(max(0,min(1,CR)),1,BI.l_dim);
    F   = min(1,trnd(1,3*N,1).*sqrt(0.1) + MF(randi(end,3*N,1)));
    while any(F<=0)
        F(F<=0) = min(1,trnd(1,sum(F<=0),1).*sqrt(0.1) + MF(randi(end,sum(F<=0),1)));
    end
    F  = repmat(F,1,BI.l_dim);
    OP = arrayfun(@(S)find(rand<=cumsum(MOP),1),1:3*N);
    OP = arrayfun(@(S)find(OP==S),1:length(MOP),'UniformOutput',false);
     % Generate offspring
    PopDec = Population1.decs;clearvars Population1
    OffDec2 = PopDec;
    OffDec2(OP{1},:) = PopDec(OP{1},:) + F(OP{1},:).*(Xp1(OP{1},:)-PopDec(OP{1},:)+Xr1(OP{1},:)-Xr2(OP{1},:));
    OffDec2(OP{2},:) = PopDec(OP{2},:) + F(OP{2},:)*B*R1*B'.*(Xp1(OP{2},:)-PopDec(OP{2},:)+Xr1(OP{2},:)-Xr3(OP{2},:));
    OffDec2(OP{3},:) = F(OP{3},:)*B*R1*B'.*(Xr1(OP{3},:)+Xp2(OP{3},:)-Xr3(OP{3},:));
    OffDec2(OP{4},:) = PopDec(OP{4},:) + F(OP{4},:)*B1*R2*B1'.*(Xp1(OP{4},:)-PopDec(OP{4},:)+Xr1(OP{4},:)-Xr3(OP{4},:));
    OffDec2(OP{5},:) = F(OP{5},:)*B1*R2*B1'.*(Xr1(OP{5},:)+Xp2(OP{5},:)-Xr3(OP{5},:));
    if rand < 0.4
        Site = rand(size(CR)) > CR;
        OffDec2(Site) = PopDec(Site);
    else
        p1 = randi(BI.u_dim,3*N,1);
        p2 = arrayfun(@(S)find([rand(1,BI.u_dim),2]>CR(S,1),1),1:3*N);
        for i = 1 : 3*N
            Site = [1:p1(i)-1,p1(i)+p2(i):BI.u_dim];
            OffDec2(i,Site) = PopDec(i,Site);
        end
    end
    clf                            = py.sklearn.tree.DecisionTreeClassifier('gini','best',int16(5));
    clf.fit(py.numpy.array([Xtr;incre_Xtr]), py.numpy.array([Y_lable;incre_lable]));
    %%RCPS
      % te data
     for i=1:N
           RCPS_Xte(6*(i-1)+1,:) =    [OffDec2(3*(i-1)+1,:),OffDec2(3*(i-1)+2,:)];
           RCPS_Xte(6*(i-1)+2,:) =    [OffDec2(3*(i-1)+1,:),OffDec2(3*(i-1)+3,:)];
           RCPS_Xte(6*(i-1)+3,:) =    [OffDec2(3*(i-1)+2,:),OffDec2(3*(i-1)+1,:)];
           RCPS_Xte(6*(i-1)+4,:) =    [OffDec2(3*(i-1)+2,:),OffDec2(3*(i-1)+3,:)];
           RCPS_Xte(6*(i-1)+5,:) =    [OffDec2(3*(i-1)+3,:),OffDec2(3*(i-1)+1,:)];
           RCPS_Xte(6*(i-1)+6,:) =    [OffDec2(3*(i-1)+3,:),OffDec2(3*(i-1)+2,:)];
     end
          RCPS_pre_lable           = clf.predict(py.numpy.array(RCPS_Xte));
          RCPS_pred                = double(RCPS_pre_lable)';
          % preselection RCPS
          for i=1:N
              for j= 1:3
                  iss(j,1)           = RCPS_pred(6*(i-1)+2*(j-1)+1,1)+RCPS_pred(6*(i-1)+2*(j-1)+2,1);
              end
              m=min(iss(:,1));
              [row y]=find(iss(:,1)==m);
              if isempty(row)
                  OffDec1(i,:)       = OffDec2(3*(i-1)+randi(3),:);
              else
                  if (size(row,1)>1)
                      OffDec1(i,:)   = OffDec2(3*(i-1)+row(randi(size(row,1))),:);
                  end
                  if (size(row,1)==1)
                      OffDec1(i,:)   = OffDec2(3*(i-1)+row(1),:);
                  end
              end
              clearvars row
          end
          OffDec = OffDec1;
       clearvars OffDec2 OffDec1 PopDec RCPS_pre_lable RCPS_pred
    %%RCES
    RCES_Xte                       = [Population.decs,OffDec];
    pre_lable                      = clf.predict(py.numpy.array(RCES_Xte));
    RCES_pred                      = double(pre_lable)';
    RCES_replace                   =  RCES_pred-1;
    Offspring                      = Population;
    RCES_row                       = find(RCES_replace==1);
    for i = 1 : size(RCES_row,1)
        [F_CES,C_CES,PopDec_CES]   = LL_evaluate(xu,OffDec(RCES_row(i),:),BI);
        Population_CES             = SOLUTION(BI,PopDec_CES,F_CES,C_CES);
        if(FitnessSingle(Population(RCES_row(i)))>FitnessSingle(Population_CES))
            Offspring(RCES_row(i))  = Population_CES  ;
        else
            Incre                   = [Population(RCES_row(i)).dec,Population_CES.dec];
            Incre_learning          = [Incre_learning;Incre];
        end
    end
    if( cycle>=50)
        for i = 1 : N
            [F(i,:),C(i,:),PopDec(i,:)]= LL_evaluate(xu,OffDec(i,:),BI);
        end
        Offspring = SOLUTION(BI,PopDec,F,C);
    end
    % Update the population and archive
    delta   = FitnessSingle(Population) - FitnessSingle(Offspring);
    replace = delta > 0;
    Archive = [Archive,Population(replace)];
    Archive = Archive(randperm(end,min(end,ceil(aRate*N))));
    Population(replace) = Offspring(replace);
    %% build and update train_data
    for i = 1:N
        if(replace(i))
            train_data{1,i}       = [Population(i),train_data{1,i}];
            train_data{1,i}       = train_data{1,i}(1:min(end,2));
        end
    end
    % Update CR, F, and probabilities of operators
    for i=1:N
        for j= 1:3                              % 预选子代数量为3
            delta1(3*(i-1)+j,:)   = delta(i,:);
            replace1(3*(i-1)+j,:) = replace(i,:);
            Population2(3*(i-1)+j) = Population(i);
        end
    end
    if any(replace1)
        w      = delta1(replace1)./sum(delta1(replace1));
        MCR(k) = (w'*CR(replace1,1).^2)./(w'*CR(replace1,1));
        MF(k)  = (w'*F(replace1,1).^2)./(w'*F(replace1,1));
        k      = mod(k,length(MCR)) + 1;
        cycle  = 0;
    else
        cycle  = cycle+1;
        MCR(k) = 0.2;
        MF(k)  = 0.2;
    end
    delta1 = max(0,delta1./abs(FitnessSingle(Population2)));
    if any(cellfun(@isempty,OP))
        MOP = ones(1,5)/5;
    else
        MOP = cellfun(@(S)mean(delta1(S)),OP);
        MOP = max(0.1,min(0.9,MOP./sum(MOP)));
    end
     clearvars delta1 replace1 Population2
     clearvars pre_lable RCES_pred RCES_replace RCES_row replace RCES_Xte 
    if isempty(Population.best)
    else
        tt=tt+1;
        record(tt,1) =  Population.best.obj;
    if (tt>ImprIter && abs(record(tt,1)-record(tt-ImprIter+1,1)) <  1e-1)&&(sqp.stop == 0)
        x0   = Population.best.dec;
        [level_obj,lever_y, ~] = SQP(xu,x0,BI);
        if(level_obj<Population.best.obj)
            Population.best.obj =level_obj;
            Population.best.dec = lever_y;
            Population.best.con = 0;
            best.stop = 1;
        end
          sqp.stop = 1;
    end
     if (tt>ImprIter && abs(record(tt,1)-record(tt-ImprIter+1,1)) <  1e-5)
           best.stop = 1;
     end
    end
    else
    end
end
if isempty(Population.best)
  [~,Best] = min(Population.objs);
bestLX = Population(Best).dec;
bestLF = Population(Best).obj;
bestLC = Population(Best).con;
else
bestLX = Population.best.dec;
bestLF = Population.best.obj;
bestLC = Population.best.con;
end
end
function [F,C,xl] = LL_evaluate(xu,xl,BI)
for j = 1 : size(xl,2)
    xl(:,j) = max(min(xl(:,j),repmat(BI.l_ub(1,j),size(xl,1),1)),repmat(BI.l_lb(1,j),size(xl,1),1));
end
[F,~,C] = llTestProblem(xl,BI.fn,xu);
 C = sum(max(0,C));
end
function [B,Population] = Level_Initialization(xu,level_Archive,BI,groups,groupNum)
 PopDec1 = unifrnd(repmat(BI.l_lb,BI.l_N,1),repmat(BI.l_ub,BI.l_N,1));
 if(size(level_Archive,1)>=5)
    [knownledge_xl] = get_knownledge(groups,groupNum,xu,level_Archive,BI,3);
     PopDec2  = knownledge_xl;
     PopDec_B =level_Archive(1:5,BI.u_dim+1:end);

      % Eigencoordinate System
      B      = eig(cov(PopDec_B));
 else
      B      = [];
%   PopDec2 = level_Archive(randi(end,1,0.6*BI.l_N),1:BI.l_dim);
   PopDec2 = unifrnd(repmat(BI.l_lb, 0.6*BI.l_N,1),repmat(BI.l_ub,0.6*BI.l_N,1));
 end
  PopDect  =[PopDec2;PopDec1];
  PopDec  =PopDect(1:BI.l_N,:);
for i = 1 : BI.l_N
  [F(i,:),C(i,:),PopDec(i,:)]= LL_evaluate(xu,PopDec(i,:),BI);  
end
Population = SOLUTION(BI,PopDec,F,C);
end
function F = SQP_LLevaluate_F(xu,xl,BI)
global LC
for j = 1 : size(xl,2)
    xl(:,j) = max(min(xl(:,j),repmat(BI.l_ub(1,j),size(xl,1),1)),repmat(BI.l_lb(1,j),size(xl,1),1));
end
[F,~,C] = llTestProblem(xl,BI.fn,xu);
C = sum(max(0,C));
LC = C;
end
%% 获取知识
function [knownledge_xl] = get_knownledge(groups,groupNum,xu,level_Archive,BI,KnowNum)
global  best_lx
bestVector = zeros(KnowNum,BI.dim);
u_matrix   = [];
l_matrix   = [];
for i = 1:groupNum
    variable = [groups{i}{:}];
    if all(variable <= BI.u_dim)
        u_matrix    = [u_matrix,variable(variable <= BI.u_dim)];
    elseif all(variable > BI.u_dim)
        l_matrix    = [l_matrix,variable(variable > BI.u_dim)];
    else
        bl_level_Archive = [level_Archive(:,variable(variable <= BI.u_dim)),level_Archive(:,variable(variable > BI.u_dim))];
        matrix1=xu(:,variable(variable <= BI.u_dim));
        matrix2 =bl_level_Archive(:,(variable <= BI.u_dim));
        d = computeDistance(matrix1, matrix2);
        [~, closestParent] = sort( d,'ascend');
        best =bl_level_Archive(closestParent(1:KnowNum),(variable > BI.u_dim));
        bestVector(:,variable(variable > BI.u_dim)) = best;
    end
end
if isempty(best_lx)
ul_lla = [level_Archive(:,u_matrix),level_Archive(:,l_matrix)];
matrix_xu = xu(:,u_matrix);
matrix_lla_xu = ul_lla(:,1:size(u_matrix,2));
d = computeDistance(matrix_xu, matrix_lla_xu);
[~, closestParent] = sort( d,'ascend');
best = ul_lla(closestParent(1:KnowNum),size(u_matrix,2)+1:end);
bestVector(:,l_matrix) = best;
else
tt  =l_matrix-BI.u_dim; 
for ii=1:KnowNum
    bestVector(ii,l_matrix) = best_lx(1,tt);
end
end
knownledge_xl          = bestVector(:,BI.u_dim+1:end);
end
%% 下层函数得到最优化时对应的下层变量
function [level_obj,lever_y, level_funcCount] = SQP(xu,x0,BI)
 options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
 options = optimoptions(options,'Display', 'off');
% options = optimoptions('fmincon','iter','Algorithm','sqp');
%  options = optimset('Display','off');
options.MaxFunctionEvaluations = 1e2;
options.OptimalityTolerance =1e-10;
problem.options = options;
problem.solver = 'fmincon';
problem.objective = @(x)SQP_LLevaluate_F(xu,x,BI);
problem.nonlcon = @unitdisk2;
problem.x0 = x0;
problem.lb = BI.l_lb;
problem.ub = BI.l_ub;
[x,fval,exitflag,output] = fmincon(problem);
level_funcCount                = output.funcCount;
lever_y                        = x;
level_obj                      = fval;
end
function [c,ceq] = unitdisk2(xl)
 global LC
% global SQP_xu 
% global SQP_BI 
% for j = 1 : size(xl,2)
%     xl(:,j) = max(min(xl(:,j),repmat(SQP_BI.l_ub(1,j),size(xl,1),1)),repmat(SQP_BI.l_lb(1,j),size(xl,1),1));
% end
% [F,~,C] = llTestProblem(xl,SQP_BI.fn,SQP_xu);
c = LC';
ceq = [];
end
%% 计算两个上层解决方案之间的距离
function d = computeDistance(matrix1, matrix2)
 
    %Computes pairwise distance between rows of matrix1 and matrix2
    sz1 = size(matrix1, 1);
    sz2 = size(matrix2, 1);
    
    for i = 1:sz1
        for j = 1:sz2
            d(i,j) = sqrt(sum((matrix1(i,:)-matrix2(j,:)).^2));
        end
    end
end