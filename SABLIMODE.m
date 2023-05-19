function ins = SABLIMODE(BI,groups,groupNum)
global ulFunctionEvaluations;
global llFunctionEvaluations;
global UUFE;
global  best_lx
ulFunctionEvaluations = 0;
llFunctionEvaluations = 0;
%% Parameter setting
Archive = [];
elite = [];
record = [];
Population_model =[];
minN     = 4;
aRate    = 2.6;
MCR = zeros(20*BI.u_dim,1) + 0.2;
MF  = zeros(20*BI.u_dim,1) + 0.2;
k   = 1;
MOP = ones(1,3)/3;
maxIter = ceil(BI.UmaxFEs / BI.u_N);
ImprIter = ceil(BI.UmaxImprFEs / BI.u_N);
best.stop = 0;
cycle    = 0;
% [helper_UX] = helper(BI,1);
%% Generate random population
[Population,level_Archive] = Upper_Initialization(BI,groups,groupNum);
UUFE = 0;
tt=0;
%% init dataset
train_data             = [];
Incre_learning         = [];
[~,init_rank]         = sort(FitnessSingle(Population));
init_Population       = Population(init_rank(1:BI.u_N));
for i = 1:BI.u_N
    train_data{1,i}    = init_Population(i);
end
for iter = 1 : maxIter
    if(best.stop == 0 )
    % Reduce the population size
    N          = ceil((minN-BI.u_N)*UUFE/BI.UmaxFEs) + BI.u_N; 
    [~,rank]   = sort(FitnessSingle(Population));
    Population = Population(rank(1:N));
    Archive    = Archive(randperm(end,min(end,ceil(aRate*N))));
    % Eigencoordinate System
    [~,I]  = sort(Population.objs,'ascend');
    num    = min(N,BI.u_dim);
    TopDec = Population(I(1:num)).decs;
    B      = eig(cov(TopDec));
    R1= deal(zeros(size(B,2)));
    R1(logical(eye(size(B,2)))) = rand(1,size(B,2));
     %% RCES
        % construct train set
         RCPS_Xte            = [];
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
    CR  = repmat(max(0,min(1,CR)),1,BI.u_dim);
    F   = min(1,trnd(1,3*N,1).*sqrt(0.1) + MF(randi(end,3*N,1)));
    while any(F<=0)
        F(F<=0) = min(1,trnd(1,sum(F<=0),1).*sqrt(0.1) + MF(randi(end,sum(F<=0),1)));
    end
    F  = repmat(F,1,BI.u_dim);
    OP = arrayfun(@(S)find(rand<=cumsum(MOP),1),1:3*N);
    OP = arrayfun(@(S)find(OP==S),1:length(MOP),'UniformOutput',false);
    % Generate offspring
    PopDec = Population1.decs;clearvars Population1
    OffDec2 = PopDec;
    OffDec2(OP{1},:) = PopDec(OP{1},:) + F(OP{1},:).*(Xp1(OP{1},:)-PopDec(OP{1},:)+Xr1(OP{1},:)-Xr2(OP{1},:));
    OffDec2(OP{2},:) = PopDec(OP{2},:) + F(OP{2},:)*B*R1*B'.*(Xp1(OP{2},:)-PopDec(OP{2},:)+Xr1(OP{2},:)-Xr3(OP{2},:));
    OffDec2(OP{3},:) = F(OP{3},:)*B*R1*B'.*(Xr1(OP{3},:)+Xp2(OP{3},:)-Xr3(OP{3},:));
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
        [Population_CES]           = Offspring_solution(OffDec(RCES_row(i),:),BI,level_Archive,groups,groupNum);
        if(FitnessSingle(Population(RCES_row(i)))>FitnessSingle(Population_CES))
            Offspring(RCES_row(i))  = Population_CES  ;
        else
            Incre                   = [Population(RCES_row(i)).dec,Population_CES.dec];
            Incre_learning          = [Incre_learning;Incre];
        end
    end
    if( cycle>=50)
        [Offspring] = Offspring_solution(OffDec,BI,level_Archive,groups,groupNum);
    end
%     OffDec
     %  offspring evaluation
    [Population,Offspring,elite] = Cal_solution(Population,Offspring,BI,elite,level_Archive,groups,groupNum);
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
        MCR(k) = 0.5;
        MF(k)  = 0.5;
    end
    delta1 = max(0,delta1./abs(FitnessSingle(Population2)));
    if any(cellfun(@isempty,OP))
        MOP = ones(1,3)/3;
    else
        MOP = cellfun(@(S)mean(delta1(S)),OP);
        MOP = max(0.1,min(0.9,MOP./sum(MOP)));
    end
     clearvars delta1 replace1 Population2
       clearvars pre_lable RCES_pred RCES_replace RCES_row replace RCES_Xte
%     %%更新下层精英库
%     for i =1:N
%         level_Archive_LX(i,:) = [Population(i).dec,Population(i).lx];
%     end
%     level_Archive = [level_Archive;level_Archive_LX];
%     level_Archive = level_Archive(randperm(end,min(end,BI.levelArchive_N)),:);
    Population_model          = [Population,Population_model];
    Builtmodel_data           = [Population_model.decs,Population_model.lxs,Population_model.objs];
    Builtmodel_data_uni        = unique(Builtmodel_data,'rows');
     [~,rank]                  = sort(Builtmodel_data_uni(:,end));
    build_model_data           = Builtmodel_data_uni(rank(1:end),1:end-1);
    level_Archive              = build_model_data;
      %%停止条件
    if isempty(Population.best)
     [~,Best] = min(Population.objs);
         best_lx = Population(Best).lx;
    else
        best_lx =  Population.best.lx;
        tt=tt+1;
        record(tt,1) =  Population.best.obj;
    if (tt>ImprIter && abs(record(tt,1)-record(tt-ImprIter+1,1)) <  1e-7)
%          best.stop = 1;
    end
    end
    if isempty(Population.best)
    else
        reachTheTarget_a = abs(Population.best.obj-BI.u_fopt)< 1e-7;
        reachTheTarget_b = abs(Population.best.lf-BI.l_fopt)< 1e-7 ;
        if(reachTheTarget_a)&&(reachTheTarget_b)
            best.stop = 1;
        end
    end
    else
    end
end
if isempty(Population.best)
    [~,Best] = min(Population.objs);
    ins.UF = Population(Best).obj;
    ins.LF = Population(Best).lf;
    ins.UX = Population(Best).dec;
    ins.LX = Population(Best).lx;
    ins.UFEs = ulFunctionEvaluations;
    ins.LFEs = llFunctionEvaluations;
else
    ins.UF = Population.best.obj;
    ins.LF = Population.best.lf;
    ins.UX = Population.best.dec;
    ins.LX = Population.best.lx;
    ins.UFEs = ulFunctionEvaluations;
    ins.LFEs = llFunctionEvaluations;
end
end
%%产生子代
function[Offspring] = Offspring_solution(PopDec,BI,level_Archive,groups,groupNum)
for i = 1 : size(PopDec,1)
    [bestLX(i,:),bestLF(i,:),bestLC(i,:)] = lowerLevelSearch(PopDec(i,:),level_Archive,BI,groups,groupNum);
     [F(i,:),C(i,:),UX(i,:),LX(i,:)]= UL_evaluate(PopDec(i,:),bestLX(i,:),BI);  
end
C=[C,bestLC];
Offspring = SOLUTION(BI,UX,F,C,LX,bestLF);
end
%%父代子代于最优解比较
function[Population,Offspring,elite] = Cal_solution(Population,Offspring,BI,elite,level_Archive,groups,groupNum)
if (isempty(Offspring.best))||(isempty(Population.best))
else
    if isempty(elite)
        if isempty(Offspring.best)&&(isempty(Population.best)==0)
            elite       = Population.best;
        end
        if (isempty(Offspring.best)==0)&&isempty(Population.best)
            elite       = Offspring.best;
        end
        if (isempty(Offspring.best)==0)&&(isempty(Population.best)==0)
            if(Offspring.best.obj<Population.best.obj)
                elite       = Offspring.best;
            else
                elite       = Population.best;
            end
        end
        elite     = refine(elite,BI,level_Archive,groups,groupNum) ;
    else
        if upperLevelComparator(Offspring,elite)
            RF_idx = find(Offspring.objs<elite.obj);
            for i=1:size(RF_idx,1)
                a = refine(Offspring(RF_idx(i)),BI,level_Archive,groups,groupNum);
                Offspring(RF_idx(i)).obj =a.obj;
            end
            if upperLevelComparator(Offspring,elite)
                elite = Offspring.best;
            end
        end
        if upperLevelComparator(Population,elite)
            RF_idx = find(Population.objs<elite.obj);
            for i=1:size(RF_idx,1)
                a = refine(Population(RF_idx(i)),BI,level_Archive,groups,groupNum);
                Population(RF_idx(i)).obj =a.obj;
            end
            if upperLevelComparator(Population,elite)
                elite = Population.best;
            end
        end
        if rand>0.5
            elite = refine(elite,BI,level_Archive,groups,groupNum);
        end
    end
end
end

%%初始化种群
function[Population,level_Archive] = Upper_Initialization(BI,groups,groupNum)
PopDec1 = unifrnd(repmat(BI.u_lb,BI.u_N,1),repmat(BI.u_ub,BI.u_N,1));
% PopDec2 = help_dec;
% level_Archive = unifrnd(repmat(BI.l_lb, BI.levelArchive_N,1),repmat(BI.l_ub, BI.levelArchive_N,1));
level_Archive = [];
for i = 1 : BI.u_N
    [bestLX(i,:),bestLF(i,:),bestLC(i,:)] = lowerLevelSearch(PopDec1(i,:),level_Archive,BI,groups,groupNum);
     [F(i,:),C(i,:),UX(i,:),LX(i,:)]= UL_evaluate(PopDec1(i,:),bestLX(i,:),BI);  
end
C=[C,bestLC];
Population = SOLUTION(BI,UX,F,C,LX,bestLF);
for i =1:BI.u_N
level_Archive_LX(i,:) = [Population(i).dec,Population(i).lx];
end
level_Archive = [level_Archive;level_Archive_LX];
level_Archive = level_Archive(randperm(end,min(end,BI.levelArchive_N)),:);
end

%%
function [F,C,UX,LX] = UL_evaluate(UPop,LPOP,BI)
for j = 1 : size(UPop,2)
    UPop(:,j) = max(min(UPop(:,j),repmat(BI.u_ub(1,j),size(UPop,1),1)),repmat(BI.u_lb(1,j),size(UPop,1),1));
end
for j = 1 : size(LPOP,2)
    LPOP(:,j) = max(min(LPOP(:,j),repmat(BI.l_ub(1,j),size(LPOP,1),1)),repmat(BI.l_lb(1,j),size(LPOP,1),1));
end
UX=UPop;
LX=LPOP;
[F,~,C] = ulTestProblem(UPop, LPOP, BI.fn);
C = sum(max(0,C));
end
function [isNoWorseThan] = upperLevelComparator(P,Q)
if isempty(P)
    isNoWorseThan = flase;
else
    isNoWorseThan = P.best.obj <= Q.best.obj;
end
end
function Q = refine(P,BI,level_Archive,groups,groupNum)
Q = P;
[R.LX,Q.lf,R.LC] = lowerLevelSearch(Q.dec,level_Archive,BI,groups,groupNum);
 if lowerLevelComparator(Q,P)&&(R.LC<=0)
    [Q.obj,~,Q.dec,Q.lx]= UL_evaluate(Q.dec,R.LX,BI);
else
    Q = P;
 end
end
function isNoWorseThan = lowerLevelComparator(Q,P)
    isNoWorseThan = Q.lf <= P.lf;
end