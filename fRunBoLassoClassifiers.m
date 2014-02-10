function  out = fRunBoLassoClassifiers(X,Y)

%% options
ntrees      = 75;
nboots      = 129;
nxvalLasso  = 3;

options = statset('UseParallel','always');

Loptions             = glmnetSet;
Loptions.maxit       = 500;

%% preprocessing
X = fCenterSphereData(X')';
N = length(Y);
D = size(X,2);
[OutlierNDX] = fkNNprototyping(X,X,round(sqrt(N)),'linear');
Info.Outliers.N = sum(OutlierNDX);
Info.Outliers.OutlierNDX = OutlierNDX;
Info.Outliers.KeepedData = ~OutlierNDX;
X = X(~OutlierNDX,:);
Y = Y(~OutlierNDX,:);
N = length(Y);
NBlocks = min(21,max(round(sqrt(N)),4));
[RBI,NBlocks] = fGenXvalBlockIndex(N,NBlocks);
BoLasso.coeffs = [];
%% Correct logs
Lasso.correctlog        = [zeros(N,1) (1:N)'];
BoLasso.GLM.correctlog  = [zeros(N,1) (1:N)'];
BoLasso.LDA.correctlog  = [zeros(N,1) (1:N)'];
BoLasso.QDA.correctlog  = [zeros(N,1) (1:N)'];
BoLasso.SVM.correctlog  = [zeros(N,1) (1:N)'];
SVM.correctlog          = [zeros(N,1) (1:N)'];
RF.correctlog           = [zeros(N,1) (1:N)'];

Info.Xval.NBlocks = NBlocks;
Info.Xval.Index   = RBI;

for xval = 1:NBlocks
    %% parse
    trIDX = RBI~=xval;
    teIDX = RBI==xval;
    Xtr = X(trIDX,:);Xte = X(teIDX,:);
    Ytr = Y(trIDX);Yte = Y(teIDX);
    
    Info.Xval.Tr(xval).TrIDX = trIDX;
    Info.Xval.Te(xval).TeIDX = teIDX;
    %% Lasso
    Lassofit            = cvglmnet(Xtr,Ytr,nxvalLasso,[],'response','binomial',Loptions,0);
    Ypred               = glmnetPredict(Lassofit.glmnet_object,'response',Xte,Lassofit.lambda_min);
    Lasso.mse(xval)     = mean(((Yte-1)-Ypred).^2,1);
    Lasso.err(xval)   	= mean((Yte-1)~=round(Ypred),1);
    Lasso.correctlog(teIDX,1) = (Yte-1)==round(Ypred);
    Lasso.YteYp(xval).YteYp   = [Yte Ypred];
    
    %% BoLasso    
    for ii = 1:5
        [BoLasso.Vars(xval).SelectedVars,...
            BoLasso.Vars(xval).Boots,...
            BoLasso.Vars(xval).coeffsBestL,...
            BoLasso.Vars(xval).coeffarrays,...
            BoLasso.Vars(xval).BestL,...
            BoLasso.Vars(xval).MSE] = fBootLogisticLasso(Xtr,Ytr,nboots,nxvalLasso);
        xvsum = sum(BoLasso.Vars(xval).SelectedVars);
        if xvsum > 0
            break;
        else
            disp('BoLasso failed... Retrying');
        end
    end %    
    SelectedVars = logical(BoLasso.Vars(xval).SelectedVars);
    Ztr = Xtr(:,SelectedVars);
    Zte = Xte(:,SelectedVars);
    % Refit and eval
    % eval
    fit                                     = glmfit(Ztr,Ytr-1,'binomial');
    BoLasso.GLM.glmfit(xval).fit    = fit;
    Ypred                                   = glmval(fit,Zte,'logit');
    BoLasso.GLM.mse(xval)           = mean(((Yte-1)-Ypred).^2,1);    % mse
    BoLasso.GLM.err(xval)           = mean((Yte-1)~=round(Ypred),1); % perc err 0/1 loss
    BoLasso.GLM.correctlog(teIDX,1) = (Yte-1)==round(Ypred);
    BoLasso.GLM.Posterior(xval).Posterior = Ypred;
    BoLasso.GLM.YteYp(xval).YteYp   = [Yte Ypred];    
    % lDA
    [Ypred,~,Ypost]             = classify(Zte,Ztr,Ytr,'linear');
    BoLasso.LDA.Posterior(xval).Posterior = Ypost;
    BoLasso.LDA.mse(xval)       = mean(((Yte-1)-Ypost(:,2)).^2,1);    % mse
    BoLasso.LDA.err(xval)       = mean(Yte~=Ypred);
    BoLasso.LDA.correctlog(teIDX,1) = Yte==Ypred;
    BoLasso.LDA.YteYp(xval).YteYp   = [Yte Ypost(:,2)];
    
    % QDA
    [Ypred,~,Ypost]                         = classify(Zte,Ztr,Ytr,'quadratic');
    BoLasso.QDA.Posterior(xval).Posterior   = Ypost;
    BoLasso.QDA.mse(xval)                   = mean(((Yte-1)-Ypost(:,2)).^2,1);    % mse
    BoLasso.QDA.err(xval)                   = mean(Yte~=Ypred);
    BoLasso.QDA.correctlog(teIDX,1)         = Yte==Ypred;
    BoLasso.QDA.YteYp(xval).YteYp           = [Yte Ypost(:,2)];
    
    % SVM    
    Model                               = train(Ytr,sparse(Ztr),'-s 1 -q');
    [Ypred,~,Ypost]                     = predict(Yte,sparse(Zte),Model,'-b -q');
    BoLasso.SVM.Posterior(xval).Posterior = normcdf(Ypost);
    BoLasso.SVM.err(xval)               = mean(Ypred~=Yte);
    BoLasso.SVM.mse(xval)               = mean((normcdf(Ypost)-(Yte-1)).^2);
    BoLasso.SVM.YteYp(xval).YteYp       = [Yte normcdf(Ypost)];
    BoLasso.SVM.correctlog(teIDX,1)     = Yte==Ypred;
    
    %% non-BoLasso
    % SVM-Linear
    Model                               = train(Ytr,sparse(Xtr),'-s 1 -q');
    [Ypred,~,Ypost]                     = predict(Yte,sparse(Xte),Model,'-b -q');
    SVM.Posterior(xval).Posterior       = normcdf(Ypost);
    SVM.err(xval)               = mean(Ypred~=Yte);
    SVM.mse(xval)               = mean((normcdf(Ypost)-(Yte-1)).^2);
    SVM.YteYp(xval).YteYp       = [Yte normcdf(Ypost)];
    SVM.correctlog(teIDX,1)     = Yte==Ypred;
    % RF
    Model                       = TreeBagger(ntrees,Xtr,Ytr,'method','classification','OOBVarImp','on','Options',options);
    [~,Ypost]             = predict(Model,Xte);
    [~,Ypred]                   = max(Ypost,[],2); % annoyingly outputs winning class labels as cells, but we can get it from the probability outputs
    Rf.ProbOuts(xval).Posterior = Ypost;
    RF.err(xval)                = mean(Yte~=Ypred);
    RF.mse(xval)                = mean(((Yte-1)-Ypost(:,2)).^2,1);
    RF.YteYp(xval).YteYp       = [Yte Ypost(:,2)];
    
    
    %% store
end % xvalidation

out.Lasso   = Lasso;
out.BoLasso = BoLasso;
out.SVM     = SVM;
out.RF      = RF;
out.Info    = Info;

end % function