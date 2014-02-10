function [SelectedVars,Boots,coeffsBestL,coeffarrays,BestL] = fBootGaussLasso(X,Y,nbootstraps,CVruns)
%
%   This is the first version of a BootstrapLasso function
%   Samples n with replacement from n data cases for each bootstrap
%   Uses glmnet and cv from glmnet for optimal lambda
%   Logistic Regression case only
%
%   defaults are for SSVEPAT experiment
%
%   nboots should be odd
n = length(Y);
d = size(X,2);
X = fCenterSphereData(X')';
options             = glmnetSet;

if ~exist('nbootstraps','var')
    nbootstraps = 129;
elseif isempty(nbootstraps);
    nbootstraps = 129;
end
if ~exist('CVruns','var')
    CVruns = 32;
elseif isempty(nbootstraps);
    CVruns = 32;
end
bootndx             = randsample(n,n,'true');
Xb                  = X(bootndx,:);
Yb                  = Y(bootndx,:);
CVfitb              = cvglmnet(Xb,Yb,3,[],'response','gaussian',options,0);
options.lambda      = CVfitb.glmnet_object.lambda;
options.lambda_min  = min(options.lambda);
options.nlambda     = length(options.lambda);
Boots   = struct;
Betas   = zeros(nbootstraps,d,options.nlambda);
BetasIn = zeros(nbootstraps,d,options.nlambda);
BestL   = zeros(nbootstraps,1);
parfor boot = 1:nbootstraps
    bootndx             = randsample(n,n,'true');
    Xb                  = X(bootndx,:);
    Yb                  = Y(bootndx,:);
    CVfitb              = cvglmnet(Xb,Yb,CVruns,[],'response','gaussian',options,0);
    Boots(boot).bootndx = bootndx;
    Boots(boot).BestL   = CVfitb.lambda_min;
    BestL(boot,1)       = CVfitb.lambda_min;
    Boots(boot).fit     = CVfitb.glmnet_object;
    Boots(boot).betasIn = sign(abs(CVfitb.glmnet_object.beta));
end
for boot = 1:nbootstraps
    ngoodl              = size(Boots(boot).fit.beta,2);
    st                  = options.nlambda-ngoodl+1;
    Betas(boot,:,st:end)     = Boots(boot).fit.beta;
    BetasIn(boot,:,st:end)   = Boots(boot).betasIn;
end

coeffarrays.Betas      = Betas;
coeffarrays.BetasIn    = BetasIn;
coeffarrays.BestL      = BestL;
coeffarrays.Lambdas    = options.lambda;
[~,idx] = min(abs(options.lambda-mean(coeffarrays.BestL)));
coeffsBestL      = squeeze(coeffarrays.Betas(:,:,idx));

%% select variables
SelectedVars = round(mean(coeffarrays.BetasIn(:,:,idx),1));
end % function