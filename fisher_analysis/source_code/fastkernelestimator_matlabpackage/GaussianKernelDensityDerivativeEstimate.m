function [kde,time_taken]=GaussianKernelDensityDerivativeEstimate(X,Y,h,r,method,epsil)
% Fast univariate (Gaussian) kernel density derivative estimation
%--------------------------------------------------------------------------
% REFERENCE 
%--------------------------------------------------------------------------
% Fast Computation of Kernel Estimators
% Vikas Raykar, Ramani Duraiswami, and Linda Zhao
% Journal of COmputational and Graphical Statistics
%--------------------------------------------------------------------------
% USAGE [See example1.m]
%--------------------------------------------------------------------------
% [kde,time_taken]=GaussianKernelDensityDerivativeEstimate(X,Y);
% [kde,time_taken]=GaussianKernelDensityDerivativeEstimate(X,Y,0.2);
% [kde,time_taken]=GaussianKernelDensityDerivativeEstimate(X,Y,0.2,1);
% [kde,time_taken]=GaussianKernelDensityDerivativeEstimate(X,Y,0.2,1,'direct');
% [kde,time_taken]=GaussianKernelDensityDerivativeEstimate(X,Y,0.2,0,'fast',1e-3);
%--------------------------------------------------------------------------
% INPUT
%--------------------------------------------------------------------------
% X                 --> 1 x N vector of N sample points.
% Y                 --> 1 x M vector of M evaluation points.
% h                 --> the bandwidth of the Gaussian kernel.
%                       [DEFAULTS to the normal reference rule]
% r                 --> the derivative order (r=0 corresponds to KDE).
%                       [DEFAULTS to 0]
% method            --> can be 'direct' naive direct implementation 
%                              'fast'   the proposed fast method 
%                       [DEFAULTS to 'fast']
% epsil             --> the desired accuracy for the proposed fast method.
%                       [DEFAULTS to 1e-6]  
%--------------------------------------------------------------------------
% OUTPUT
%--------------------------------------------------------------------------
% kde                --> 1 x M vector of the density derivative
%                          evaluated at  each evaluaiton point.
% time_taken         --> time taken in seconds to compute the estimate
%--------------------------------------------------------------------------
% SIGNATURE
%--------------------------------------------------------------------------
% Author: Vikas Chandrakant Raykar
% E-Mail: vikasraykar@gmail.com
%--------------------------------------------------------------------------
% SEE ALSO
%--------------------------------------------------------------------------
% UnivariateDensityDerivative
% FastUnivariateDensityDerivative
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Set default values
%--------------------------------------------------------------------------

if ~exist('r','var')
    r=0;
end

if ~exist('method','var')
    method='fast';
end

if strcmp(method,'fast')==1
    if ~exist('epsil','var')
        epsil=1e-6;
    end
end

if ~exist('h','var')
    h=std(X)*((4/(3*length(X)))^(1/5));
end

%--------------------------------------------------------------------------
% Centering and Scaling the data
%--------------------------------------------------------------------------

min_x=min(X); min_y=min(Y); shift=min(min_x, min_y); 

X_shifted=X-shift; Y_shifted=Y-shift;

max_x=max(X_shifted); max_y=max(Y_shifted); scale=1/max(max_x,max_y); 

X_shifted_scaled=X_shifted*scale; Y_shifted_scaled=Y_shifted*scale;

h_scaled=h*scale;

N=length(X_shifted_scaled);
M=length(Y_shifted_scaled);

%--------------------------------------------------------------------------

if strcmp(method,'direct')==1
    to=clock;    
    [kde]=UnivariateDensityDerivative(N,M,X_shifted_scaled,Y_shifted_scaled,h_scaled,r);
    time_taken=etime(clock,to);
end

%--------------------------------------------------------------------------


if strcmp(method,'fast')==1
    Q=1/(sqrt(2*pi)*(h^(r+1)));
    epsil=epsil/Q;
    to=clock;    
    [kde]=FastUnivariateDensityDerivative(N,M,X_shifted_scaled,Y_shifted_scaled,h_scaled,r,epsil);
    time_taken=etime(clock,to);
end

%--------------------------------------------------------------------------

kde=(scale^(r+1))*kde;

return;