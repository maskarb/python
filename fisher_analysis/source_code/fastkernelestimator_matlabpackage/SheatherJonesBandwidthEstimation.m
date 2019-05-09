function [h]=SheatherJonesBandwidthEstimation(X,method,epsil)
%  AMISE-optimal bandwidth estimation for univariate kernel density estimation. 
%  [Shether Jones Solve-the-equation plug-in method.]
%--------------------------------------------------------------------------
% REFERENCE 
%--------------------------------------------------------------------------
% S.J. Sheather and M.C. Jones. 'A reliable data-based
% bandwidth selection method for kernel density estimation'
% J. Royal Statist. Soc. B, 53:683-690, 1991
%
% Fast Computation of Kernel Estimators
% Vikas Raykar, Ramani Duraiswami, and Linda Zhao
% Journal of Computational and Graphical Statistics
%--------------------------------------------------------------------------
% USAGE [see example2.m]
%--------------------------------------------------------------------------
% [h]=SheatherJonesBandwidthEstimation(X)
% [h]=SheatherJonesBandwidthEstimation(X,'direct')
% [h]=SheatherJonesBandwidthEstimation(X,'fast')
% [h]=SheatherJonesBandwidthEstimation(X,'fast',1e-6)
%--------------------------------------------------------------------------
% INPUT
%--------------------------------------------------------------------------
% X                 --> 1 x N vector of N sample points.
% method            --> can be 'direct' naive direct implementation 
%                              'fast'   the proposed fast method 
%                       [DEFAULTS to 'fast']   
% epsil             --> the desired accuracy for the proposed fast method.
%                       [DEFAULTS to 1e-3]   
%--------------------------------------------------------------------------
% OUTPUT
%--------------------------------------------------------------------------
% h                 --> the optimal bandiwdth estimated using the
%                       Shether Jones Solve-the-equation plug-in method.
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
% h_function
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
% Set default values
%--------------------------------------------------------------------------

if ~exist('method','var')
    method='fast';
end

if strcmp(method,'fast')==1
    if ~exist('epsil','var')
        epsil=1e-3;
    end
    data.epsil=epsil;
end

N=length(X);

%--------------------------------------------------------------------------
% Scale the data to lie in the unit interval [0 1]
%--------------------------------------------------------------------------

shift=min(X); X_shifted=X-shift;
scale=1/(max(X_shifted)); X_shifted_scaled=X_shifted*scale;

%--------------------------------------------------------------------------
% Compute an estimate of the standard deriviation of the data
%--------------------------------------------------------------------------

sigma=std(X_shifted_scaled);

%--------------------------------------------------------------------------
% Estimate the density functionals ${\Phi}_6$ and ${\Phi}_8$ using the normal scale rule.
%--------------------------------------------------------------------------

phi6=(-15/(16*sqrt(pi)))*(sigma^(-7));
phi8=(105/(32*sqrt(pi)))*(sigma^(-9));

%--------------------------------------------------------------------------
% Estimate the density functionals ${\Phi}_4$ and ${\Phi}_6$ using the kernel density
% estimators with the optimal bandwidth based on the asymptotic MSE.
%--------------------------------------------------------------------------

g1=(-6/(sqrt(2*pi)*phi6*N))^(1/7);
g2=(30/(sqrt(2*pi)*phi8*N))^(1/9);

if strcmp(method,'direct')==1
    [D4]=UnivariateDensityDerivative(N,N,X_shifted_scaled,X_shifted_scaled,g1,4);
    phi4=sum(D4)/(N-1);
    
    [D6]=UnivariateDensityDerivative(N,N,X_shifted_scaled,X_shifted_scaled,g2,6);
    phi6=sum(D6)/(N-1);
end

if strcmp(method,'fast')==1
    [D4]=FastUnivariateDensityDerivative(N,N,X_shifted_scaled,X_shifted_scaled,g1,4,epsil);
    phi4=sum(D4)/(N-1);
    
    [D6]=FastUnivariateDensityDerivative(N,N,X_shifted_scaled,X_shifted_scaled,g2,6,epsil);
    phi6=sum(D6)/(N-1);
end

%--------------------------------------------------------------------------
% The bandwidth is the solution to the following  equation.
%--------------------------------------------------------------------------

constant1=(1/(2*sqrt(pi)*N))^(1/5);
constant2=(-6*sqrt(2)*phi4/phi6)^(1/7);

h_initial=constant1*phi4^(-1/5);

options = optimset('Display','off','TolFun',1e-14,'TolX',1e-14,'LargeScale','on');
data.N=N;
data.X=X_shifted_scaled;
data.constant1=constant1;
data.constant2=constant2;


if strcmp(method,'direct')==1
    [h,resnorm,residual,exitflag,output] = lsqnonlin('h_function',h_initial,[0],[],options,data) ;
end

if strcmp(method,'fast')==1
    [h,resnorm,residual,exitflag,output] = lsqnonlin('fast_h_function',h_initial,[0],[],options,data) ;
end

h=h/scale;

if exitflag>0    disp('The function converged to a solution.'); end
if exitflag==0   disp('The maximum number of function evaluations or iterations was exceeded.'); end
if exitflag<0    disp('The function did not converge to a solution.'); end


return

function F = h_function(h,data)

lambda=data.constant2*h^(5/7);

[D4]=UnivariateDensityDerivative(data.N,data.N,data.X,data.X,lambda,4);
phi4=sum(D4)/(data.N-1);

F=h-data.constant1*phi4^(-1/5);

return

function F = fast_h_function(h,data)

lambda=data.constant2*h^(5/7);
[D4]=FastUnivariateDensityDerivative(data.N,data.N,data.X,data.X,lambda,4,data.epsil);
phi4=sum(D4)/(data.N-1);
F=h-data.constant1*phi4^(-1/5);

return
