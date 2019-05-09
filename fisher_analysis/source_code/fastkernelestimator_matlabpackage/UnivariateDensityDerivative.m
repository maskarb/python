function [D]=UnivariateDensityDerivative(N,M,X,Y,h,r)
% Direct implementation of the r^{th} kernel density derivative
% estimate based on the Gaussian kernel.
%
% C++ Implementation.
% Loads UnivariateDensityDerivative.dll
%--------------------------------------------------------------------------
% Author: Vikas Chandrakant Raykar
% E-Mail: vikasraykar@gmail.com
%--------------------------------------------------------------------------
% INPUTS
%--------------------------------------------------------------------------
% N                 --> number of source points.
% M                 --> number of target points.
% X                 --> 1 x N matrix of N source points.
% Y                 --> 1 x M matrix of M target points.
% h                 --> source bandwidth or scale.
% r                 --> derivative order.
%--------------------------------------------------------------------------
% OUTPUTS
%--------------------------------------------------------------------------
% D                --> 1 x M vector of the density derivative
%                      evaluated at  each target point.
%--------------------------------------------------------------------------

