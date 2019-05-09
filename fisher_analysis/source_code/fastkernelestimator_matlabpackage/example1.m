clear all;
close all;
clear functions;
clc;

disp('---------------------------------------------');
disp(sprintf('Example to demonstrate the proposed fast method.'));
disp('---------------------------------------------');

% sample normal mixture density of Marron and Wand [ can vary form 1-15]

density_number=6;

% number of datapoints

N=5000;

disp('---------------------------------------------');
disp(sprintf('Sampling %d points from a sample normal mixture density %d of Marron and Wand',N,density_number));
disp('---------------------------------------------');

Y=[-5.0:0.005:5.0];
[actual_density,X]=marron_wand_normal_mixtures(density_number,Y,N);

figure;
plot(Y,actual_density,'k'); hold on;
plot(X,0,'k+');
title(sprintf('N=%d points sample from the Marron-Wand normal mixture %d.',N,density_number));
legend('Actual density','Sample points');

input('Press any key to continue');

h=std(X)*((4/(3*N))^(1/5));

disp('---------------------------------------------');
disp(sprintf('Estimating the bandwidth using normal reference rule h=%f',h));
disp('---------------------------------------------');

input('Press any key to continue');

M=length(Y);

disp('---------------------------------------------');
disp(sprintf('Computing the Kernel Density Estimate at %d points--Direct Slow Implementation.',M));
disp('---------------------------------------------');

r=0;
method='direct';
epsil=1e-3;
[kde_direct,time_taken_direct]=GaussianKernelDensityDerivativeEstimate(X,Y,h,r,method,epsil);

disp(sprintf('Time taken=%f seconds',time_taken_direct));

input('Press any key to continue');

disp('---------------------------------------------');
disp(sprintf('Computing the Kernel Density Estimate at %d points--Proposed Fast method with epsilon=%e',M,epsil));
disp('---------------------------------------------');

r=0;
method='fast';
epsil=1e-3;
[kde_fast,time_taken_fast]=GaussianKernelDensityDerivativeEstimate(X,Y,h,r,method,epsil);

disp(sprintf('Time taken=%f seconds',time_taken_fast));

speedup=time_taken_direct/time_taken_fast;
max_abs_error=max(abs(kde_direct-kde_fast));

disp('---------------------------------------------');
disp(sprintf('Speedup=%f Maximum absolute error=%e ',speedup,max_abs_error));
disp('---------------------------------------------');

close all;

figure;
plot(Y,actual_density,'k'); hold on;
plot(Y,kde_direct,'b'); hold on;
plot(Y,kde_fast,'r'); hold on;
plot(X,0,'k+');
title(sprintf('N=%d points sample from the Marron-Wand normal mixture %d.',N,density_number));
legend('Actual density','KDE Direct method','KDE Proposed fast method','Sample points');


