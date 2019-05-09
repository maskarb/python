---------------------------------------------------------------------------
MATLAB package for
Fast computation of kernel estimators
Vikas Raykar, Ramani Duraswami, and Linda Zhao
publised in Journal of Computational and Graphical Statistics.
---------------------------------------------------------------------------

The package contains the MATLAB code for the proposed algorithm for
the fast kernel density derivative estimation and also for the bandwidth 
selection method of Sheather and Jones 1991. The core computation is 
written in C++ with a MATLAB wrapper. This package includes the compiled
dll files for windows platform along with their MATLAB interface. 

To get started peruse example1.m and example2.m

example1 -- Demonstartes the proposed fast method for kernel density estimation
example2 -- Demonstrates the proposed fast method for Sheather-Jones bandwidth selection 

The following are the two main functions

GaussianKernelDensityDerivativeEstimate.m
SheatherJonesBandwidthEstimation.m 

The C++ source code files are also available online at 
http://www.umiacs.umd.edu/~vikas/Software/optimal_bw/optimal_bw_code.html
under the GNU Lesser General Public License. For other OS or if you have problems
with the dlls you will have to recompile the C++ source code files provided

