//-------------------------------------------------------------------
// The code was written by Vikas C. Raykar 
// and is copyrighted under the Lessr GPL: 
//
// Copyright (C) 2006 Vikas C. Raykar
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; version 2.1 or later.
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
// See the GNU Lesser General Public License for more details. 
// You should have received a copy of the GNU Lesser General Public
// License along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, 
// MA 02111-1307, USA.  
//
// The author may be contacted via email at: vikas(at)cs(.)umd(.)edu 
//-------------------------------------------------------------------

//-------------------------------------------------------------
// File    : UnivariateDensityDerivative.cpp
// Purpose : Implementation for for UnivariateDensityDerivative
// Author  : Vikas C. Raykar (vikas@cs.umd.edu)
// Date    : September 17, 2005
//-------------------------------------------------------------

#include <math.h>
#include <boost/python.hpp>
namespace python = boost::python;

//-------------------------------------------------------------------
// Constructor.
//
// PURPOSE                                                    
// -------   
// Initialize the class. 
// Read the parameters.
//
// PARAMETERS                                                      
// ----------
// NSources		      --> number of sources, N.
// MTargets		      --> number of targets, M.
// pSources		      --> pointer to sources, px(N).
// pTargets             --> pointer to the targets, py(M).
// Bandwidth		--> the source bandwidth, h.
// Order                --> order of the derivative, r.  
// pDensityDerivative   --> pointer the the evaluated Density 
//-------------------------------------------------------------------
class UnivariateDensityDerivative {
	public:
	// constructor:
	UnivariateDensityDerivative(
		int NSources,
		int MTargets,
		python::list pSources,
		python::list pTargets,
		double Bandwidth,
		int Order);

		python::list pDensityDerivative;

	//destructor
	~UnivariateDensityDerivative();

	void evaluate();
	double hermite(double x, int r);

	private:
		int N; //number of sources.
		int M; //number of targets.
		double* px; //pointer to sources, (N).
		double* py; //pointer to the targets, (M).
		double h; //the source bandwidth.
		int r; //the rth density derivative.

		double* pD; //pointer to the evaluated Density Derivative, (M).

		double* converter_double(python::list lis);
		python::list converter_list(double* lis);

		double two_h_square;
		double pi;
		double q;
};

UnivariateDensityDerivative::UnivariateDensityDerivative(
            int NSources,
			int MTargets,
            python::list pSources,
            python::list pTargets,
			double Bandwidth,
			int Order)
{	

	N = NSources;
	M = MTargets;
	px = converter_double(pSources);
	h = Bandwidth;
	r = Order;
	py = converter_double(pTargets);
	pD = new double[N];

	two_h_square=2*h*h;
	pi=3.14159265358979;
	q=(pow(-1,r))/(sqrt(2*pi)*N*(pow(h,(r+1))));

}

double* UnivariateDensityDerivative::converter_double(python::list lis)
{
    int length = len(lis);
    double* temp = new double[length];
    for (int i = 0; i < length; i++)
    {
        temp[i] = python::extract<double>(lis[i]);
    }
    return temp;
}

python::list UnivariateDensityDerivative::converter_list(double* lis)
{
    python::list temp;
    for (int i = 0; i < N; i++)
    {
        temp.append(lis[i]);
    }
    return temp;
}
//-------------------------------------------------------------------
// Destructor.
//-------------------------------------------------------------------

UnivariateDensityDerivative::~UnivariateDensityDerivative()
{
}

//-------------------------------------------------------------------
// Actual function to evaluate the Univariate Density Derivative.
//-------------------------------------------------------------------

void
UnivariateDensityDerivative::evaluate()
{
	for(int j=0; j<M; j++)
	{
		pD[j]=0.0;

		for(int i=0; i<N; i++)
		{
			double temp=py[j]-px[i];
			double norm=temp*temp;
			
			pD[j] = pD[j]+(hermite(temp/h,r)*exp(-norm/two_h_square));			

		}
		pD[j]=pD[j]*q;
	}
	pDensityDerivative = converter_list(pD);
}

//-------------------------------------------------------------------
// Recursive implementation of the Hermite polynomial.
//-------------------------------------------------------------------

double
UnivariateDensityDerivative::hermite(double x, int r)
{
	if(r==0)
	{
		return (1.0);
	}
	else if(r==1)
	{
		return (x);
	}
	else
	{
		return (x*hermite(x,r-1))-((r-1)*hermite(x,r-2));
	}

}

BOOST_PYTHON_MODULE(fast_slow_deriv)
{
    namespace python = boost::python;
    python::class_<UnivariateDensityDerivative>("UnivariateDensityDerivative", 
        python::init<int, 
                     int, 
                     python::list,
                     python::list,
                     double,
                     int
                    >())
        .def("evaluate", &UnivariateDensityDerivative::evaluate)
        .def_readonly("pD", &UnivariateDensityDerivative::pDensityDerivative)
    ;

}