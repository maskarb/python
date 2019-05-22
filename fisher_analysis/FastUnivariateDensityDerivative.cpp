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
// File    : FastUnivariateDensityDerivative.cpp
// Purpose : Implementation for for FastUnivariateDensityDerivative
// Author  : Vikas C. Raykar (vikas@cs.umd.edu)
// Date    : September 17, 2005
//-------------------------------------------------------------

// #include "FastUnivariateDensityDerivative.h"
#include <math.h>
#include <boost/python.hpp>

#define  min(a,b) (((a)<(b))?(a):(b)) 
#define  max(a,b) (((a)>(b))?(a):(b)) 
#define  P_UL 500
#define  R 1.0

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



class FastUnivariateDensityDerivative{
	public:
		//constructor 
		FastUnivariateDensityDerivative(int NSources,
			int MTargets,
			double *pSources,
			double *pTargets,
			double Bandwidth,
		    int Order,
			double epsilon,
			double *pDensityDerivative);

		//destructor
		~FastUnivariateDensityDerivative();

		//function to evaluate the Density Derivative
		void Evaluate();

		//function to evaluate the Hermite polynomial.
		double hermite(double x, int r);

	private:
		int N;				//number of sources.
		int M;				//number of targets.
		double *px;			//pointer to sources, (N).
		double *py;		    //pointer to the targets, (M).
		double  h;			//the source bandwidth.
		int r;              //the rth density derivative.
		double eps;         //the desired error
		double *pD;         //pointer the the evaluated Density Derivative, (M).

		double rx;
		double rr;
		double ry;
		int K;
		int p;
		double h_square;
		double two_h_square;

		double *pClusterCenter;
		int *pClusterIndex;

		int num_of_a_terms;
		double *a_terms;

		int num_of_B_terms;
		double *B_terms;

		double pi;
		double q;

		
		//MATLAB applications should always call mxMalloc rather than malloc to allocate memory

		void *operator new[] (size_t s){ return malloc(s);}
	    void operator delete[] (void* mem){ free(mem);}

		int factorial(int n);
		void choose_parameters();
		void space_sub_division();
		void compute_a();
		void compute_B();


};

FastUnivariateDensityDerivative::FastUnivariateDensityDerivative(int NSources,
			int MTargets,
			double *pSources,
			double *pTargets,
			double Bandwidth,
		    int Order,
			double epsilon,
			double *pDensityDerivative)
{	
	// Read the arguments.

	N=NSources;
	M=MTargets;
	px=pSources;
	h=Bandwidth;
	r=Order;
	py=pTargets;
	pD=pDensityDerivative;
	eps=epsilon;

	
	h_square=h*h;
	two_h_square=2*h_square;

	pi=3.14159265358979;
	q=(pow(-1,r))/(sqrt(2*pi)*N*(pow(h,(r+1))));
	//printf("q=%f \n",q);

	
	// Choose the parameters for the algorithm.
	choose_parameters();

	// Space sub-division
	space_sub_division();

	// Compute the constant a
	compute_a();

	// Compute the constant B
	compute_B();

}

//-------------------------------------------------------------------
// Destructor.
//-------------------------------------------------------------------

FastUnivariateDensityDerivative::~FastUnivariateDensityDerivative()
{
	delete []a_terms;
	delete []B_terms;
}




//-------------------------------------------------------------------
// Compute the factorial.
//-------------------------------------------------------------------

int
FastUnivariateDensityDerivative::factorial(int n){
	
	int fact=1;

	for ( int i = 1; i <= n; i++){
		fact=fact*i;
	}

	return fact;
}

//-------------------------------------------------------------------
// Choose the parameters
// 1. rx --> interval length.
// 2. K  --> number of intervals.
// 3. rr --> cutoff radius.
// 4. ry --> cluster cutoff radius.
// 5. p  --> truncation number.
//-------------------------------------------------------------------

void
FastUnivariateDensityDerivative::choose_parameters(){

	// 1. rx --> interval length.

	rx=h/2;

	// 2. K  --> number of intervals.

	K=(int)ceil(1.0/rx);
	rx=1.0/K;
	double rx_square=rx*rx;

	// 3. rr --> cutoff radius.

    //double r_term=pow(2.0,r/2)*sqrt((double)factorial(r));
	double r_term=sqrt((double)factorial(r));

	rr=min(R,2*h*sqrt(log(r_term/eps)));

	// 4. ry --> cluster cutoff radius.
	ry=rx+rr;

	// 5. p  --> truncation number.

	p=0;
	double error=1;
	double temp=1;
	double comp_eps=eps/r_term;
		
	while((error > comp_eps) & (p <= P_UL)){
		p++;
		double b=min(((rx+sqrt((rx_square)+(8*p*h_square)))/2),ry);
		double c=rx-b;
		temp=temp*(((rx*b)/h_square)/p);
		error=temp*(exp(-(c*c)/2*two_h_square));			
	}	
	p=p+1;



	//printf("h=%f r=%d eps=%f K=%d rx=%f rr=%f ry=%f p=%d\n",h,r,eps,K,rx,rr,ry,p);

}


//-------------------------------------------------------------------
// Space subdivision
//-------------------------------------------------------------------

void
FastUnivariateDensityDerivative::space_sub_division(){

	// 1. Cluster Centers

	pClusterCenter=new double[K];
	for(int i=0; i<K; i++){
		pClusterCenter[i]=(i*rx)+(rx/2);
		//printf("%f\n",pClusterCenter[i]);
	}

	//2. Allocate each source to the corresponding interval

    pClusterIndex=new int[N];
	for(int i=0; i<N; i++){
		pClusterIndex[i]=min((int)floor(px[i]/rx),K-1);
		//printf("x=%f Cluster=%d\n",px[i],pClusterIndex[i]);
	}

}

//-------------------------------------------------------------------
// Compute the contant term a_{lm}.
// l=0...floor(r/2)
// m=0...r-2l 
//-------------------------------------------------------------------

void
FastUnivariateDensityDerivative::compute_a(){

	double r_factorial=(double)factorial(r);
	//printf("%f \n",r_factorial);

    double *l_constant;
	l_constant=new double[((int)floor((double)r/2))+1];
	l_constant[0]=1;
	for(int l=1; l <= (int)floor((double)r/2); l++){
		l_constant[l]=l_constant[l-1]*(-1.0/(2*l));
		//printf("%f \n",l_constant[l]);
	}

	double *m_constant;
	m_constant=new double[r+1];
	m_constant[0]=1;
	for(int m=1; m <= r; m++){
		m_constant[m]=m_constant[m-1]*(-1.0/m);
		//printf("%f \n",m_constant[m]);
	}

	num_of_a_terms=0;
	for(int l=0; l <= (int)floor((double)r/2); l++){
		for(int m=0; m <= r-(2*l); m++){			
			num_of_a_terms++;
		}
	}

	//printf("r=%d num_of_a_terms=%d\n",r,num_of_a_terms);

	a_terms=new double[num_of_a_terms];
	int k=0;
	for(int l=0; l <= (int)floor((double)r/2); l++){
		for(int m=0; m <= r-(2*l); m++){
			a_terms[k]=(l_constant[l]*m_constant[m]*r_factorial)/((double)factorial(r-(2*l)-m));
			//printf("%f \n",a_terms[k]);
			k++;			
		}
	}
	delete []l_constant;
	delete []m_constant;

}

//-------------------------------------------------------------------
// Compute the contant term B^{n}_{km} for all the clusters.
// n=0...K-1
// k=0...p-1
// m=0...r 
//-------------------------------------------------------------------

void
FastUnivariateDensityDerivative::compute_B(){

	num_of_B_terms=K*p*(r+1);

	//printf("K=%d p=%d r=%d num_of_B_terms=%d\n",K,p,r,num_of_B_terms);

	B_terms=new double[num_of_B_terms];

	double *k_factorial;
	k_factorial=new double[p];

	k_factorial[0]=1;
	for(int i=1; i<p ;i++){
		k_factorial[i]=k_factorial[i-1]/i;
		//printf("%f \n",k_factorial[i]);
	}

	double *temp3;
	temp3=new double[p+r];

	for(int n=0; n<K; n++){
		//printf("Cluster %d ",n);
		for(int k=0; k<p; k++){
			for(int m=0; m< r+1; m++){
				B_terms[(n*p*(r+1))+((r+1)*k)+m]=0.0;;
				//printf("%f ",B_terms[(n*p*(r+1))+((r+1)*k)+m]);
			}
		}
		//printf("\n");
	}

	for(int i=0; i<N; i++){
		int cluster_number=pClusterIndex[i];
		double temp1=(px[i]-pClusterCenter[cluster_number])/h;
		double temp2=exp(-temp1*temp1/2);
		temp3[0]=1;
		for(int k=1; k<p+r; k++){
			temp3[k]=temp3[k-1]*temp1;
		}

		for(int k=0; k<p; k++){
			for(int m=0; m< r+1; m++){
				B_terms[(cluster_number*p*(r+1))+((r+1)*k)+m]+=(temp2*temp3[k+m]);
			}
		}
	}

	for(int n=0; n<K; n++){
		//printf("Cluster %d ",n);
		for(int k=0; k<p; k++){
			for(int m=0; m< r+1; m++){
				B_terms[(n*p*(r+1))+((r+1)*k)+m]*=(k_factorial[k]*q);
				//printf("%f ",B_terms[(n*p*(r+1))+((r+1)*k)+m]);
			}
		}
		//printf("\n");
	}


	delete []k_factorial;
	delete []temp3;
	

}


//-------------------------------------------------------------------
// Actual function to evaluate the Univariate Density Derivative.
//-------------------------------------------------------------------

void
FastUnivariateDensityDerivative::Evaluate()
{

	//int num_of_influential_neighbors=(int)ceil(ry/rx);
	//printf("Num of influential right or left neighbors = %d\n",num_of_influential_neighbors);

	double *temp3;
	temp3=new double[p+r];


	for(int j=0; j<M; j++){
		pD[j]=0.0;
		
		int target_cluster_number=min((int)floor(py[j]/rx),K-1);
		double temp1=py[j]-pClusterCenter[target_cluster_number];
		double dist=abs(temp1);
		while (dist <= ry && target_cluster_number <K && target_cluster_number >=0){

			//printf("j=%d y=%f Influential cluster=%d\n",j,py[j],target_cluster_number);
			//Do something
			double temp2=exp(-temp1*temp1/two_h_square);
			double temp1h=temp1/h;
			temp3[0]=1;
			for(int i=1; i<p+r; i++){
				temp3[i]=temp3[i-1]*temp1h;
			}

			for(int k=0; k<=p-1; k++){
				int dummy=0;
				for(int l=0; l <= (int)floor((double)r/2); l++){
					for(int m=0; m <= r-(2*l); m++){
						pD[j]=pD[j]+(a_terms[dummy]*B_terms[(target_cluster_number*p*(r+1))+((r+1)*k)+m]*temp2*temp3[k+r-(2*l)-m]);
						dummy++;
					}
				}
			}
			//
				

			target_cluster_number++;
			temp1=py[j]-pClusterCenter[target_cluster_number];
			dist=abs(temp1);
		}

		target_cluster_number=min((int)floor(py[j]/rx),K-1)-1;
		if (target_cluster_number >=0){
			double temp1=py[j]-pClusterCenter[target_cluster_number];
			double dist=abs(temp1);
			while (dist <= ry && target_cluster_number <K && target_cluster_number >=0){
				//printf("j=%d y=%f Influential cluster=%d\n",j,py[j],target_cluster_number);
				//Do something
				double temp2=exp(-temp1*temp1/two_h_square);
			    double temp1h=temp1/h;
				temp3[0]=1;
				for(int i=1; i<p+r; i++){
					temp3[i]=temp3[i-1]*temp1h;
				}

				for(int k=0; k<=p-1; k++){
					int dummy=0;
					for(int l=0; l <= (int)floor((double)r/2); l++){
						for(int m=0; m <= r-(2*l); m++){
							pD[j]=pD[j]+(a_terms[dummy]*B_terms[(target_cluster_number*p*(r+1))+((r+1)*k)+m]*temp2*temp3[k+r-(2*l)-m]);
							dummy++;
						}
					}
				}
				//
				target_cluster_number--;
				temp1=py[j]-pClusterCenter[target_cluster_number];
				dist=abs(temp1);
			}
		}

	}

	
	delete []temp3;
}
