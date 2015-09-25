#ifndef _NN_MATH_H_
#define _NN_MATH_H_

typedef float REAL;
#include "Blas.h"
#include <cmath>

namespace nn_math{

enum{
	OPT_SGD,OPT_SGD_MOMENTUM,
	OPT_ADAGRAD,OPT_RMSPROP,OPT_ADAM
};

void op_A_mult_B(REAL *C,const REAL *A,const REAL *B,const int m,const int n,const int dimk,
			const bool transA,const bool transB,const REAL a,const REAL b);
void opt_update(int way,int n,REAL lrate,REAL wdecay,REAL m_alpha,REAL rms_smooth,
			REAL* w,REAL* grad,REAL* momentum,REAL* square,int mbsize);
bool opt_changelrate(int way);

}

#endif
