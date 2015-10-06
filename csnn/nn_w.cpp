/*
 * nn_w.cpp
 *
 *  Created on: Sep 20, 2015
 *      Author: zzs
 */

#include "nn_wv.h"
#include "nn_wb.h"
#include "nn_math.h"

//---------------------------------------------------------
//for nn_wb
void nn_wb::forward(/*const*/REAL* in,REAL* out,int bsize)
{
	// classical matrix multiplication -- column major
	// out   =   w   *   in   +   b
	// o*bs		o*i		i*bs
	for (int e=0; e<bsize; e++)
		memcpy(out+e*odim,b,odim*sizeof(REAL));
	nn_math::op_A_mult_B(out,w,in,odim,bsize,idim,false,false,1,1);
}

void nn_wb::backward(/*const*/REAL* ograd,REAL* igrad,/*const*/REAL* in,int bsize)
{
	// again matrix
    // backprop gradient:   igrad   +=        w'        *   ograd
    //                    idim x bsize = (odim x idim)'  *  odim x bsize
	// here += means accumulated gradient (the task of clear belongs to the outside)
	nn_math::op_A_mult_B(igrad,w,ograd,idim,bsize,odim,true,false,1,1);

	//accumulate gradients into tmp ones
	if(updating){
		//gradient of b
		REAL *gptr = ograd;
		for (int e=0; e<bsize; e++, gptr+=odim)
			nn_math::op_y_plus_ax(odim,b_grad,ograd,1);
		//gradient of w
		//gw += ograd * in'
		//o*i	o*b		b*i
		nn_math::op_A_mult_B(w_grad,ograd,in,odim,idim,bsize,false,true,1,1);
	}
}

void nn_wb::update(int way,REAL lrate,REAL wdecay,REAL m_alpha,REAL rms_smooth,int mbsize)
{
	//update
	nn_math::opt_update(way,idim*odim,lrate,wdecay,m_alpha,rms_smooth,w,w_grad,w_moment,w_square,mbsize);
	nn_math::opt_update(way,odim,lrate,wdecay,m_alpha,rms_smooth,b,b_grad,b_moment,b_square,mbsize);
	//clear the gradient
	clear_grad();
}

//----------------------------------------------------------
//for nn_wv
void nn_wv::forward(int index,REAL* out,int adding)
{
	if(!adding){
		if(index<0)
			memset(out,0,sizeof(REAL)*dim);
		memcpy(out,w+index*dim,sizeof(REAL)*dim);
	}
	else if(index>=0){
		nn_math::op_y_plus_ax(dim,out,w+index*dim,1);
	}
}

void nn_wv::backward(int index,const REAL* grad)
{
	if(index<0)
		return;
	hit_index->insert(index);
	nn_math::op_y_plus_ax(dim,w_grad+index*dim,grad,1);
}

void nn_wv::update(int way,REAL lrate,REAL wdecay,REAL m_alpha,REAL rms_smooth,int mbsize)
{
	//update
	for(IntSet::iterator i = hit_index->begin();i!=hit_index->end();i++){
		int index = *i;
		if(index<0)
			continue;
		nn_math::opt_update(way,dim,lrate,wdecay,m_alpha,rms_smooth,
				w+index*dim,w_grad+index*dim,w_moment+index*dim,w_square+index*dim,mbsize);
	}
	//clear the gradient
	clear_grad();
}


