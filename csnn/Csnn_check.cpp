/*
 * Csnn_check.cpp
 *
 *  Created on: Oct 4, 2015
 *      Author: zzs
 */

#include "Csnn.h"

void Csnn::check_gradients(nn_input* in, vector<REAL>* goals)
{
	//assuming softmax output
	//adjust some conf --- no check in real situations
	REAL tmp_drop = the_option->NN_dropout;
	int tmp_dim = the_option->NN_untied_dim;
	int tmp_softmax = the_option->NN_out_prob;
	the_option->NN_dropout = 0;
	the_option->NN_untied_dim = 0;
	the_option->NN_out_prob = 1;


	//calculate real gradients
	REAL *target = forward(in,0);
	REAL origin_loss = 0;
	REAL *grad = new REAL[this_bsize*the_option->NN_out_size];
	memcpy(grad,target,sizeof(REAL)*this_bsize*the_option->NN_out_size);
	for(int i=0;i<this_bsize;i++){
		REAL* tmp = &grad[i*the_option->NN_out_size+goals->at(i)];
		origin_loss -= log(*tmp);	//cross-entropy loss
		*tmp -= 1;
	}
	backward(grad);

	//random choose approximate --- repeat some times
	//1.p_untied[0]
	nn_wb *to_change = p_untied->at(0);
	for(int i=0;i<10;i++){
		const REAL step = 1e-2;
		const REAL threshold = 1e-5;
		//choose one
		int choose = int(the_option->get_NN_wv_wrsize(get_order())*the_option->NN_wrsize*drand48());
		REAL* choose_w = to_change->get_w(choose);
		REAL choose_grad = to_change->get_g(choose);
		//forward
		*choose_w += step;
		REAL *target = forward(in,0);
		*choose_w -= step;
		//check
		REAL appr_loss = 0;
		for(int i=0;i<this_bsize;i++){
			REAL tmp = target[i*the_option->NN_out_size+goals->at(i)];
			appr_loss -= log(tmp);	//cross-entropy loss
		}
		REAL appr_grad = (appr_loss-origin_loss) / step;
		if(abs(choose_grad-appr_grad) > threshold){
			cerr << "GRADIENT ERROR: calculated " << choose_grad << " vs. approximate " << appr_grad << endl;
		}
	}

	clear_params();
	the_option->NN_dropout = tmp_drop;
	the_option->NN_untied_dim = tmp_dim;
	the_option->NN_out_prob = tmp_softmax;
}


