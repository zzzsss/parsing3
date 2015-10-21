/*
 * Csnn_check.cpp
 *
 *  Created on: Oct 4, 2015
 *      Author: zzs
 */

#include "Csnn.h"

void Csnn::check_gradients(nn_input* in)
{
	vector<int>* goals = in->goals;
	//assuming softmax output
	//adjust some conf --- no check in real situations
	REAL tmp_drop = the_option->NN_dropout;
	int tmp_dim = the_option->NN_untied_dim;
	int tmp_softmax = the_option->NN_out_prob;
	the_option->NN_dropout = 0;
	the_option->NN_untied_dim = 0;
	the_option->NN_out_prob = 1;

	clear_params();
	//calculate real gradients
	REAL *target = forward(in,0);
	REAL origin_loss = 0;
	REAL *grad = new REAL[this_bsize*the_option->NN_out_size];
	memcpy(grad,target,sizeof(REAL)*this_bsize*the_option->NN_out_size);
	for(int i=0;i<this_bsize;i++){
		REAL* tmp = &grad[i*the_option->NN_out_size+goals->at(i)];
		origin_loss -= log(*tmp);	//cross-entropy loss
		/*if(! (origin_loss>0)){	//DEBUG
			cout << "what" << endl;
		}*/
		*tmp -= 1;
	}
	backward(grad);

	//random choose approximate --- repeat some times
	//check some
	vector<nn_wb*> to_changes;
	to_changes.push_back(p_out);
	to_changes.push_back(p_h);
	to_changes.push_back(p_untied->at(0));
	const int TMP_each_times = 10;
	for(int p=0;p<to_changes.size();p++){
	for(int i=0;i<TMP_each_times;i++){
		const REAL step = 1e-2;
		const REAL threshold = 1e-5;
		nn_wb* to_change = to_changes[p];
		//choose one
		int choose = int(to_change->geti()*to_change->geto()*drand48());
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
		if(abs(choose_grad-appr_grad)/this_bsize > threshold){
			cerr << "GRADIENT ERROR: calculated (/bs) " << choose_grad/this_bsize << " vs. approximate "
					<< appr_grad/this_bsize << " at " << p << endl;
		}
		else{
			cerr << "GRADIENT OK: calculated (/bs) " << choose_grad/this_bsize << " vs. approximate "
					<< appr_grad/this_bsize << " at " << p << endl;
		}
		delete []target;
	}
	}

	clear_params();
	the_option->NN_dropout = tmp_drop;
	the_option->NN_untied_dim = tmp_dim;
	the_option->NN_out_prob = tmp_softmax;
	delete []target;
	delete []grad;
}


