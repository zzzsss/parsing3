/*
 * Csnn.cpp
 *
 *  Created on: Sep 25, 2015
 *      Author: zzs
 */

#include "Csnn.h"


//--prepares--
void Csnn::construct_caches(){
	//this is done when init and read
	c_allcaches = new vector<nn_cache*>();
	c_out = new nn_cache(the_option->NN_init_bsize,the_option->NN_out_size);	c_allcaches->push_back(c_out);
	c_h = new nn_cache(the_option->NN_init_bsize,the_option->NN_hidden_size);	c_allcaches->push_back(c_h);
	c_repr = new nn_cache(the_option->NN_init_bsize,the_option->get_NN_rsize());	c_allcaches->push_back(c_repr);
	c_wrepr = new nn_cache(the_option->NN_init_bsize,the_option->NN_wrsize);	c_allcaches->push_back(c_wrepr);
	c_srepr = new nn_cache(the_option->NN_init_bsize,the_option->get_NN_srsize());	c_allcaches->push_back(c_srepr);
	c_wv = new nn_cache(the_option->NN_init_bsize,the_option->get_NN_wv_wrsize(get_order()));	c_allcaches->push_back(c_wv);
}

void Csnn::prepare_caches(){
	//only need to clear all the gradients and get possible drop-out ready
	//this is done before each MiniBatch
	for(vector<nn_cache*>::iterator i=c_allcaches->begin();i!=c_allcaches->end();i++){
		(*i)->clear_gradients();
		if(the_option->NN_dropout > 0)
			(*i)->gen_dropout(the_option->NN_dropout);
	}
}
