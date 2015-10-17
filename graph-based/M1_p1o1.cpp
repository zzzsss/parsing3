/*
 * M1_p1o1.cpp
 *
 *  Created on: Oct 14, 2015
 *      Author: zzs
 */

#include "M1_p1o1.h"

//only before training (and after building dictionary)
void M1_p1o1::each_create_machine()
{
	//several need setting places
	hp->hp_nn.NN_wnum = dict->getnum_word();
	hp->hp_nn.NN_pnum = dict->getnum_pos();
	hp->hp_nn.NN_dnum = dict->get_helper()->get_distance_num();
	hp->hp_nn.NN_out_prob = 1;
	if(hp->CONF_labeled)
		hp->hp_nn.NN_out_size = dict->getnum_deprel()+1;
	else
		hp->hp_nn.NN_out_size = 2;
	//create the new mach
	mach = Csnn::create(1,&hp->hp_nn);
}

void M1_p1o1::each_test_one(DependencyInstance* x)
{
	Process::parse_o1(x);
}

void M1_p1o1::each_train_one_iter()
{
	//per-sentence approach
	int mini_batch = hp->CONF_minibatch;
	int num_sentences = training_corpus->size();
	//statistics
	int skip_sent_num = 0;
	int all_forward_instance = 0;
	//some useful info
	int odim = mach->get_odim();
	//training
	time_t now;
	time(&now); //ctime is not rentrant ! use ctime_r() instead if needed
	cout << "##*** Start the training for iter " << cur_iter << " at " << ctime(&now)
			<< "with lrate " << cur_lrate << endl;
	cout << "#Sentences is " << num_sentences << " and resample (about)" << num_sentences*hp->CONF_NN_resample << endl;
	for(int i=0;i<num_sentences;){
		mach->prepare_batch();
		//if nesterov update before each batch (pre-update)
		if(hp->CONF_NESTEROV_MOMENTUM)
			mach->nesterov_update(hp->CONF_UPDATE_WAY,hp->CONF_MOMENTUM_ALPHA);
		//main batch
		for(int t=0;t<mini_batch && i<num_sentences;t++,i++){
			//random skip (instead of shuffling every time)
			if(drand48() > hp->CONF_NN_resample){
				skip_sent_num ++;
				continue;
			}

			//forward
			DependencyInstance* x = training_corpus->at(i);
			nn_input* the_inputs;
			REAL *fscores = forward_scores_o1(x,mach,&the_inputs,dict->get_helper(),0);
			all_forward_instance += the_inputs->get_numi();

			//prepare gradients --- softmax -> fscores as gradient
			double *to_change = fscores;
			for(int i=0;i<the_inputs->get_numi();i++){
				int tmp_goal = the_inputs->goals->at(i);
				to_change[tmp_goal] -= 1;	//-1 for the right one
				to_change += odim;
			}

			//backward
			mach->backward(fscores);

			//
			mach->check_gradients(the_inputs);

			delete the_inputs;
			delete []fscores;
		}
		//real update
		mach->update(hp->CONF_UPDATE_WAY,cur_lrate,hp->CONF_NN_WD,hp->CONF_MOMENTUM_ALPHA,hp->CONF_RMS_SMOOTH);
	}
	cout << "Iter done, skip " << skip_sent_num << " sentences and f&b " << all_forward_instance << endl;
}
