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
	for(int i=0;i<num_sentences;){
		mach->prepare_batch();
		for(int t=0;t<mini_batch && i<num_sentences;t++,i++){

		}
		mach->update();
	}
}
