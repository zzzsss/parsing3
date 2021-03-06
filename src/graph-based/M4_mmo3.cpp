/*
 * M4_mmo3.cpp
 *
 *  Created on: Mar 11, 2016
 *      Author: zzs
 */

#include "M4_mm.h"
#include "../algorithms/Eisner.h"
#include "../algorithms/EisnerO3g.h"


void M4_o3::each_create_machine()
{
	//several need setting places
	hp->hp_nn.NN_wnum = dict->getnum_word();
	hp->hp_nn.NN_pnum = dict->getnum_pos();
	hp->hp_nn.NN_dnum = dict->get_helper()->get_distance_num();
	hp->hp_nn.NN_out_prob = 0;	//not softmax here
	if(hp->CONF_labeled)
		hp->hp_nn.NN_out_size = dict->getnum_deprel();
	else
		hp->hp_nn.NN_out_size = 1;
	//create the new mach
	mach = Csnn::create(3,&hp->hp_nn);
}

void M4_o3::each_test_one(DependencyInstance* x,int dev)
{
	int noc_dev = dev ? hp->CONF_score_noc_dev : 0;
	if(noc_dev){
		if(x->length() >= hp->CONF_higho_toolong){
			//default is set to 0
			x->predict_heads = new vector<int>(x->length(),0);
			x->predict_deprels = new vector<int>(x->length(),0);
		}
		else{
			Process::parse_o3g(x,mfo1,0,0);
		}
	}
	else{
		if(x->length() >= hp->CONF_higho_toolong){
			//tricky ...
			Csnn* tmp = mach;
			mach = mso2;
			Process::parse_o2sib(x,mfo1,mso1);
			mach = tmp;
		}
		else{
			Process::parse_o3g(x,mfo1,mso1,mso2);
		}
	}
}

void M4_o3::each_train_one_iter()
{
	//per-sentence approach
	int num_sentences = training_corpus->size();
	//statistics
	int skip_sent_num = 0;
	int all_forward_instance = 0;
	int all_inst_right = 0;
	int all_inst_wrong = 0;
	//some useful info
	int odim = mach->get_odim();
	//training
	time_t now;
	time(&now); //ctime is not rentrant ! use ctime_r() instead if needed
	cout << "##*** //M4O1// Start the training for iter " << cur_iter << " at " << ctime(&now)
			<< "with lrate " << cur_lrate << endl;
	cout << "#Sentences is " << num_sentences << " and resample (about)" << num_sentences*hp->CONF_NN_resample << endl;

	vector<DependencyInstance*> xs;
	int all_token=0,all_right=0;
	for(int i=0;i<num_sentences;){
		//random skip (instead of shuffling every time)
		if(drand48() > hp->CONF_NN_resample || training_corpus->at(i)->length() >= hp->CONF_higho_toolong){
			skip_sent_num ++;
			i ++;
			continue;
		}
		//main batch
		int this_instance_toupdate = 0;
		int this_tokens = 0;
		for(;;){
			//forward
			DependencyInstance* x = training_corpus->at(i);
			xs.push_back(x);

			Process::parse_o3g(x,mfo1,0,0, true);	//add margin MAYBE
			// -- statistic
			all_token += x->length()-1;
			for(int i2=1;i2<x->length();i2++){	//ignore root
				if((*(x->predict_heads))[i2] == (*(x->heads))[i2])
					all_right ++;
				else
					this_instance_toupdate++;
			}
			//
			this_tokens += x->length() - 1;
			i++;

			if(i>=num_sentences)
				break;
			//out of the mini-batch
			while(training_corpus->at(i)->length() >= hp->CONF_higho_toolong){	//HAVE to compromise, bad choice
				skip_sent_num ++;
				i ++;
			}
			if(i>=num_sentences)
				break;
			if (hp->CONF_minibatch > 0) {
				if (int(xs.size()) >= hp->CONF_minibatch)
					break;
			}
			else {
				if (this_instance_toupdate >= -1 * hp->CONF_minibatch)
					break;
			}
		}

		//backward
		for(int ii=0;ii<xs.size();ii++){
			DependencyInstance* x = xs[ii];
			nn_input* good;
			nn_input* bad;
			M3_pro2::get_nninput_o3g(x,&good,&bad,dict);
			MM_margin_backward(mach, good, 1, hp->CONF_score_p2reg);
			MM_margin_backward(mach, bad, -1, hp->CONF_score_p2reg);
			delete good;delete bad;
		}
		int this_sentence = xs.size();
		xs.clear();
		//real update
		if (hp->CONF_mbatch_way == 1)
			mach->set_this_mbsize(this_tokens*this_tokens);
		else if (hp->CONF_mbatch_way == 2)
			mach->set_this_mbsize(this_sentence*this_sentence);
		mach->update(hp->CONF_UPDATE_WAY,cur_lrate,hp->CONF_NN_WD,hp->CONF_MOMENTUM_ALPHA,hp->CONF_RMS_SMOOTH);
	}
	cout << "Iter done, skip " << skip_sent_num << " sentences." << "AND training UAS:"
			<< all_right << "/" << all_token << "=" << all_right/(0.0+all_token) << endl;
}



