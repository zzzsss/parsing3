/*
 * mmo1.cpp
 *
 *  Created on: 2015.8.20
 *      Author: zzs
 */

#include "Method11_mmo1.h"
#include "../algorithms/Eisner.h"


void Method11_mmo1::nn_train_one_iter()
{
 	int sentences = training_corpus->size();
	int sentences_skip = 0;
	int all_forward = 0;
	time_t now;
	time(&now); //ctime is not rentrant ! use ctime_r() instead if needed
	cout << "##*** Start the training for iter " << cur_iter << " at " << ctime(&now)
			<< "with lrate " << cur_lrate << endl;
	cout << "#Sentences is " << sentences << " and resample (about)" << sentences*parameters->CONF_NN_resample << endl;

	for(int i=0;i<sentences;i++){
		if(double(rand())/RAND_MAX > parameters->CONF_NN_resample){
			sentences_skip++;
			continue;
		}

		//1.allocate(maybe)
		DependencyInstance* x = training_corpus->at(i);
		int length = x->length();
		int to_alloc = length*2;
		if(alloc_sample_size < to_alloc){
			//allocate spaces
			delete []data;
			delete []gradient;
			data = new REAL[to_alloc*mach->GetIdim()];
			gradient = new REAL[to_alloc*mach->GetOdim()];
			alloc_sample_size = to_alloc;
		}

		//2.inference(adding loss score)
		vector<int> *ret = 0;
		{
			//adding loss scores (max-margin framework)
			double *tmp_scores = get_scores_o1(x,parameters,mach,feat_gen);
			for(int mod=1;mod<length;mod++){
				//-1 for right ones, maybe the same for inference
				tmp_scores[get_index2(length,x->heads->at(mod),mod)] -= 1;
			}
			ret = decodeProjective(x->length(),tmp_scores);
			delete []tmp_scores;
		}

		//3.set up for training
		int real_num_forw = 0;
		REAL* assign_x = data;
		REAL* assign_g = gradient;
		for(int ii=1;ii<length;ii++){
			int guess = ret->at(ii);
			int right = x->heads->at(ii);
			if(guess != right){
				real_num_forw += 2;
				feat_gen->fill_one(assign_x,x,right,ii);
				assign_x += mach->GetIdim();
				*assign_g++ = 1;
				feat_gen->fill_one(assign_x,x,guess,ii);
				assign_x += mach->GetIdim();
				*assign_g++ = -1;
			}
		}

		//4.training
		all_forward += real_num_forw;
		mach->mach_forwback(data,gradient,cur_lrate, parameters->CONF_NN_WD,real_num_forw);
		delete ret;
	}
	cout << "Iter done, skip " << sentences_skip << " sentences and f&b " << all_forward << endl;
}