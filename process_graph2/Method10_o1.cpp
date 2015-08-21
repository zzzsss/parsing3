/*
 * METHOD10_o1.cpp
 *
 *  Created on: 2015.8.20
 *      Author: zzs
 */

#include "Method10_o1.h"

void Method10_o1::nn_train_one_iter()
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
		int to_alloc = length*length;
		if(alloc_sample_size < to_alloc){
			//allocate spaces
			delete []data;
			delete []gradient;
			data = new REAL[to_alloc*mach->GetIdim()];
			gradient = new REAL[to_alloc*mach->GetOdim()];
			alloc_sample_size = to_alloc;
		}
		//2.featgen_fill
		REAL* assign_x = data;
		REAL* assign_g = gradient;
		int real_num_forw = 0;
		for(int m=1;m<length;m++){
			for(int h=0;h<length;h++){
				if(h != m){
					feat_gen->fill_one(assign_x,x,h,m);
					assign_x += mach->GetIdim();
					*assign_g = 0;
					*(assign_g+1) = 0;
					if(x->heads->at(m)==h)
						*(assign_g+1) = 1;
					else
						*assign_g = 1;
					assign_g += 2;
					real_num_forw++;
				}
			}
		}
		//3.training
		assign_x = data;
		assign_g = gradient;
		all_forward += real_num_forw;
		int bsize = mach->GetWidth();
		for(int nn=0;nn<real_num_forw;nn+=bsize){
			int this_num = bsize;
			if(real_num_forw < (nn+bsize))
				this_num = real_num_forw - nn;
			mach->SetDataIn(assign_x);
			mach->Forw(this_num);
			mach->SetGradOut(assign_g);
			REAL* this_output = mach->GetDataOut();
			for(int kk=0;kk<this_num*mach->GetOdim();kk++){
				assign_g[kk] -= this_output[kk];
			}
			mach->Backw(cur_lrate, parameters->CONF_NN_WD,this_num);
			assign_x += this_num*mach->GetIdim();
			assign_g += this_num*mach->GetOdim();
		}
	}
	cout << "Iter done, skip " << sentences_skip << " sentences and f&b " << all_forward << endl;
}