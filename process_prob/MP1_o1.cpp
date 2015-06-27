/*
 * MP1_o1.cpp
 *
 *  Created on: 2015��6��10��
 *      Author: zzs
 */

#include "MP1_o1.h"
#include "../algorithms/Eisner.h"

void MP1_o1::nn_train_one_iter()
{
	int sentences = training_corpus->size();
	int sentences_skip = 0;
	int all_forward = 0;
	int zero_backward = 0;
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
			gradient = new REAL[to_alloc];
			alloc_sample_size = to_alloc;
			if(parameters->CONF_MP_training_rearrange){
				delete []rearrange_data;
				delete []rearrange_gradient;
				rearrange_data = new REAL[to_alloc*mach->GetIdim()];
				rearrange_gradient = new REAL[to_alloc];
			}
		}
		//1.5 m start for where
		int m_start = 1;
		if(parameters->CONF_MP_o1training_0m)
			m_start = 0;
		//2.featgen_fill
		REAL* assign_x = data;
		REAL* assign_g = gradient;
		int real_num_forw = 0;
		for(int m=m_start;m<length;m++){
			for(int h=0;h<length;h++){
				if(h != m){
					feat_gen->fill_one(assign_x,x,h,m);
					assign_x += mach->GetIdim();
					if(x->heads->at(m)==h)
						*assign_g = 1;
					else
						*assign_g = 0;
					assign_g ++;
					real_num_forw++;
				}
			}
		}
		//3.forward
		all_forward += real_num_forw;
		REAL* mach_y = mach->mach_forward(data,real_num_forw);
		//4.scores
		REAL* assign_y = mach_y;
		double* tmp_scores = new double[length*length];
		for(int ii=0;ii<length*length;ii++)
			tmp_scores[ii] = DOUBLE_LARGENEG;
		for(int m=m_start;m<length;m++){
			for(int h=0;h<length;h++){
				if(h != m){
					tmp_scores[get_index2(length,h,m)] = *assign_y;
					assign_y ++;
				}
			}
		}
		//5.gradients
		double* tmp_marginals = encodeMarginals(length,tmp_scores);
		assign_g = gradient;
		REAL gradient_small = parameters->CONF_MP_gradient_small;
		REAL* assign_redata = rearrange_data;
		REAL* aasign_regrad = rearrange_gradient;
		int num_rearrange = 0;
		for(int m=m_start;m<length;m++){
			for(int h=0;h<length;h++){
				if(h != m){
					*assign_g -= tmp_marginals[get_index2(length,h,m)];
					//if gradient is too small, just ignore it to avoid numeric issues
					if(*assign_g < gradient_small && *assign_g > -gradient_small){
						*assign_g = 0;
						zero_backward ++;
					}
					else if(parameters->CONF_MP_training_rearrange){
						//re-arrange
						num_rearrange++;
						feat_gen->fill_one(assign_redata,x,h,m);
						assign_redata += mach->GetIdim();
						*aasign_regrad = *assign_g;
						aasign_regrad++;
					}
					assign_g++;
				}
			}
		}
		//6.back backward (also need forward if bs is small)
		if(parameters->CONF_MP_training_rearrange)
			mach->mach_forwback(rearrange_data,rearrange_gradient,cur_lrate, parameters->CONF_NN_WD,num_rearrange);
		else
			mach->mach_backward(data,gradient,cur_lrate, parameters->CONF_NN_WD,real_num_forw);
		delete []mach_y;
		delete []tmp_marginals;
		delete []tmp_scores;
	}
	cout << "Iter done, skip " << sentences_skip << " sentences and f&b " << all_forward
			<< ",zero-back " << zero_backward << endl;
}

