/*
 * MP2_o2sib.cpp
 *
 *  Created on: 2015Äê7ÔÂ8ÈÕ
 *      Author: zzs
 */

#include "MP2_o2sib.h"
#include "../algorithms/EisnerO2sib.h"
#include "../algorithms/Eisner.h"

void MP2_o2sib::nn_train_one_iter()
{
	int sentences = training_corpus->size();
	int sentences_skip = 0;
	int all_forward = 0;
	int zero_backward = 0;
	time_t now;
	time(&now); //ctime is not rentrant ! use ctime_r() instead if needed
	cout << "##*** mp2-o2sib Start the training for iter " << cur_iter << " at " << ctime(&now)
			<< "with lrate " << cur_lrate << endl;
	cout << "#Sentences is " << sentences << " and resample (about)" << sentences*parameters->CONF_NN_resample << endl;

	static bool** all_noprob_o1 = 0;
	//---always perform filtering
	//sweep all once and count
	if(all_noprob_o1==0){
	FeatureGenO1* feat_temp_o1 = new FeatureGenO1(dict,parameters->CONF_x_window,
					parameters->CONF_add_distance,parameters->CONF_add_pos,parameters->CONF_add_direction);
	all_noprob_o1 = new bool*[sentences];
	int all_tokens_train=0,all_token_filter_wrong=0;
	for(int i=0;i<sentences;i++){
			DependencyInstance* x = training_corpus->at(i);
			int len = x->length();
			double* scores_o1_filter = get_scores_o1(x,parameters,mach_o1_filter,feat_temp_o1);
			all_tokens_train += len;
			all_noprob_o1[i] = get_noprob_o1(len,scores_o1_filter);
			for(int m=1;m<len;m++){
				if(all_noprob_o1[i][get_index2(len,x->heads->at(m),m)])
					all_token_filter_wrong ++;
			}
			delete []scores_o1_filter;
	}
	cout << "For o1 filter: all " << all_tokens_train << ";filter wrong " << all_token_filter_wrong << endl;
	time_t now;
	time(&now);cout << "#Finish o1-filter at " << ctime(&now) << endl;
	}

	for(int i=0;i<sentences;i++){
		if(double(rand())/RAND_MAX > parameters->CONF_NN_resample){
			sentences_skip++;
			continue;
		}
		//1.allocate(maybe)
		DependencyInstance* x = training_corpus->at(i);
		int length = x->length();
		bool* o1_noprob = all_noprob_o1[i];
		int to_alloc = length*length*length;	//2nd order
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
		//2.featgen_fill
		REAL* assign_x = data;
		REAL* assign_g = gradient;
		int real_num_forw = 0;
		int idim = mach->GetIdim();
		for(int m=1;m<length;m++){
			for(int h=0;h<length;h++){
				if(h==m)
					continue;
				//get information
				int small = GET_MIN_ONE(m,h);
				int large = GET_MAX_ONE(m,h);
				bool link_hm = (x->heads->at(m)==h);
				int noprob_hm = o1_noprob[get_index2(length,h,m)];
				int c=-1;	//inside sibling
				if(link_hm){
				if(h>m){
					for(int ii=m+1;ii<h;ii++)
						if(x->heads->at(ii)==h){c = ii;break;}
				}
				else{
					for(int ii=m-1;ii>h;ii--)
						if(x->heads->at(ii)==h){c = ii;break;}
				}}
				//assign
				if(link_hm && c<0){
					feat_gen->fill_one(assign_x,x,h,m,-1);assign_x += idim;
					*assign_g = 1;
					assign_g ++;
					real_num_forw++;
				}
				else if(noprob_hm){}
				else{
					feat_gen->fill_one(assign_x,x,h,m,-1);assign_x += idim;
					*assign_g = 0;
					assign_g ++;
					real_num_forw++;
				}
				for(int mid=small+1;mid<large;mid++){
					if(link_hm && mid==c){
						feat_gen->fill_one(assign_x,x,h,m,mid);assign_x += idim;
						*assign_g = 1;
						assign_g ++;
						real_num_forw++;
					}
					else if(noprob_hm || o1_noprob[get_index2(length,h,mid)]){}
					else{
						feat_gen->fill_one(assign_x,x,h,m,mid);assign_x += idim;
						*assign_g = 0;
						assign_g ++;
						real_num_forw++;
					}
				}
			}
		}
		//3.forward
		all_forward += real_num_forw;
		REAL* mach_y = mach->mach_forward(data,real_num_forw);
		//4.scores
		REAL* assign_y = mach_y;
		assign_g = gradient;
		double* tmp_scores = new double[length*length*length];
		for(int ii=0;ii<length*length*length;ii++)
			tmp_scores[ii] = DOUBLE_LARGENEG;
		for(int m=1;m<length;m++){
			for(int h=0;h<length;h++){
				if(h==m)
					continue;
				//get information
				int small = GET_MIN_ONE(m,h);
				int large = GET_MAX_ONE(m,h);
				bool link_hm = (x->heads->at(m)==h);
				int noprob_hm = o1_noprob[get_index2(length,h,m)];
				int c=-1;	//inside sibling
				if(link_hm){
				if(h>m){
					for(int ii=m+1;ii<h;ii++)
						if(x->heads->at(ii)==h){c = ii;break;}
				}
				else{
					for(int ii=m-1;ii>h;ii--)
						if(x->heads->at(ii)==h){c = ii;break;}
				}}
				//assign
				if(link_hm && c<0){
					tmp_scores[get_index2_o2sib(length,h,h,m)] = *assign_y;
					*assign_g -= 2*parameters->CONF_MP_scale_reg*(*assign_y);
					assign_g++;
					assign_y ++;
				}
				else if(noprob_hm){}
				else{
					tmp_scores[get_index2_o2sib(length,h,h,m)] = *assign_y;
					*assign_g -= 2*parameters->CONF_MP_scale_reg*(*assign_y);
					assign_g++;
					assign_y ++;
				}
				for(int mid=small+1;mid<large;mid++){
					if(link_hm && mid==c){
						tmp_scores[get_index2_o2sib(length,h,mid,m)] = *assign_y;
						*assign_g -= 2*parameters->CONF_MP_scale_reg*(*assign_y);
						assign_g++;
						assign_y ++;
					}
					else if(noprob_hm || o1_noprob[get_index2(length,h,mid)]){}
					else{
						tmp_scores[get_index2_o2sib(length,h,mid,m)] = *assign_y;
						*assign_g -= 2*parameters->CONF_MP_scale_reg*(*assign_y);
						assign_g++;
						assign_y ++;
					}
				}
			}
		}
		//5.gradients
		double* tmp_marginals = encodeMarginals_o2sib(length,tmp_scores);
		assign_g = gradient;
		REAL gradient_small = parameters->CONF_MP_gradient_small;
		REAL* assign_redata = rearrange_data;
		REAL* aasign_regrad = rearrange_gradient;
		int num_rearrange = 0;
		for(int m=1;m<length;m++){
			for(int h=0;h<length;h++){
				if(h==m)
					continue;
				//get information
				int small = GET_MIN_ONE(m,h);
				int large = GET_MAX_ONE(m,h);
				bool link_hm = (x->heads->at(m)==h);
				int noprob_hm = o1_noprob[get_index2(length,h,m)];
				int c=-1;	//inside sibling
				if(link_hm){
				if(h>m){
					for(int ii=m+1;ii<h;ii++)
						if(x->heads->at(ii)==h){c = ii;break;}
				}
				else{
					for(int ii=m-1;ii>h;ii--)
						if(x->heads->at(ii)==h){c = ii;break;}
				}}
				//assign
				if(link_hm && c<0){
					*assign_g -= tmp_marginals[get_index2_o2sib(length,h,h,m)];
					//if gradient is too small, just ignore it to avoid numeric issues
					if(*assign_g < gradient_small && *assign_g > -gradient_small){
						*assign_g = 0;
						zero_backward ++;
					}
					else if(parameters->CONF_MP_training_rearrange){
						//re-arrange
						num_rearrange++;
						feat_gen->fill_one(assign_redata,x,h,m,-1);
						assign_redata += mach->GetIdim();
						*aasign_regrad = *assign_g;
						aasign_regrad++;
					}
					assign_g++;
				}
				else if(noprob_hm){}
				else{
					*assign_g -= tmp_marginals[get_index2_o2sib(length,h,h,m)];
					//if gradient is too small, just ignore it to avoid numeric issues
					if(*assign_g < gradient_small && *assign_g > -gradient_small){
						*assign_g = 0;
						zero_backward ++;
					}
					else if(parameters->CONF_MP_training_rearrange){
						//re-arrange
						num_rearrange++;
						feat_gen->fill_one(assign_redata,x,h,m,-1);
						assign_redata += mach->GetIdim();
						*aasign_regrad = *assign_g;
						aasign_regrad++;
					}
					assign_g++;
				}
				for(int mid=small+1;mid<large;mid++){
					if(link_hm && mid==c){
						*assign_g -= tmp_marginals[get_index2_o2sib(length,h,mid,m)];
						//if gradient is too small, just ignore it to avoid numeric issues
						if(*assign_g < gradient_small && *assign_g > -gradient_small){
							*assign_g = 0;
							zero_backward ++;
						}
						else if(parameters->CONF_MP_training_rearrange){
							//re-arrange
							num_rearrange++;
							feat_gen->fill_one(assign_redata,x,h,m,mid);
							assign_redata += mach->GetIdim();
							*aasign_regrad = *assign_g;
							aasign_regrad++;
						}
						assign_g++;
					}
					else if(noprob_hm || o1_noprob[get_index2(length,h,mid)]){}
					else{
						*assign_g -= tmp_marginals[get_index2_o2sib(length,h,mid,m)];
						//if gradient is too small, just ignore it to avoid numeric issues
						if(*assign_g < gradient_small && *assign_g > -gradient_small){
							*assign_g = 0;
							zero_backward ++;
						}
						else if(parameters->CONF_MP_training_rearrange){
							//re-arrange
							num_rearrange++;
							feat_gen->fill_one(assign_redata,x,h,m,mid);
							assign_redata += mach->GetIdim();
							*aasign_regrad = *assign_g;
							aasign_regrad++;
						}
						assign_g++;
					}
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

vector<int>* MP2_o2sib::each_test_one(DependencyInstance* x)
{
	int length = x->length();
	FeatureGenO1* feat_temp_o1 = new FeatureGenO1(dict,parameters->CONF_x_window,
			parameters->CONF_add_distance,parameters->CONF_add_pos,parameters->CONF_add_direction);
	//filter
	double* scores_o1_filter = get_scores_o1(x,parameters,mach_o1_filter,feat_temp_o1);	//same parameters
	bool *whether_cut_o1 = get_noprob_o1(length,scores_o1_filter);
	double *tmp_scores = get_scores_o2sib(x,parameters,mach,feat_gen,whether_cut_o1);
	delete []whether_cut_o1;
	delete []scores_o1_filter;
	//combine scores??
	if(parameters->CONF_MP_o1mach.length() > 0){
		double* score_of_o1 = get_scores_o1(x,parameters,mach_o1_score,feat_temp_o1);
		for(int m=1;m<length;m++){
			for(int h=0;h<length;h++){
				if(m!=h){
					double score_tmp = score_of_o1[get_index2(length,h,m)];
					tmp_scores[get_index2_o2sib(length,h,h,m)] += score_tmp;
					for(int c=h+1;c<m;c++)
						tmp_scores[get_index2_o2sib(length,h,c,m)] += score_tmp;
					for(int c=m+1;c<h;c++)
						tmp_scores[get_index2_o2sib(length,h,c,m)] += score_tmp;
				}
			}
		}
		delete []score_of_o1;
	}
	delete feat_temp_o1;
	vector<int> *ret = decodeProjective_o2sib(length,tmp_scores);
	delete []tmp_scores;
	return ret;
}

