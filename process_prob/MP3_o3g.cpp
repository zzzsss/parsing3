/*
 * MP3_o3g.cpp
 *
 *  Created on: 2015Äê7ÔÂ16ÈÕ
 *      Author: zzs
 */

#include "MP3_o3g.h"
#include "../algorithms/EisnerO3g.h"
#include "../algorithms/EisnerO2sib.h"
#include "../algorithms/EisnerO2g.h"
#include "../algorithms/Eisner.h"

// ----------- REALLY REALLY REALLY BAD design ... //
void MP3_o3g::nn_train_one_iter()
{
	int sentences = training_corpus->size();
	int sentences_skip = 0;
	int sentences_skip_long = 0;
	int all_forward = 0;
	int zero_backward = 0;
	time_t now;
	time(&now); //ctime is not rentrant ! use ctime_r() instead if needed
	cout << "##*** mp3-o3g Start the training for iter " << cur_iter << " at " << ctime(&now)
			<< "with lrate " << cur_lrate << endl;
	cout << "#Sentences is " << sentences << " and resample (about)" << sentences*parameters->CONF_NN_resample << endl;

	static double** all_scores_o1 = 0;
	//---always perform filtering
	//sweep all once and count
	if(all_scores_o1==0){
	FeatureGenO1* feat_temp_o1 = new FeatureGenO1(dict,parameters->CONF_x_window,
					parameters->CONF_add_distance,parameters->CONF_add_pos,parameters->CONF_add_direction);
	all_scores_o1 = new double*[sentences];
	for(int i=0;i<sentences;i++){
		DependencyInstance* x = training_corpus->at(i);
		all_scores_o1[i] = get_scores_o1(x,parameters,mach_o1_filter,feat_temp_o1);
	}
	delete feat_temp_o1;
	}

	//decide the filtering
	bool** all_noprob_o1 = new bool*[sentences];
	int all_tokens_train=0,all_token_filter_wrong=0;
	for(int i=0;i<sentences;i++){
			DependencyInstance* x = training_corpus->at(i);
			int len = x->length();
			double* scores_o1_filter = all_scores_o1[i];
			all_tokens_train += len;
			all_noprob_o1[i] = get_noprob_o1(len,scores_o1_filter);
			for(int m=1;m<len;m++){
				if(all_noprob_o1[i][get_index2(len,x->heads->at(m),m)])
					all_token_filter_wrong ++;
			}
	}
	{
	cout << "For o1 filter at cut of " << parameters->CONF_NN_highO_o1filter_cut
			<< ": all " << all_tokens_train << ";filter wrong " << all_token_filter_wrong << endl;
	time_t now;
	time(&now);cout << "#Finish o1-filter at " << ctime(&now) << endl;
	}

	for(int i=0;i<sentences;i++){
		DependencyInstance* x = training_corpus->at(i);
		long length = x->length();

		if(double(rand())/RAND_MAX > parameters->CONF_NN_resample){
			sentences_skip++;
			continue;
		}
		else if(length >= parameters->CONF_MP_o3g_toolong){
			sentences_skip_long++;
			continue;
		}
		//1.allocate(maybe)
		bool* o1_noprob = all_noprob_o1[i];
		long to_alloc = length*length*length*length;	//3nd order
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
		//loop------------------
			for(int m=1;m<length;m++){
				//2.1 special (0,0,c,m)	when h==0
				int noprob_0m = o1_noprob[get_index2(length,0,m)];
				int link_0m = (x->heads->at(m)==0);
				int c = -1;
				for(int mid=m-1;mid>0;mid--){
					if(x->heads->at(mid)==0){
						c = mid;break;}
				}
				if(link_0m && c<0){
					feat_gen->fill_one(assign_x,x,0,m,-1,0);assign_x += idim;
					*assign_g=1;assign_g++;real_num_forw++;
				}
				else if(noprob_0m){}
				else{
					feat_gen->fill_one(assign_x,x,0,m,-1,0);assign_x += idim;
					*assign_g=0;assign_g++;real_num_forw++;
				}
				for(int mid=1;mid<m;mid++){
					if(link_0m && mid==c){
						feat_gen->fill_one(assign_x,x,0,m,mid,0);assign_x += idim;
						*assign_g=1;assign_g++;real_num_forw++;
					}
					else if(noprob_0m || o1_noprob[get_index2(length,0,mid)]){}
					else{
						feat_gen->fill_one(assign_x,x,0,m,mid,0);assign_x += idim;
						*assign_g=0;assign_g++;real_num_forw++;
					}
				}
				//2.2. ordinary ones
				for(int h=1;h<length;h++){	//h>=1
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
					//for g and c
					for(int g=0;g<length;g++){
						if(g<=large && g>=small)
							continue;
						bool link_gh = (x->heads->at(h)==g);
						int noprob_gh = o1_noprob[get_index2(length,g,h)];
						int nonproj_g = (g>=small && g<=large);
						if(link_hm && link_gh && c<0){
							feat_gen->fill_one(assign_x,x,h,m,-1,g);assign_x += idim;
							*assign_g=1;assign_g++;real_num_forw++;
						}
						else if(noprob_hm || noprob_gh || nonproj_g){}
						else{
							feat_gen->fill_one(assign_x,x,h,m,-1,g);assign_x += idim;
							*assign_g=0;assign_g++;real_num_forw++;
						}
						for(int mid=small+1;mid<large;mid++){
							if(link_hm && link_gh && mid==c){
								feat_gen->fill_one(assign_x,x,h,m,mid,g);assign_x += idim;
								*assign_g=1;assign_g++;real_num_forw++;
							}
							else if(noprob_hm || noprob_gh || nonproj_g || o1_noprob[get_index2(length,h,mid)]){}
							else{
								feat_gen->fill_one(assign_x,x,h,m,mid,g);assign_x += idim;
								*assign_g=0;assign_g++;real_num_forw++;
							}
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
		double* tmp_scores = new double[length*length*length*length];
		for(int ii=0;ii<length*length*length*length;ii++)
			tmp_scores[ii] = DOUBLE_LARGENEG;
		//loop------------------
			for(int m=1;m<length;m++){
				//2.1 special (0,0,c,m)	when h==0
				int noprob_0m = o1_noprob[get_index2(length,0,m)];
				int link_0m = (x->heads->at(m)==0);
				int c = -1;
				for(int mid=m-1;mid>0;mid--){
					if(x->heads->at(mid)==0){
						c = mid;break;}
				}
				if(link_0m && c<0){
					tmp_scores[get_index2_o3g(length,0,0,0,m)] = *assign_y;
					*assign_g -= 2*parameters->CONF_MP_scale_reg*(*assign_y);
					assign_g++;
					assign_y ++;
				}
				else if(noprob_0m){}
				else{
					tmp_scores[get_index2_o3g(length,0,0,0,m)] = *assign_y;
					*assign_g -= 2*parameters->CONF_MP_scale_reg*(*assign_y);
					assign_g++;
					assign_y ++;
				}
				for(int mid=1;mid<m;mid++){
					if(link_0m && mid==c){
						tmp_scores[get_index2_o3g(length,0,0,mid,m)] = *assign_y;
						*assign_g -= 2*parameters->CONF_MP_scale_reg*(*assign_y);
						assign_g++;
						assign_y ++;
					}
					else if(noprob_0m || o1_noprob[get_index2(length,0,mid)]){}
					else{
						tmp_scores[get_index2_o3g(length,0,0,mid,m)] = *assign_y;
						*assign_g -= 2*parameters->CONF_MP_scale_reg*(*assign_y);
						assign_g++;
						assign_y ++;
					}
				}
				//2.2. ordinary ones
				for(int h=1;h<length;h++){	//h>=1
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
					//for g and c
					for(int g=0;g<length;g++){
						if(g<=large && g>=small)
							continue;
						bool link_gh = (x->heads->at(h)==g);
						int noprob_gh = o1_noprob[get_index2(length,g,h)];
						int nonproj_g = (g>=small && g<=large);
						if(link_hm && link_gh && c<0){
							tmp_scores[get_index2_o3g(length,g,h,h,m)] = *assign_y;
							*assign_g -= 2*parameters->CONF_MP_scale_reg*(*assign_y);
							assign_g++;
							assign_y ++;
						}
						else if(noprob_hm || noprob_gh || nonproj_g){}
						else{
							tmp_scores[get_index2_o3g(length,g,h,h,m)] = *assign_y;
							*assign_g -= 2*parameters->CONF_MP_scale_reg*(*assign_y);
							assign_g++;
							assign_y ++;
						}
						for(int mid=small+1;mid<large;mid++){
							if(link_hm && link_gh && mid==c){
								tmp_scores[get_index2_o3g(length,g,h,mid,m)] = *assign_y;
								*assign_g -= 2*parameters->CONF_MP_scale_reg*(*assign_y);
								assign_g++;
								assign_y ++;
							}
							else if(noprob_hm || noprob_gh || nonproj_g || o1_noprob[get_index2(length,h,mid)]){}
							else{
								tmp_scores[get_index2_o3g(length,g,h,mid,m)] = *assign_y;
								*assign_g -= 2*parameters->CONF_MP_scale_reg*(*assign_y);
								assign_g++;
								assign_y ++;
							}
						}
					}
				}
			}
		//5.gradients
		double* tmp_marginals = encodeMarginals_o3g(length,tmp_scores);
		assign_g = gradient;
		REAL gradient_small = parameters->CONF_MP_gradient_small;
		REAL* assign_redata = rearrange_data;
		REAL* aasign_regrad = rearrange_gradient;
		int num_rearrange = 0;
		//loop------------------
			for(int m=1;m<length;m++){
				//2.1 special (0,0,c,m)	when h==0
				int noprob_0m = o1_noprob[get_index2(length,0,m)];
				int link_0m = (x->heads->at(m)==0);
				int c = -1;
				for(int mid=m-1;mid>0;mid--){
					if(x->heads->at(mid)==0){
						c = mid;break;}
				}
				if(link_0m && c<0){
					*assign_g -= tmp_marginals[get_index2_o3g(length,0,0,0,m)];
					//if gradient is too small, just ignore it to avoid numeric issues
					if(*assign_g < gradient_small && *assign_g > -gradient_small){
						*assign_g = 0;
						zero_backward ++;
					}
					else if(parameters->CONF_MP_training_rearrange){
						//re-arrange
						num_rearrange++;
						feat_gen->fill_one(assign_redata,x,0,m,-1,0);
						assign_redata += mach->GetIdim();
						*aasign_regrad = *assign_g;
						aasign_regrad++;
					}
					assign_g++;
				}
				else if(noprob_0m){}
				else{
					*assign_g -= tmp_marginals[get_index2_o3g(length,0,0,0,m)];
					//if gradient is too small, just ignore it to avoid numeric issues
					if(*assign_g < gradient_small && *assign_g > -gradient_small){
						*assign_g = 0;
						zero_backward ++;
					}
					else if(parameters->CONF_MP_training_rearrange){
						//re-arrange
						num_rearrange++;
						feat_gen->fill_one(assign_redata,x,0,m,-1,0);
						assign_redata += mach->GetIdim();
						*aasign_regrad = *assign_g;
						aasign_regrad++;
					}
					assign_g++;
				}
				for(int mid=1;mid<m;mid++){
					if(link_0m && mid==c){
						*assign_g -= tmp_marginals[get_index2_o3g(length,0,0,mid,m)];
						//if gradient is too small, just ignore it to avoid numeric issues
						if(*assign_g < gradient_small && *assign_g > -gradient_small){
							*assign_g = 0;
							zero_backward ++;
						}
						else if(parameters->CONF_MP_training_rearrange){
							//re-arrange
							num_rearrange++;
							feat_gen->fill_one(assign_redata,x,0,m,mid,0);
							assign_redata += mach->GetIdim();
							*aasign_regrad = *assign_g;
							aasign_regrad++;
						}
						assign_g++;
					}
					else if(noprob_0m || o1_noprob[get_index2(length,0,mid)]){}
					else{
						*assign_g -= tmp_marginals[get_index2_o3g(length,0,0,mid,m)];
						//if gradient is too small, just ignore it to avoid numeric issues
						if(*assign_g < gradient_small && *assign_g > -gradient_small){
							*assign_g = 0;
							zero_backward ++;
						}
						else if(parameters->CONF_MP_training_rearrange){
							//re-arrange
							num_rearrange++;
							feat_gen->fill_one(assign_redata,x,0,m,mid,0);
							assign_redata += mach->GetIdim();
							*aasign_regrad = *assign_g;
							aasign_regrad++;
						}
						assign_g++;
					}
				}
				//2.2. ordinary ones
				for(int h=1;h<length;h++){	//h>=1
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
					//for g and c
					for(int g=0;g<length;g++){
						if(g<=large && g>=small)
							continue;
						bool link_gh = (x->heads->at(h)==g);
						int noprob_gh = o1_noprob[get_index2(length,g,h)];
						int nonproj_g = (g>=small && g<=large);
						if(link_hm && link_gh && c<0){
							*assign_g -= tmp_marginals[get_index2_o3g(length,g,h,h,m)];
							//if gradient is too small, just ignore it to avoid numeric issues
							if(*assign_g < gradient_small && *assign_g > -gradient_small){
								*assign_g = 0;
								zero_backward ++;
							}
							else if(parameters->CONF_MP_training_rearrange){
								//re-arrange
								num_rearrange++;
								feat_gen->fill_one(assign_redata,x,h,m,-1,g);
								assign_redata += mach->GetIdim();
								*aasign_regrad = *assign_g;
								aasign_regrad++;
							}
							assign_g++;
						}
						else if(noprob_hm || noprob_gh || nonproj_g){}
						else{
							*assign_g -= tmp_marginals[get_index2_o3g(length,g,h,h,m)];
							//if gradient is too small, just ignore it to avoid numeric issues
							if(*assign_g < gradient_small && *assign_g > -gradient_small){
								*assign_g = 0;
								zero_backward ++;
							}
							else if(parameters->CONF_MP_training_rearrange){
								//re-arrange
								num_rearrange++;
								feat_gen->fill_one(assign_redata,x,h,m,-1,g);
								assign_redata += mach->GetIdim();
								*aasign_regrad = *assign_g;
								aasign_regrad++;
							}
							assign_g++;
						}
						for(int mid=small+1;mid<large;mid++){
							if(link_hm && link_gh && mid==c){
								*assign_g -= tmp_marginals[get_index2_o3g(length,g,h,mid,m)];
								//if gradient is too small, just ignore it to avoid numeric issues
								if(*assign_g < gradient_small && *assign_g > -gradient_small){
									*assign_g = 0;
									zero_backward ++;
								}
								else if(parameters->CONF_MP_training_rearrange){
									//re-arrange
									num_rearrange++;
									feat_gen->fill_one(assign_redata,x,h,m,mid,g);
									assign_redata += mach->GetIdim();
									*aasign_regrad = *assign_g;
									aasign_regrad++;
								}
								assign_g++;
							}
							else if(noprob_hm || noprob_gh || nonproj_g || o1_noprob[get_index2(length,h,mid)]){}
							else{
								*assign_g -= tmp_marginals[get_index2_o3g(length,g,h,mid,m)];
								//if gradient is too small, just ignore it to avoid numeric issues
								if(*assign_g < gradient_small && *assign_g > -gradient_small){
									*assign_g = 0;
									zero_backward ++;
								}
								else if(parameters->CONF_MP_training_rearrange){
									//re-arrange
									num_rearrange++;
									feat_gen->fill_one(assign_redata,x,h,m,mid,g);
									assign_redata += mach->GetIdim();
									*aasign_regrad = *assign_g;
									aasign_regrad++;
								}
								assign_g++;
							}
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
	cout << "Also skip long sentences " << sentences_skip_long << endl;

	for(int i=0;i<sentences;i++)
		delete [](all_noprob_o1[i]);
	delete []all_noprob_o1;
}

vector<int>* MP3_o3g::each_test_one(DependencyInstance* x)
{
	double* score_of_o1 = 0;
	double* score_of_o2sib = 0;
	double* score_of_o2g = 0;
	int length = x->length();
	FeatureGenO1* feat_temp_o1 = new FeatureGenO1(dict,parameters->CONF_x_window,
			parameters->CONF_add_distance,parameters->CONF_add_pos,parameters->CONF_add_direction);
	//filter
	double* scores_o1_filter = get_scores_o1(x,parameters,mach_o1_filter,feat_temp_o1);	//same parameters
	bool *whether_cut_o1 = get_noprob_o1(length,scores_o1_filter);
	double *tmp_scores = get_scores_o3g(x,parameters,mach,feat_gen,whether_cut_o1);
	delete []scores_o1_filter;
	//combine scores??
	if(parameters->CONF_MP_o1mach.length() > 0){
		score_of_o1 = get_scores_o1(x,parameters,mach_o1_score,feat_temp_o1);
	}
	delete feat_temp_o1;

	//o2sib
	if(parameters->CONF_MP_o2sibmach.length() > 0){
		FeatureGenO2sib* feat_temp_o2sib = new FeatureGenO2sib(dict,parameters->CONF_x_window,
				parameters->CONF_add_distance,parameters->CONF_add_pos,parameters->CONF_add_direction);
		score_of_o2sib = get_scores_o2sib(x,parameters,mach_o2g_score,feat_temp_o2sib,whether_cut_o1);	//same parameters
		delete feat_temp_o2sib;
	}

	//combining
	for(int m=1;m<length;m++){
		double s_0m = 0,s_0xm=0,s_00m = 0;
		if(score_of_o1)
			s_0m = score_of_o1[get_index2(length,0,m)];
		if(score_of_o2sib)
			s_0xm = score_of_o2sib[get_index2_o2sib(length,0,0,m)];
		if(score_of_o2g)
			s_00m = score_of_o2g[get_index2_o2g(length,0,0,m)];
		tmp_scores[get_index2_o3g(length,0,0,0,m)] += s_0m + s_0xm + s_00m;
		for(int c=m-1;c>0;c--){
			if(score_of_o2sib)
				s_0xm = score_of_o2sib[get_index2_o2sib(length,0,c,m)];
			tmp_scores[get_index2_o3g(length,0,0,c,m)] += s_0m + s_0xm + s_00m;
		}
	}
	for(int s=1;s<length;s++){
		for(int t=s+1;t<length;t++){
			double s_st=0,s_ts=0;
			if(score_of_o1){
				s_st = score_of_o1[get_index2(length,s,t)];
				s_ts = score_of_o1[get_index2(length,t,s)];
			}
			for(int g=0;g<length;g++){
				if(g>=s && g<=t)	//no non-projective
					continue;
				double s_sxt=0,s_txs=0,s_gst=0,s_gts=0;
				if(score_of_o2sib){
					s_sxt = score_of_o2sib[get_index2_o2sib(length,s,s,t)];
					s_txs = score_of_o2sib[get_index2_o2sib(length,t,t,s)];
				}
				if(score_of_o2g){
					s_gst = score_of_o2g[get_index2_o2g(length,g,s,t)];
					s_gts = score_of_o2g[get_index2_o2g(length,g,t,s)];
				}
				tmp_scores[get_index2_o3g(length,g,s,s,t)] += s_st + s_sxt + s_gst;
				tmp_scores[get_index2_o3g(length,g,t,t,s)] += s_ts + s_txs + s_gts;
				for(int c=s+1;c<t;c++){
					double s_sct=0,s_tcs=0;
					if(score_of_o2sib){
						s_sct = score_of_o2sib[get_index2_o2sib(length,s,c,t)];
						s_tcs = score_of_o2sib[get_index2_o2sib(length,t,c,s)];
					}
					tmp_scores[get_index2_o3g(length,g,s,c,t)] += s_st + s_sct + s_gst;
					tmp_scores[get_index2_o3g(length,g,t,c,s)] += s_ts + s_tcs + s_gts;
				}
			}
		}
	}

	vector<int> *ret = decodeProjective_o3g(length,tmp_scores);
	delete []tmp_scores;
	delete []score_of_o1;
	delete []score_of_o2sib;
	delete []whether_cut_o1;
	return ret;
}

