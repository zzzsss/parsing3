/*
 * Process_scores.cpp
 *
 *  Created on: Oct 11, 2015
 *      Author: zzs
 */

#include "Process.h"

// the routine for getting scores
// -- m means the current machine(might not be the same as options)
// -- t for assigning inputs, need outside to delete it
// -- testing for the forward of nn
double* Process::forward_scores_o1(DependencyInstance* x,Csnn* m,nn_input** t,nn_input_helper* h,int testing)
{
	//default order1 parsing
	int odim = m->get_odim();	//1 or 2 for no-labeled, otherwise...
	int length = x->length();
	//prepare scores
	int num_pair_togo = 0;
	vector<int>* the_inputs = new vector<int>();
	//loop --- (h,m)
	for(int m=1;m<length;m++){
		for(int h=0;h<length;h++){
			if(m != h){
				the_inputs->push_back(h);the_inputs->push_back(m);
				num_pair_togo ++;
			}
		}
	}
	(*t) = new nn_input(num_pair_togo,2,the_inputs,x->index_forms,x->index_pos,h);
	double* tmp_scores = m->forward(*t,testing);
	return tmp_scores;
}
double* Process::forward_scores_o2sib(DependencyInstance* x,Csnn* m,nn_input** t,nn_input_helper* h,int testing,bool* cut_o1)
{
	//o2sib
	int odim = m->get_odim();	//1 or 2 for no-labeled, otherwise...
	int length = x->length();
	//prepare scores
	int num_togo = 0;
	vector<int>* the_inputs = new vector<int>();
	bool* score_o1 = cut_o1;
	//loop --- (h,m,s)
	for(int m=1;m<length;m++){
		for(int h=0;h<length;h++){
			if(h==m)
				continue;
			bool norpob_hm = score_o1[get_index2(length,h,m)];
			//h,m,-1
			if(!norpob_hm){
				the_inputs->push_back(h);the_inputs->push_back(m);the_inputs->push_back(-1);
				num_togo += 1;
			}
			//h,m,c
			int small = GET_MIN_ONE(m,h);
			int large = GET_MAX_ONE(m,h);
			if(!norpob_hm){
				for(int c=small+1;c<large;c++){
					if(!score_o1[get_index2(length,h,c)]){
						the_inputs->push_back(h);the_inputs->push_back(m);the_inputs->push_back(c);
						num_togo += 1;
					}
				}
			}
		}
	}
	(*t) = new nn_input(num_togo,3,the_inputs,x->index_forms,x->index_pos,h);
	double* tmp_scores = m->forward(*t,testing);
	return tmp_scores;
}
double* Process::forward_scores_o3g(DependencyInstance* x,Csnn* m,nn_input** t,nn_input_helper* h,int testing,bool* cut_o1)
{
	//o3g
	int odim = m->get_odim();	//1 or 2 for no-labeled, otherwise...
	int length = x->length();
	//prepare scores
	int num_togo = 0;
	vector<int>* the_inputs = new vector<int>();
	bool* score_o1 = cut_o1;
	//loop --- (h,m,s,g)
	//1. 0,0,c,m
	for(int m=1;m<length;m++){
		//0,0,0,m
		if(!score_o1[get_index2(length,0,m)]){
			{the_inputs->push_back(0);the_inputs->push_back(m);the_inputs->push_back(-1);the_inputs->push_back(-1);}
			//0,0,c,m
			for(int c=m-1;c>0;c--){
				if(!score_o1[get_index2(length,0,c)])
					{the_inputs->push_back(0);the_inputs->push_back(m);the_inputs->push_back(c);the_inputs->push_back(-1);}
			}
		}
	}
	//2. g,h,c,m
	for(int h=1;h<length;h++){
		for(int m=1;m<length;m++){
			if(h==m)
				continue;
			if(!score_o1[get_index2(length,h,m)]){
				int small = GET_MIN_ONE(h,m);
				int large = GET_MAX_ONE(h,m);
				for(int g=0;g<small;g++){
					if(!score_o1[get_index2(length,g,h)]){
						{the_inputs->push_back(h);the_inputs->push_back(m);the_inputs->push_back(-1);the_inputs->push_back(g);}
						for(int c=small+1;c<large;c++){
							if(!score_o1[get_index2(length,h,c)])
							{the_inputs->push_back(h);the_inputs->push_back(m);the_inputs->push_back(c);the_inputs->push_back(g);}
						}
					}
				}
				for(int g=large+1;g<length;g++){
					if(!score_o1[get_index2(length,g,h)]){
						{the_inputs->push_back(h);the_inputs->push_back(m);the_inputs->push_back(-1);the_inputs->push_back(g);}
						for(int c=small+1;c<large;c++){
							if(!score_o1[get_index2(length,h,c)])
							{the_inputs->push_back(h);the_inputs->push_back(m);the_inputs->push_back(c);the_inputs->push_back(g);}
						}
					}
				}
			}
		}
	}
	(*t) = new nn_input(num_togo,4,the_inputs,x->index_forms,x->index_pos,h);
	double* tmp_scores = m->forward(*t,testing);
	return tmp_scores;
}

//-----------------------------------rearrange------------------------------------------
#include "Process_trans.cpp"	//special
// x for length, m for odim, t for outputs, fscores for forward-scores(no arrange)
// prob_output for output is label+1; prob_trans to trans prob only when prob_output
double* Process::rearrange_scores_o1(DependencyInstance* x,Csnn* m,nn_input* the_inputs,double* fscores,
		bool prob_output,bool prob_trans)
{
	const int THE_DIM = 2;
	int length = x->length();
	int fs_dim = m->get_odim();
	int num_label = fs_dim;
	if(prob_output)	//if prob output, there is one for no-rel
		num_label -= 1;
	//prepare
	double* rscores = new double[length*length*num_label];
	for(int i=0;i<length*length*num_label;i++)
		rscores[i] = DOUBLE_LARGENEG;
	//make sure the width is THE_DIM
	if(the_inputs->get_numw() != THE_DIM){
		cerr << "!!!Wrong nn_input" << endl;
		return 0;
	}
	//get scores
	vector<int>* inputs_list = the_inputs->inputs;
	double *to_assign = fscores;
	for(int i=0;i<the_inputs->num_inst;i+=THE_DIM){
		int tmph = inputs_list->at(i);
		int tmpm = inputs_list->at(i+1);
		memcpy(&rscores[get_index2(length,tmph,tmpm,0,num_label)],to_assign,sizeof(double)*num_label);
		//(if prob, the no-rel must be at the end, which is different from before)
		to_assign += fs_dim;
	}
	//if prob-transfrom
	if(prob_output && prob_trans){
		//1.prepare the ignored nope prob
		double* nope_probs = new double[length*length];
		for(int i=0;i<length*length;i++)
			nope_probs[i] = 1;		//set to one firstly
		double *to_assign = fscores;
		for(int i=0;i<the_inputs->num_inst;i+=THE_DIM){
			int tmph = inputs_list->at(i);
			int tmpm = inputs_list->at(i+1);
			nope_probs[get_index2(length,tmph,tmpm)] = to_assign[fs_dim-1];	//the last one
			to_assign += fs_dim;
		}
		trans_o1(rscores,nope_probs,length,num_label);
		delete []nope_probs;
	}
	return rscores;
}

double* Process::rearrange_scores_o2sib(DependencyInstance* x,Csnn* m,nn_input* the_inputs,double* fscores,
		bool prob_output,bool prob_trans,double* rscores_o1)
{
	const int THE_DIM = 3;
	int length = x->length();
	int fs_dim = m->get_odim();
	int num_label = fs_dim;
	if(prob_output)	//if prob output, there is one for no-rel
		num_label -= 1;
	//prepare
	double* rscores = new double[length*length*length*num_label];
	for(int i=0;i<length*length*length*num_label;i++)
		rscores[i] = DOUBLE_LARGENEG;
	//make sure the width is THE_DIM
	if(the_inputs->get_numw() != THE_DIM){
		cerr << "!!!Wrong nn_input" << endl;
		return 0;
	}
	//get scores
	vector<int>* inputs_list = the_inputs->inputs;
	double *to_assign = fscores;
	for(int i=0;i<the_inputs->num_inst;i+=THE_DIM){
		int tmph = inputs_list->at(i);
		int tmpm = inputs_list->at(i+1);
		int tmps = inputs_list->at(i+2);
		if(tmps<0)
			tmps = tmph;
		memcpy(&rscores[get_index2_o2sib(length,tmph,tmps,tmpm,0,num_label)],to_assign,sizeof(double)*num_label);
		//(if prob, the no-rel must be at the end, which is different from before)
		to_assign += fs_dim;
	}
	//if prob-transfrom
	if(prob_output && prob_trans){
		//1.prepare the ignored nope prob
		double* nope_probs = new double[length*length*length];
		for(int i=0;i<length*length*length;i++)
			nope_probs[i] = 1;		//set to one firstly
		double *to_assign = fscores;
		for(int i=0;i<the_inputs->num_inst;i+=THE_DIM){
			int tmph = inputs_list->at(i);
			int tmpm = inputs_list->at(i+1);
			int tmps = inputs_list->at(i+2);
			if(tmps<0)
				tmps = tmph;
			nope_probs[get_index2_o2sib(length,tmph,tmps,tmpm)] = to_assign[fs_dim-1];	//the last one
			to_assign += fs_dim;
		}
		trans_o2sib(rscores,nope_probs,length,num_label);
		delete []nope_probs;
	}
	//if combining scores
	if(rscores_o1){
		//the provided score must be re-arranged and must have the same label-dim
		for(int m=1;m<length;m++){
			for(int h=0;h<length;h++){
				if(m!=h){
					for(int la=0;la<num_label;la++){
					double score_tmp = rscores_o1[get_index2(length,h,m,la,num_label)];
					rscores[get_index2_o2sib(length,h,h,m,la,num_label)] += score_tmp;
					for(int c=h+1;c<m;c++)
						rscores[get_index2_o2sib(length,h,c,m,la,num_label)] += score_tmp;
					for(int c=m+1;c<h;c++)
						rscores[get_index2_o2sib(length,h,c,m,la,num_label)] += score_tmp;
					}
				}
			}
		}
	}
	return rscores;
}

double* rearrange_scores_o3g(DependencyInstance* x,Csnn* m,nn_input* the_inputs,double* fscores,
		bool prob_output,bool prob_trans,double* rscores_o1,double* rscores_o2sib)
{
	const int THE_DIM = 3;
	int length = x->length();
	int fs_dim = m->get_odim();
	int num_label = fs_dim;
	if(prob_output)	//if prob output, there is one for no-rel
		num_label -= 1;
	//prepare
	double* rscores = new double[length*length*length*length*num_label];
	for(int i=0;i<length*length*length*length*num_label;i++)
		rscores[i] = DOUBLE_LARGENEG;
	//make sure the width is THE_DIM
	if(the_inputs->get_numw() != THE_DIM){
		cerr << "!!!Wrong nn_input" << endl;
		return 0;
	}
	//get scores
	vector<int>* inputs_list = the_inputs->inputs;
	double *to_assign = fscores;
	for(int i=0;i<the_inputs->num_inst;i+=THE_DIM){
		int tmph = inputs_list->at(i);
		int tmpm = inputs_list->at(i+1);
		int tmps = inputs_list->at(i+2);
		int tmpg = inputs_list->at(i+3);
		if(tmps<0)
			tmps = tmph;
		if(tmpg<0)
			tmpg = 0;
		memcpy(&rscores[get_index2_o3g(length,tmpg,tmph,tmps,tmpm,0,num_label)],to_assign,sizeof(double)*num_label);
		//(if prob, the no-rel must be at the end, which is different from before)
		to_assign += fs_dim;
	}
	//if prob-transfrom
	if(prob_output && prob_trans){
		//1.prepare the ignored nope prob
		double* nope_probs = new double[length*length*length*length];
		for(int i=0;i<length*length*length*length;i++)
			nope_probs[i] = 1;		//set to one firstly
		double *to_assign = fscores;
		for(int i=0;i<the_inputs->num_inst;i+=THE_DIM){
			int tmph = inputs_list->at(i);
			int tmpm = inputs_list->at(i+1);
			int tmps = inputs_list->at(i+2);
			int tmpg = inputs_list->at(i+3);
			if(tmps<0)
				tmps = tmph;
			if(tmpg<0)
				tmpg = 0;
			nope_probs[get_index2_o3g(length,tmpg,tmph,tmps,tmpm)] = to_assign[fs_dim-1];	//the last one
			to_assign += fs_dim;
		}
		trans_o3g(rscores,nope_probs,length,num_label);
		delete []nope_probs;
	}
	//if combining scores
	if(rscores_o1 || rscores_o2sib){
		//the provided score must be re-arranged and must have the same label-dim
		//1.0,0,?,m
		for(int m=1;m<length;m++){
			for(int la=0;la<num_label;la++){	//LABEL
			double s_0m = 0,s_0xm=0;
			if(rscores_o1)
				s_0m = rscores_o1[get_index2(length,0,m,la,num_label)];
			if(rscores_o2sib)
				s_0xm = rscores_o2sib[get_index2_o2sib(length,0,0,m,la,num_label)];
			rscores[get_index2_o3g(length,0,0,0,m,la,num_label)] += s_0m + s_0xm;
			for(int c=m-1;c>0;c--){
				if(rscores_o2sib)
					s_0xm = rscores_o2sib[get_index2_o2sib(length,0,c,m,la,num_label)];
				rscores[get_index2_o3g(length,0,0,c,m,la,num_label)] += s_0m + s_0xm;
			}
			}
		}
		for(int s=1;s<length;s++){
			for(int t=s+1;t<length;t++){
				for(int la=0;la<num_label;la++){	//LABEL
				double s_st=0,s_ts=0;
				if(rscores_o1){
					s_st = rscores_o1[get_index2(length,s,t,la,num_label)];
					s_ts = rscores_o1[get_index2(length,t,s,la,num_label)];
				}
				for(int g=0;g<length;g++){
					if(g>=s && g<=t)	//no non-projective
						continue;
					double s_sxt=0,s_txs=0;
					if(rscores_o2sib){
						s_sxt = rscores_o2sib[get_index2_o2sib(length,s,s,t,la,num_label)];
						s_txs = rscores_o2sib[get_index2_o2sib(length,t,t,s,la,num_label)];
					}
					rscores[get_index2_o3g(length,g,s,s,t,la,num_label)] += s_st + s_sxt;
					rscores[get_index2_o3g(length,g,t,t,s,la,num_label)] += s_ts + s_txs;
					for(int c=s+1;c<t;c++){
						double s_sct=0,s_tcs=0;
						if(rscores_o2sib){
							s_sct = rscores_o2sib[get_index2_o2sib(length,s,c,t,la,num_label)];
							s_tcs = rscores_o2sib[get_index2_o2sib(length,t,c,s,la,num_label)];
						}
						rscores[get_index2_o3g(length,g,s,c,t,la,num_label)] += s_st + s_sct;
						rscores[get_index2_o3g(length,g,t,c,s,la,num_label)] += s_ts + s_tcs;
					}
				}
				}
			}
		}
	}
	return rscores;
}


