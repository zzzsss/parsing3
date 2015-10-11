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

//prepare bool for high-order filtering
//-score is the rearrange-score and here must be unlabeled prob
//-cut>0 means absolute,cut<0 means compared to max
bool* get_cut_o1(int len,double* scores,double cut)
{
	double* scores_max = new double[len];
	for(int m=1;m<len;m++){	//each token's max-score head
		scores_max[m] = scores[get_index2(len,0,m)];
		for(int h=1;h<len;h++){
			if(h==m) continue;
			double tmp_s = scores[get_index2(len,h,m)];
			if(tmp_s > scores_max[m])
				scores_max[m] = tmp_s;
		}
	}
	bool* ret = new bool[len*len];
	for(int m=1;m<len;m++){
		for(int h=0;h<len;h++){
			if(h==m) continue;
			if(cut<0)
				ret[get_index2(len,h,m)] = (scores[get_index2(len,h,m)] < (scores_max[m]*-1*cut));
			else
				ret[get_index2(len,h,m)] = (scores[get_index2(len,h,m)] < cut);
		}
	}
	return ret;
}

//-----------------------------------rearrange------------------------------------------
// x for length, m for odim, t for outputs, fscores for forward-scores(no arrange)
// prob_output for output is label+1; prob_trans to trans prob only when prob_output
static double* rearrange_scores_o1(DependencyInstance* x,Csnn* m,nn_input* t,double* fscores,
		bool prob_ouput,bool prob_trans)
{
	int length = x->length();
	int fs_dim = m->get_odim();
}

