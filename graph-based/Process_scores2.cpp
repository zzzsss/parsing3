/*
 * Process_scores2.cpp
 *
 *  Created on: Oct 13, 2015
 *      Author: zzs
 */
#include "Process.h"
//1.for getting cut

//prepare bool for high-order filtering
//-score is the rearrange-score and here must be unlabeled prob
//-cut>0 means absolute,cut<0 means compared to max
static bool* TMP_get_cut_o1(int len,double* scores,double cut)
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

bool* Process::get_cut_o1(DependencyInstance* x,CsnnO1* o1_filter,Dictionary *dict,double cut)
{
	if(o1_filter->get_odim() != 2)
		cerr << "Bad mach as filter" << endl;
	nn_input* o1f_the_input;
	double* scores = get_scores_o1(x,o1_filter,dict,0);	//no trans
	bool* o1f_cut = TMP_get_cut_o1(x->length(),scores,cut);
	delete o1f_the_input;
	delete []scores;
	return o1f_cut;
}

//2.the getting-score functions for parsing
//used when testing or possible parsing in training
double* Process::get_scores_o1(DependencyInstance* x,Csnn* m,Dictionary* dict,bool trans)
{
	//1.get the numbers and informations
	int dictionary_labelnum = dict->getnum_deprel();
	int mach_outputnum = m->get_odim();
	bool is_prob = ((mach_outputnum==2) || (mach_outputnum > dictionary_labelnum));
	bool is_trans = trans;
	//2.getting the scores
	nn_input* the_input;
	double* fscores = forward_scores_o1(x,m,&the_input,dict->get_helper(),1);	//testing-mode, forward scores
	double* rscores = rearrange_scores_o1(x,m,the_input,fscores,is_prob,is_trans);
	delete the_input;
	delete []fscores;
	return rscores;
}

double* Process::get_scores_o2sib(DependencyInstance* x,Csnn* m,Dictionary* dict,bool trans,
		bool* cut_o1,double* rscores_o1)
{
	//1.get the numbers and informations
	int dictionary_labelnum = dict->getnum_deprel();
	int mach_outputnum = m->get_odim();
	bool is_prob = ((mach_outputnum==2) || (mach_outputnum > dictionary_labelnum));
	bool is_trans = trans;
	//2.getting the scores
	nn_input* the_input;
	double* fscores = forward_scores_o2sib(x,m,&the_input,dict->get_helper(),1,cut_o1);	//testing-mode, forward scores
	double* rscores = rearrange_scores_o2sib(x,m,the_input,fscores,is_prob,is_trans,rscores_o1);
	delete the_input;
	delete []fscores;
	return rscores;
}

double* Process::get_scores_o3g(DependencyInstance* x,Csnn* m,Dictionary* dict,bool trans,
		bool* cut_o1,double* rscores_o1,double* rscores_o2sib)
{
	//1.get the numbers and informations
	int dictionary_labelnum = dict->getnum_deprel();
	int mach_outputnum = m->get_odim();
	bool is_prob = ((mach_outputnum==2) || (mach_outputnum > dictionary_labelnum));
	bool is_trans = trans;
	//2.getting the scores
	nn_input* the_input;
	double* fscores = forward_scores_o3g(x,m,&the_input,dict->get_helper(),1,cut_o1);	//testing-mode, forward scores
	double* rscores = rearrange_scores_o3g(x,m,the_input,fscores,is_prob,is_trans,rscores_o1,rscores_o2sib);
	delete the_input;
	delete []fscores;
	return rscores;
}

