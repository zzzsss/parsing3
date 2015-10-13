/*
 * HypherParameters.h
 *
 *  Created on: Oct 10, 2015
 *      Author: zzs
 */

#ifndef PARTS_HYPHERPARAMETERS_H_
#define PARTS_HYPHERPARAMETERS_H_

#include "../csnn/options.h"
#include <string>
using std::string;

class HypherParameters{
public:
	nn_options hp_nn;	//used by nn

public:
//1.0
int CONF_method;	//which method
//1.1-files
string CONF_train_file;	//the training file
string CONF_dev_file;	//dev files
string CONF_test_file;	//test files
string CONF_output_file;
string CONF_gold_file;	//golden files
//1.2-other files
string CONF_dict_file;		//for dictionary
string CONF_mach_name;		//mach name
string CONF_mach_cur_suffix;
string CONF_mach_best_suffix;
//1.3-some training criteria
double CONF_NN_LRATE;
int CONF_NN_ITER;
int CONF_NN_ITER_decrease;		//at lease cut lrate this times when stopping(so real iters maybe more than iter)
int CONF_NN_ITER_force_half;	//force cut half if no cutting for how many iters
double CONF_NN_LMULT;			//-1~0: schedule rate
double CONF_NN_WD;				//weight decay

double CONF_NN_resample;				//re-sample rate
int CONF_NN_FIXVEC;						//no update on embeddings any longer

//1.4-others
int CONF_dict_remove;	//remove words appears only less than this times
int CONF_random_seed;
int CONF_minibatch;		//how many sentences for a batch(this is not bsize of nn)
int CONF_labeled;		//labeled or not

//1.5-scores
int CONF_score_prob;	//whether give transform score, only for M1 (0,1)
double CONF_score_o1filter_cut;
int CONF_score_combine_o1;
int CONF_score_combine_o2sib;

};



#endif /* PARTS_HYPHERPARAMETERS_H_ */
