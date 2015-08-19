/*
 * MP3_o3g.h
 *
 *  Created on: 2015Äê7ÔÂ16ÈÕ
 *      Author: zzs
 */

#ifndef PROCESS_PROB_MP3_O3G_H_
#define PROCESS_PROB_MP3_O3G_H_

//the o3g probabilistic model
#include "ProbProcess.h"
#include "../parts/FeatureGenO3g.h"

class MP3_o3g: public ProbProcess{
protected:
	NNInterface * mach_o1_score;
	NNInterface * mach_o2g_score;
	NNInterface * mach_o1_filter;	//2-output unit
	virtual void nn_train_one_iter();
	virtual vector<int>* each_test_one(DependencyInstance* x);

	virtual void each_get_featgen(int if_testing){
		if(if_testing){
			if(! feat_gen)	//when testing
				feat_gen = new FeatureGenO3g(dict,parameters->CONF_x_window,parameters->CONF_add_distance,
						parameters->CONF_add_pos,parameters->CONF_add_direction,parameters->CONF_NN_MVEC);
			feat_gen->deal_with_corpus(dev_test_corpus);
		}
		else{
			feat_gen = new FeatureGenO3g(dict,parameters->CONF_x_window,parameters->CONF_add_distance,
					parameters->CONF_add_pos,parameters->CONF_add_direction,parameters->CONF_NN_MVEC);
			feat_gen->deal_with_corpus(training_corpus);
		}
	}

public:
	MP3_o3g(string c):ProbProcess(c){
		if(parameters->CONF_MP_o1mach.length() > 0)
			mach_o1_score = NNInterface::Read(parameters->CONF_MP_o1mach);
		else
			mach_o1_score = 0;
		if(parameters->CONF_MP_o2sibmach.length() > 0)
			mach_o2g_score = NNInterface::Read(parameters->CONF_MP_o2sibmach);
		else
			mach_o2g_score = 0;
		mach_o1_filter = NNInterface::Read(parameters->CONF_NN_highO_o1mach);
	}
	virtual ~MP3_o3g(){}
};


#endif /* PROCESS_PROB_MP3_O3G_H_ */
