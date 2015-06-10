/*
 * ProbProcess.h
 *
 *  Created on: 2015Äê6ÔÂ10ÈÕ
 *      Author: zzs
 */

#ifndef PROCESS_PROB_PROBPROCESS_H_
#define PROCESS_PROB_PROBPROCESS_H_

//the base class for all MP methods
//--- the MP classes do not inherit from Method* classes because of different training-codes
//--- this is again bad design for implementing-reuse,but have to re-implement the test codes although quite similar to those

#include "../process_graph/Process.h"

class ProbProcess: public Process{
protected:
	static const double GRADIENT_SMALL;

	REAL* data;
	REAL* gradient;
	int alloc_sample_size;

	virtual int each_get_mach_outdim(){return 1;}	//only one output scores
	virtual void nn_train_one_iter()=0;				//different training ways
	virtual vector<int>* each_test_one(DependencyInstance* x)=0;

	//no-use
	virtual void each_prepare_data_oneiter(){}
	virtual REAL* each_next_data(int*){}
	virtual void each_get_grad(int){}

public:
	ProbProcess(string c):Process(c),data(0),gradient(0),alloc_sample_size(0){
		parameters->CONF_score_prob = 0;	//no-log-transform
	}
	virtual ~ProbProcess(){}
};


#endif /* PROCESS_PROB_PROBPROCESS_H_ */
