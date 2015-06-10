/*
 * MP1_o1.h
 *
 *  Created on: 2015Äê6ÔÂ10ÈÕ
 *      Author: zzs
 */

#ifndef PROCESS_PROB_MP1_O1_H_
#define PROCESS_PROB_MP1_O1_H_

//the o1 probabilistic model
#include "ProbProcess.h"

class MP1_o1: public ProbProcess{
protected:
	virtual void nn_train_one_iter();
	virtual vector<int>* each_test_one(DependencyInstance* x){
		return parse_o1(x);
	}
public:
	MP1_o1(string c):ProbProcess(c){}
	virtual ~MP1_o1(){}
};


#endif /* PROCESS_PROB_MP1_O1_CPP_ */
