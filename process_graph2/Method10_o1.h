/*
 * METHOD10_o1.h
 *
 *  Created on: 2015.8.20
 *      Author: zzs
 */

#ifndef METHOD_O10_PROCESS_H_
#define METHOD_O10_PROCESS_H_

// different from M1 in only using per-sentence training

#include "../process_graph/Process.h"

class Method10_o1: public Process{
protected:
	REAL* data;
	REAL* gradient;
	int alloc_sample_size;

	virtual int each_get_mach_outdim(){return 2;}
	virtual void nn_train_one_iter();				//different training ways
	virtual vector<int>* each_test_one(DependencyInstance* x){
		return parse_o1(x);
	}

	//no-use
	virtual void each_prepare_data_oneiter(){}
	virtual REAL* each_next_data(int*){}
	virtual void each_get_grad(int){}

public:
	Method10_o1(string c):Process(c),data(0),gradient(0),alloc_sample_size(0){}
	virtual ~Method10_o1(){}
};

#endif
