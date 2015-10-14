/*
 * M1_p1o1.h
 *
 *  Created on: Oct 14, 2015
 *      Author: zzs
 */

#ifndef GRAPH_BASED_M1_P1O1_H_
#define GRAPH_BASED_M1_P1O1_H_

#include "Process.h"
//the method 1: probability 1 order 1
class M1_p1o1: public Process{
protected:
	virtual void each_create_machine();
	virtual void each_train_one_iter();
	virtual void each_test_one(DependencyInstance*);		//set predict_head or predict_deprels here

public:
	M1_p1o1(string cname):Process(cname){
	}
};



#endif /* GRAPH_BASED_M1_P1O1_H_ */
