/*
 * M3_pr.h
 *
 *  Created on: Dec 10, 2015
 *      Author: zzs
 */

#ifndef GRAPH_BASED_M3_PR_H_
#define GRAPH_BASED_M3_PR_H_

#include "Process.h"

//Perceptron for order2
class M3_pro2: public Process{
private:
	CsnnO1 *mfo1;	//filter
	CsnnO1 *mso1;	//scorer
protected:
	virtual void each_create_machine();
	virtual void each_train_one_iter();
	virtual void each_test_one(DependencyInstance*,int);		//set predict_head or predict_deprels here

	virtual void train();		//override
	virtual void test(string);	//override

	//static methods
	void get_nninput_o1(DependencyInstance* x,nn_input** good,nn_input**bad);
	void get_nninput_o2sib(DependencyInstance* x,nn_input** good,nn_input**bad);
public:
	M3_pro2(string cname):Process(cname){
		//need special conf for perceptron learning
		mfo1 = dynamic_cast<CsnnO1*>(Csnn::read(hp->CONF_score_mach_fo1));
		//mso1 and mach(o2sib) are read in each_create_machine or test

		//conf
		hp->CONF_embed_WL = "";
		hp->CONF_embed_EM = "";
		hp->CONF_score_prob = 0;
	}
};



#endif /* GRAPH_BASED_M3_PR_H_ */
