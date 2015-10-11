/*
 * Process.h
 *
 *  Created on: Oct 10, 2015
 *      Author: zzs
 */

#ifndef GRAPH_BASED_PROCESS_H_
#define GRAPH_BASED_PROCESS_H_

#include "../parts/HypherParameters.h"
#include "../parts/Dictionary.h"
#include "../tools/CONLLReader.h"
#include "../tools/DependencyEvaluator.h"
#include "../csnn/Csnn.h"
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
using namespace std;

#define DOUBLE_LARGENEG -10000000.0		//maybe it is enough
#define GET_MAX_ONE(a,b) (((a)>(b))?(a):(b))
#define GET_MIN_ONE(a,b) (((a)>(b))?(b):(a))

class Process{
protected:
	//data
	HypherParameters* hp;
	Dictionary* dict;
	vector<DependencyInstance*>* training_corpus;
	vector<DependencyInstance*>* dev_test_corpus;
	//when training
	REAL cur_lrate;
	int cur_iter;
	double * dev_results;	//the results of dev-data
	//--lrate schedule
	int lrate_cut_times;		//number of times of lrate cut (due to dev-result drop)
	int lrate_force_cut_times;	//force cut times
	int last_cut_iter;			//last cutting time (after the iter)
	//mach
	Csnn* mach;

	//0.virtuals
	virtual void each_create_machine();
	virtual void each_train_one_iter();
	virtual void each_test_one(DependencyInstance*);		//set predict_head or predict_deprels here

	//1.1:helper-learning rate
	int set_lrate_one_iter();	//lrate schedule
	virtual int whether_keep_trainning();
	//1.2:helper-for training
	void nn_train_prepare();

	//2.test and dev
	double nn_dev_test(string to_test,string output,string gold);

	//3.1:forward scores and rearrange scores
	static double* forward_scores_o1(DependencyInstance* x,Csnn* m,nn_input** t,nn_input_helper* h,int testing);
	static bool* get_cut_o1(int len,double* scores,double cut);
	static double* forward_scores_o2sib(DependencyInstance* x,Csnn* m,nn_input** t,nn_input_helper* h,int testing,bool* cut_o1);
	static double* forward_scores_o3g(DependencyInstance* x,Csnn* m,nn_input** t,nn_input_helper* h,int testing,bool* cut_o1);
	static double* rearrange_scores_o1(DependencyInstance* x,Csnn* m,nn_input* t,double* fscores,bool prob_ouput,bool prob_trans);
	static double* rearrange_scores_o2sib(DependencyInstance* x,Csnn* m,nn_input* t,double* fscores,bool prob_ouput,bool prob_trans,double* rscores_o1);
	static double* rearrange_scores_o3g(DependencyInstance* x,Csnn* m,nn_input* t,double* fscores,bool prob_ouput,bool prob_trans,double* rscores_o1,double* rscores_o2sib);

public:
	Process(string);
	virtual ~Process(){}
	virtual void train();
	virtual void test(string);
};


#endif /* GRAPH_BASED_PROCESS_H_ */
