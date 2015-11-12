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
#define IMPOSSIBLE_INDEX -1000
#define GET_MAX_ONE(a,b) (((a)>(b))?(a):(b))
#define GET_MIN_ONE(a,b) (((a)>(b))?(b):(a))

class Process{
protected:
	template<class T>
	static void CHECK_EQUAL(T a,T b,const char* x){
		if(a != b)
			cerr << "!! warning of " << x << ":" << a << " != " << b << endl;
	}
	static void FatalError(const char* x){
		cerr << "Fatal error " << x << endl;
		exit(1);
	}
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
	virtual void each_create_machine()=0;
	virtual void each_train_one_iter()=0;
	virtual void each_test_one(DependencyInstance*)=0;		//set predict_head or predict_deprels here

	//1.1:helper-learning rate
	int set_lrate_one_iter();	//lrate schedule
	virtual int whether_keep_trainning();
	//1.2:helper-for training
	void nn_train_prepare();
	//1.3:init word-vector
	void init_embed();

	//2.test and dev
	double nn_dev_test(string to_test,string output,string gold,int testing=1);

	//3.1:forward scores and rearrange scores
	static REAL* forward_scores_o1(DependencyInstance* x,Csnn* m,nn_input** t,nn_input_helper* h,int testing,HypherParameters*hh=0);
	static REAL* forward_scores_o2sib(DependencyInstance* x,Csnn* m,nn_input** t,nn_input_helper* h,int testing,bool* cut_o1,HypherParameters*hh=0);
	static REAL* forward_scores_o3g(DependencyInstance* x,Csnn* m,nn_input** t,nn_input_helper* h,int testing,bool* cut_o1,HypherParameters*hh=0);
	static double* rearrange_scores_o1(DependencyInstance* x,Csnn* m,nn_input* t,REAL* fscores,bool prob_ouput,bool prob_trans);
	static double* rearrange_scores_o2sib(DependencyInstance* x,Csnn* m,nn_input* t,REAL* fscores,bool prob_ouput,bool prob_trans,double* rscores_o1);
	static double* rearrange_scores_o3g(DependencyInstance* x,Csnn* m,nn_input* t,REAL* fscores,bool prob_ouput,bool prob_trans,double* rscores_o1,double* rscores_o2sib);
	//3.1-c: get scores (combine forward and rearrange) --- only testing or training-parsing
	static bool* get_cut_o1(DependencyInstance* x,CsnnO1* m,Dictionary *dict,double cut);
	static double* get_scores_o1(DependencyInstance* x,Csnn* m,Dictionary* dict,bool trans);
	static double* get_scores_o2sib(DependencyInstance* x,Csnn* m,Dictionary* dict,bool trans,bool* cut_o1,double* rscores_o1);
	static double* get_scores_o3g(DependencyInstance* x,Csnn* m,Dictionary* dict,bool trans,bool* cut_o1,double* rscores_o1,double* rscores_o2sib);
	//3.2:parse (take care of the labeled situation)
	void parse_o1(DependencyInstance* x);
	void parse_o2sib(DependencyInstance* x,CsnnO1* o1_filter,CsnnO1* o1_scorer);
	void parse_o3g(DependencyInstance* x,CsnnO1* o1_filter,CsnnO1* o1_scorer,CsnnO2* o2_scorer);

public:
	Process(string);
	virtual ~Process(){}
	virtual void train();
	virtual void test(string);
	void check_o1_filter(string mach,string cut);
};


#endif /* GRAPH_BASED_PROCESS_H_ */
