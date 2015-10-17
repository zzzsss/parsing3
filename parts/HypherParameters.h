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
#include <fstream>
#include <cstdlib>
using namespace std;

class HypherParameters{
public:
	static void Error(string x){
		cerr << x << endl;
		cout << x << endl;
		exit(1);
	}
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

int CONF_UPDATE_WAY;	//see nn_math.h: OPT_*
int CONF_NESTEROV_MOMENTUM;	//set to 1 if update with it firstly
double CONF_MOMENTUM_ALPHA;
double CONF_RMS_SMOOTH;

double CONF_NN_resample;				//re-sample rate
int CONF_NN_FIXVEC;						//no update on embeddings any longer
int CONF_minibatch;		//how many sentences for a batch(this is not bsize of nn)

//1.4-others
int CONF_dict_remove;	//remove words appears only less than this times
int CONF_random_seed;
int CONF_labeled;		//labeled or not

//1.5-scores
int CONF_score_prob;	//whether give transform score, only for M1 (0,1)
double CONF_score_o1filter_cut;
int CONF_score_combine_o1;
int CONF_score_combine_o2sib;
string CONF_score_mach_fo1;		//o1-filter mach
string CONF_score_mach_so1;		//o1-scorer mach
string CONF_score_mach_so2sib;	//o2sib-scorer mach

//init
HypherParameters(string conf):hp_nn()
{
	//defaults
	CONF_output_file = "output.txt";
	CONF_dict_file="vocab.dict";
	CONF_mach_name="nn.mach";
	CONF_mach_cur_suffix=".curr";
	CONF_mach_best_suffix=".best";
	CONF_NN_ITER=10;
	CONF_NN_ITER_decrease=1;
	CONF_NN_ITER_force_half=100;
	CONF_NN_LMULT=-0.5;
	CONF_NN_WD=1e-4;
	CONF_UPDATE_WAY=0;	//opt_sgd
	CONF_NESTEROV_MOMENTUM=0;
	CONF_MOMENTUM_ALPHA=0.8;	//??
	CONF_RMS_SMOOTH=0.8;		//??
	CONF_NN_resample=0.95;
	CONF_NN_FIXVEC=100;			//100 iters later??
	CONF_minibatch=10;
	CONF_dict_remove=3;
	CONF_random_seed=12345;
	CONF_labeled=0;
	CONF_score_prob=1;
	CONF_score_o1filter_cut=1e-6;
	CONF_score_combine_o1=1;
	CONF_score_combine_o2sib=1;

	//read in conf-file
#define DATA_LINE_LEN 10000
	ifstream fin(conf.c_str());
	cout << "Dealing configure file '" << conf << "'" << endl;
	// Method config
	string temp_for_m;
	fin >> temp_for_m;
	if(temp_for_m != "M")
		Error("First of conf-file must be M.");
	fin >> CONF_method;
	//
	while(!fin.eof()){
		string buf;
		char line[DATA_LINE_LEN];
		fin >> buf;
		if (buf=="") continue; // HACK
		if (buf[0]=='#') {fin.getline(line, DATA_LINE_LEN); continue;} // skip comments
		//1.1 and 1.2: file names
		if(buf=="train")		fin >> CONF_train_file;
		else if(buf=="dev")		fin >> CONF_dev_file;
		else if(buf=="test")	fin >> CONF_test_file;
		else if(buf=="output")	fin >> CONF_output_file;
		else if(buf=="gold")	fin >> CONF_gold_file;
		else if(buf=="dict")	fin >> CONF_dict_file;
		else if(buf=="mach-prefix") fin >> CONF_mach_name;
		//1.3
		else if(buf=="nn_lrate") fin >> CONF_NN_LRATE;
		else if(buf=="nn_iters") fin >> CONF_NN_ITER;
		else if(buf=="nn_iters_dec") fin >> CONF_NN_ITER_decrease;
		else if(buf=="nn_iters_force") fin >> CONF_NN_ITER_force_half;
		else if(buf=="nn_lmult") fin >> CONF_NN_LMULT;
		else if(buf=="nn_wd")	 fin >> CONF_NN_WD;

		else if(buf=="nn_way")		fin >> CONF_UPDATE_WAY;
		else if(buf=="nn_nesterov")	fin >> CONF_NESTEROV_MOMENTUM;
		else if(buf=="nn_momentum")	fin >> CONF_MOMENTUM_ALPHA;
		else if(buf=="nn_rms")		fin >> CONF_RMS_SMOOTH;

		else if(buf=="nn_resample") fin >> CONF_NN_resample;
		else if(buf=="nn_fixv")		fin >> CONF_NN_FIXVEC;
		else if(buf=="nn_mbatch") 	fin >> CONF_minibatch;
		//1.4-others
		else if(buf=="o_removes")	fin >> CONF_dict_remove;
		else if(buf=="o_srand")		fin >> CONF_random_seed;
		else if(buf=="o_labeled")	fin >> CONF_labeled;
		//1.5-scores
		else if(buf=="s_prob") 		fin >> CONF_score_prob;
		else if(buf=="s_fo1_cut") 	fin >> CONF_score_o1filter_cut;
		else if(buf=="s_so1_combine")	fin >> CONF_score_combine_o1;
		else if(buf=="s_so2sib_combine")	fin >> CONF_score_combine_o2sib;
		else if(buf=="s_mach_fo1")	fin >> CONF_score_mach_fo1;
		else if(buf=="s_mach_so1")	fin >> CONF_score_mach_so1;
		else if(buf=="s_mach_so2sib")	fin >> CONF_score_mach_so2sib;
		//TODO 1.x -special : for nn_option

	}
}

};



#endif /* PARTS_HYPHERPARAMETERS_H_ */
