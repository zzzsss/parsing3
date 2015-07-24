/*
 * common.h
 *
 *  Created on: Dec 19, 2014
 *      Author: zzs
 */

#ifndef COMMON_H_
#define COMMON_H_

#include "../cslm/Tools.h"
#include <fstream>
#include <cstdio>
#include <sstream>

#define DOUBLE_LARGENEG -10000000.0		//maybe it is enough
#define INIT_EM_MAX_SIZE 2000000	//maybe enough

static void get_oneline_list(istream& fin,double** l,int n){
	string line;
	int assigned = 0;
	int multi = 1;
	std::getline(fin,line);
	*l = new double[n];
	stringstream ss(line);
	while(1){
		double k = 0;
		ss >> k;
		if(!ss || assigned>=n) break;
		if((int)k > 0)	//special
			multi = k;
		else{
			for(int i=0;i<multi && assigned<n;i++)
				(*l)[assigned++] = k;
			multi = 1;
		}
	}
	for(int i=assigned;i<n;i++)
		(*l)[i] = (*l)[assigned-1];	//will fault if -1
}

class  parsing_conf{
public:
//1.0
int CONF_method;	//which method
//1.1-files
string CONF_train_file;	//the training file
string CONF_dev_file;	//testing files
string CONF_test_file;	//testing files
string CONF_output_file;
string CONF_gold_file;	//golden files
//1.2-other files
string CONF_dict_file;		//for dictionary
string CONF_mach_name;		//mach name
string CONF_mach_conf_name;	//nn.conf
string CONF_mach_cur_suffix;
string CONF_mach_best_suffix;
string CONF_restart_file;		//recording the training iters
string CONF_feature_file;	//for featgen's filters
//1.3-for nn
string CONF_NN_toolkit;
//1.3.001 -- nn structure split options
int CONF_NN_split;				//whether split the first layer(context split)
int CONF_NN_split_share;		//when splitting, whether share parameters(w and b)
//other nn options
double CONF_NN_LRATE;
int CONF_NN_ITER;
int CONF_NN_ITER_decrease;		//at lease cut lrate this times when stopping(so real iters maybe more than iter)
int CONF_NN_ITER_force_half;	//force cut half if no cutting for how many iters
double CONF_NN_LMULT;	//when >=0:as mult ; -1~0: schedule rate
double CONF_NN_WD;
int CONF_NN_we;						//word-embedding size
int CONF_NN_plus_layers;				//plus number of layers(plus from base)		---changed: number of hidden except projection---
double CONF_NN_resample;				//re-sample rate
int CONF_NN_BS;						//block-size
int* CONF_NN_h_size;	//hsizes: array
double CONF_NN_drop;
int CONF_NN_act;	//activation functions

//use o1-mach for higher-order <must with the same other parameters>
int CONF_NN_highO_embed_init;
int CONF_NN_highO_score_combine;	//combine score with o1 mach
string CONF_NN_highO_o1mach;
//o1-mach as filter --- only 0-1 models(thus M7)
int CONF_NN_highO_o1filter;
double CONF_NN_highO_o1filter_cut;	//filter out when < this or special when this value itself <0
//**high-order machines(used in o3g)
string CONF_NN_highO_o2sibmach;
string CONF_NN_highO_o2gmach;
int CONF_NN_highO_score_combine_o2sib;
int CONF_NN_highO_score_combine_o2g;
int CONF_NN_highO_score_combine_o3g_self;	//maybe no gsib score in o3g(in fact not order3)
//1.3.5 -- init embedings
string CONF_NN_WL;
string CONF_NN_EM;
double CONF_NN_ISCALE;
//pre-calculation options
int CONF_NN_PRECALC;				//whether perform pre-calculation for the best machine

//1.4-for parsing basis
int CONF_x_window;	//word and pos window size
int CONF_add_distance;	//whether add distance --- and for different ways
int CONF_add_direction;	//whether add dummy direction for o2g and o3g
int CONF_dict_remove;	//remove words appears only less than this times
int CONF_add_pos;		//whether add pos
int CONF_oov_backoff;	//whether backoff to pos with oov
int CONF_dict_tolower;	//tolower all the words
//1.5-others
int CONF_random_seed;
//1.6 -- scores
int CONF_score_prob;	//whether give transform score, only for M1 (0,1)

//1.7 -- for process_prob (MethodProb)
double CONF_MP_gradient_small;	//whether cut to zero when training, 0 means off
int CONF_MP_training_rearrange;	//whether re-arrange training samples to exclude 0 gradient, only makes sense when gradient_small is on
//int CONF_MP_o1training_0m;	//for mp1, allow 0 as m when training	---<DISCARDED>
int CONF_MP_marginal_score;		//use marginal prob as scores
double CONF_MP_scale_reg;		//l2-reg directly for scores, just trying
string CONF_MP_o1mach;		//for score combining
string CONF_MP_o2sibmach;	//for o2sib score combining, here not use options
int CONF_MP_o3g_toolong;

//1.7.5 -- changes
double *CONF_NN_resample_list;
double *CONF_NN_highO_o1filter_cut_list;

//init
parsing_conf(string conf_file)
{
	CONF_NN_toolkit = string("HPerf");
	CONF_NN_act = 0;	//default activation depends on specified ones
	//defaults:
	CONF_output_file="output.txt";
	CONF_dict_file="vocab.dict";		//for dictionary
	CONF_mach_name="nn.mach";		//mach name
	CONF_mach_conf_name="nn.conf";	//nn.conf
	CONF_mach_cur_suffix=".curr";
	CONF_mach_best_suffix=".best";
	CONF_restart_file="nn.restart";		//recording the training iters
	CONF_feature_file="nn.feat";
	CONF_NN_split = 0;				//whether split the first layer(context split)
	CONF_NN_split_share = 0;		//when splitting, whether share parameters(w and b)
	CONF_NN_LRATE=0.1;
	CONF_NN_ITER=10;
	CONF_NN_ITER_decrease=1;		//cut times
	CONF_NN_ITER_force_half=100;
	CONF_NN_LMULT=-0.5;	//when >=0:as mult ; -1~0: schedule rate
	//CONF_NN_WD=3e-5;
	CONF_NN_WD=-1e-4;
	CONF_NN_we=50;						//word-embedding size
	CONF_NN_plus_layers=2;				//plus number of layers(plus from base)
	CONF_NN_resample=1.0;				//re-sample rate
	CONF_NN_BS=128;						//block-size
	CONF_NN_h_size = 0;
	CONF_NN_drop = -1;
	CONF_NN_highO_embed_init = 0;
	CONF_NN_highO_score_combine = 1;
	CONF_NN_highO_o1filter = 1;
	//CONF_NN_highO_o1filter_cut = 0.001;	//filter out when < this
	CONF_NN_highO_o1filter_cut = 1e-6;
	CONF_NN_ISCALE=0.1;
	CONF_x_window=7;	//word and pos window size
	CONF_add_distance=1;	//whether add distance --- and for different ways
	CONF_add_direction=0;
	CONF_dict_remove=3;	//remove words appears only less than this times
	CONF_add_pos=1;		//whether add pos
	CONF_oov_backoff=1;	//whether backoff to pos with oov
	CONF_dict_tolower=0;
	CONF_random_seed=12345;
	CONF_score_prob=1;
	//for o3g
	CONF_NN_highO_score_combine_o2sib = 1;
	CONF_NN_highO_score_combine_o2g = 1;
	CONF_NN_highO_score_combine_o3g_self = 1;
	//pre-calc
	CONF_NN_PRECALC = 0;
	//MP
	//CONF_MP_gradient_small = 1e-8;
	CONF_MP_gradient_small = 0;
	CONF_MP_training_rearrange = 1;
	CONF_MP_marginal_score = 0;
	CONF_MP_scale_reg = 0;
	CONF_MP_o3g_toolong = 1000;
	//changes
	CONF_NN_resample_list = 0;
	CONF_NN_highO_o1filter_cut_list = 0;

	//read in conf-file
#define DATA_LINE_LEN 10000
	ifstream fin(conf_file.c_str());
	cout << "Dealing configure file '" << conf_file << "'" << endl;
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
		//1.1
		if(buf=="train")		fin >> CONF_train_file;
		else if(buf=="dev")		fin >> CONF_dev_file;
		else if(buf=="test")	fin >> CONF_test_file;
		else if(buf=="output")	fin >> CONF_output_file;
		else if(buf=="gold")	fin >> CONF_gold_file;
		else if(buf=="dict")	fin >> CONF_dict_file;
		else if(buf=="mach-prefix") fin >> CONF_mach_name;
		//1.2
		else if(buf=="nn_tool") fin >> CONF_NN_toolkit;
		//1.3
		else if(buf=="nn_split") fin >> CONF_NN_split;
		else if(buf=="nn_split_s") fin >> CONF_NN_split_share;
		//
		else if(buf=="nn_lrate") fin >> CONF_NN_LRATE;
		else if(buf=="nn_iters") fin >> CONF_NN_ITER;
		else if(buf=="nn_iters_dec") fin >> CONF_NN_ITER_decrease;
		else if(buf=="nn_iters_force") fin >> CONF_NN_ITER_force_half;
		else if(buf=="nn_lmult") fin >> CONF_NN_LMULT;
		else if(buf=="nn_wd")	 fin >> CONF_NN_WD;
		else if(buf=="nn_we")	fin >> CONF_NN_we;
		else if(buf=="nn_plusl") fin >> CONF_NN_plus_layers;
		else if(buf=="nn_resample") fin >> CONF_NN_resample;
		else if(buf=="nn_bs") 	 fin >> CONF_NN_BS;
		else if(buf=="nn_hsize"){
			//here no checking
			CONF_NN_h_size = new int[CONF_NN_plus_layers];
			for(int i=0;i<CONF_NN_plus_layers;i++)
				fin >> CONF_NN_h_size[i];
		}
		//these two specified for 2 hidden layers
		else if(buf=="nn_hsize1"){
			//here no checking
			CONF_NN_h_size = new int[2];
			fin >> CONF_NN_h_size[0];
		}
		else if(buf=="nn_hsize2"){
			//must after nn_hsize1
			fin >> CONF_NN_h_size[1];
		}
		else if(buf=="nn_drop")	fin >> CONF_NN_drop;
		else if(buf=="nn_act") fin >> CONF_NN_act;
		//o2sib use o1
		else if(buf=="nn_o1mach") fin >> CONF_NN_highO_o1mach;
		else if(buf=="nn_o1mach_init") fin >> 	CONF_NN_highO_embed_init;
		else if(buf=="nn_o1mach_combine") fin >> CONF_NN_highO_score_combine;
		else if(buf=="nn_o1mach_filter") fin >> CONF_NN_highO_o1filter;
		else if(buf=="nn_o1mach_filter_cut") fin >> CONF_NN_highO_o1filter_cut;
		//o3g use o2...
		else if(buf=="nn_o2sibmach") fin >> CONF_NN_highO_o2sibmach;
		else if(buf=="nn_o2gmach") fin >> CONF_NN_highO_o2gmach;
		else if(buf=="nn_o2sib_combine") fin >> CONF_NN_highO_score_combine_o2sib;
		else if(buf=="nn_o2g_combine") fin >> CONF_NN_highO_score_combine_o2g;
		else if(buf=="nn_o3g_combine") fin >> CONF_NN_highO_score_combine_o3g_self;
		//1.3.5
		else if(buf=="nn_init_wl") fin >> CONF_NN_WL;
		else if(buf=="nn_init_em") fin >> CONF_NN_EM;
		else if(buf=="nn_init_scale") fin >> CONF_NN_ISCALE;
		//1.4
		else if(buf=="f_xwin") 	fin >> CONF_x_window;
		else if(buf=="f_distance") fin >> CONF_add_distance;
		else if(buf=="f_direction")	fin >> CONF_add_direction;
		else if(buf=="f_removes") fin >> CONF_dict_remove;
		else if(buf=="f_pos")	fin >> CONF_add_pos;
		else if(buf=="f_oov_bo")	fin >> CONF_oov_backoff;
		else if(buf=="f_tolower") fin >> CONF_dict_tolower;
		//1.5
		else if(buf=="o_srand") fin >> CONF_random_seed;
		//1.6
		else if(buf=="s_prob") fin >> CONF_score_prob;
		//precalc
		else if(buf=="nn_precalc") fin >> CONF_NN_PRECALC;
		//1.7
		else if(buf=="mp_gsmall") fin >> CONF_MP_gradient_small;
		else if(buf=="mp_gsmall_re") fin >> CONF_MP_training_rearrange;
		else if(buf=="mp_ms")	fin >> CONF_MP_marginal_score;
		else if(buf=="mp_reg")	fin >> CONF_MP_scale_reg;
		else if(buf=="mp_o1mach") fin >> CONF_MP_o1mach;
		else if(buf=="mp_o2sibmach") fin >> CONF_MP_o2sibmach;
		else if(buf=="mp_o3g_long") fin >> CONF_MP_o3g_toolong;
		//1.7.5
		else if(buf=="nn_resample_l"){
			get_oneline_list(fin,&CONF_NN_resample_list,CONF_NN_ITER);
			CONF_NN_resample = CONF_NN_resample_list[0];
		}
		else if(buf=="nn_filter_l"){
			get_oneline_list(fin,&CONF_NN_highO_o1filter_cut_list,CONF_NN_ITER);
			CONF_NN_highO_o1filter_cut = CONF_NN_highO_o1filter_cut_list[0];
		}
		else
			cout << "Unknown conf " << buf << endl;
	}

	//the configurations
	printf("The configurations:\n");
	printf("Data files: %s,%s,%s,%s,%s\n",CONF_train_file.c_str(),CONF_dev_file.c_str(),
			CONF_test_file.c_str(),CONF_output_file.c_str(),CONF_gold_file.c_str());
	printf("NN: lrate(%g),iters(%d),lmult(%g),wdecay(%g),"
			"word_esize(%d),plus_layers(%d),resample(%g),bsize(%d),drop_out(%g)",
			CONF_NN_LRATE,CONF_NN_ITER,CONF_NN_LMULT,CONF_NN_WD,CONF_NN_we,CONF_NN_plus_layers,
			CONF_NN_resample,CONF_NN_BS,CONF_NN_drop);
	printf("Feature: xwindow(%d),distance(%d),removes(%d)\n",
			CONF_x_window,CONF_add_distance,CONF_dict_remove);
}

};


#endif /* COMMON_H_ */
