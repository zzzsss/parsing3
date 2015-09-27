/*
	This is a small toolkit just aimed to provide some nn tools.
	Similar to CSLM toolkit, but for simplicity do not have any conf-specified modules.
	Try to combine layers into one big machine and the structure is hard-coded
	<by zzs; start from 2015.9>
*/

#ifndef CSNN_H_
#define CSNN_H_

#include "nn_cache.h"
#include "nn_input.h"
#include "nn_math.h"
#include "nn_wb.h"
#include "nn_wv.h"
#include "options.h"
#include <vector>
#include <string>
#include <fstream>

//----------------------CSNN-----------------------
//the specified nn for the specified structure for the specified task
//just for convenience and flexibility
class Csnn{
protected:
	//order of parsing --- bad design, maybe
	virtual int get_order()=0;
	virtual void f_inputs()=0;
	//options
	nn_options *the_option;
	//the caches
	vector<nn_cache*> *c_allcaches;	//all of them
	nn_cache *c_out;	//1.output layer
	nn_cache *c_h;		//2.hidden layer
	nn_cache *c_repr;	//3.combined representation /** COMBINED **/
	nn_cache *c_wrepr;	//3-1.word/window based representations + average
	nn_cache *c_srepr;	//3-2.sentence based representations
	nn_cache *c_wv;		//3-1.word vectors

	//the parameters
	nn_wb *p_out;	//hidden -> out
	nn_wb *p_h;		//repr -> hidden
	vector<nn_wb*> *p_untied;	//untied for 3-1: has number of nPOS*nPOS+1 (the index 0 is the default one)
	nn_wv *p_word;
	nn_wv *p_pos;
	nn_wv *p_distance;

	void construct_caches();			//init and read
	void prepare_caches(int);				//before minibatch
	void construct_params();			//init

	//binary mode r/w
	void read_params(std::ifstream fin);	//read	--- !!AFTER the options are ready
	void write_params(std::ofstream fout);	//write

	//sth tmp for forward/backward/update
	nn_input* this_input;
	int this_bsize;			//one-time bsize
	int this_mbsize;		//minibatch's instance number
	vector<int> this_untied_index;

public:
	//from scratch
	void get_init(nn_options * o){
		the_option = o;
		construct_caches();
		construct_params();
	}
	//read from file
	void read_init(std::string fname){
		std::ifstream fin;
		fin.open(fname.c_str(),ifstream::binary);
		the_option = new nn_options(fin);
		read_params(fin);
		fin.close();
		construct_caches();
	}
	//write out
	void write(std::string fname){
		std::ofstream fout;
		fout.open(fname.c_str(),ifstream::binary);
		the_option->write(fout);
		write_params(fout);
		fout.close();
	}
	virtual ~Csnn(){}

	//main methods
	//-- SHOULD BE: while(MiniBatch){prepare_batch;while(sent){f;b;}update;}
	void prepare_batch(){
		this_input=0;
		this_bsize=0;
		this_mbsize=0;
	}
	//forward for one sentence, untied_rate==0 when no untied, ==1 when untied if possible
	REAL* forward(nn_input* in,int testing,REAL untied_rate);	//return new ones
	//backward and accumulate the gradients --- should be immediately after a forward
	void backward(REAL* gradients);
	//update parameters
	void update(int way,REAL lrate,REAL wdecay,REAL m_alpha,REAL rms_smooth);
};

#endif
