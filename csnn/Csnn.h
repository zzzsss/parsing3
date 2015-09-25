/*
	This is a small toolkit just aimed to probive some nn tools.
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
	vector<nn_wb*> *p_untied;	//untied for 3-1: has number of nPOS*nPOS
	nn_wv *p_word;
	nn_wv *p_pos;
	nn_wv *p_distance;

	void construct_caches();			//init and read
	void prepare_caches();				//before minibatch
	void construct_params();			//init
	void read_params(std::ifstream fin);	//read
	void write_params(std::ofstream fout);	//write

public:
	void get_init(nn_options * o){
		the_option = o;
	}
	virtual ~Csnn(){}
};

#endif
