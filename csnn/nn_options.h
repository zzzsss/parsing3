/*
 * options.h
 *
 *  Created on: Sep 25, 2015
 *      Author: zzs
 */

#ifndef CSNN_NN_OPTIONS_H_
#define CSNN_NN_OPTIONS_H_

#include <fstream>
#include "nn_math.h"
//options related to nn
class nn_options{
public:
	int NN_wnum;	//number of words
	int NN_pnum;	//number of pos
	int NN_dnum;	//number of distances
	int NN_out_prob;		//whether add softmax
	int NN_out_size;		//output size
	//------------------THOSE above need setting before training------------//

	int NN_wsize;	//word embed size
	int NN_psize;	//pos embed size
	int NN_dsize;	//distance embed size

	int NN_win;		//window size

	int NN_add_average;		//whether add average feature
	int NN_add_sent;		//whether add sentence features

	int NN_untied_dim;		//0,1,2: 0 means no untied, 1 means based on m, 2 means h-m
	REAL NN_untied_2brate;	//when dim==2, maybe need some back-off with random when training

	int NN_act;				//the activation function
	int NN_hidden_size;		//hidden size(near output)
	int NN_wrsize;			//word repr size
	int NN_srsize;			//sentence repr size

	REAL NN_dropout;		//dropout rate

	REAL NN_init_wbrange;
	REAL NN_init_wvrange;

	int get_NN_srsize(){
		if(NN_add_sent)
			return NN_srsize;
		else
			return 0;
	}
	int get_NN_rsize(){					//the size of representation layer = wr+sr
		return NN_wrsize+get_NN_srsize();
	}
	int get_NN_wv_wrsize(int order){	//the word vectors' size before rsize
		int basis = (NN_wsize+NN_psize)*NN_win*(order+1)+NN_dsize*(order);	//!DEBUG:order+1
		if(NN_add_average)
			basis += order*(NN_wsize+NN_psize);
		return basis;
	}

	//init and r/w
	nn_options(){default_init();}
	nn_options(std::ifstream &fin){read(fin);}

	void default_init(){
		//!! need setting !!
		NN_wnum = 50000;
		NN_pnum = 50;
		NN_dnum = 20;
		//NN_out_prob = ?;		//whether add softmax
		//NN_out_size = ?;		//output size

		//embedding size
		NN_wsize = 50;
		NN_psize = 30;
		NN_dsize = 20;
		//nn
		NN_win = 5;
		NN_add_average = 1;
		NN_add_sent = 0;
		NN_untied_dim = 0;
		NN_untied_2brate = 0.2;
		NN_act = 0;		//ACT_TANH
		NN_hidden_size = 100;
		NN_wrsize = 200;
		NN_srsize = 0;

		NN_dropout = 0;
		NN_init_wbrange = 0.1;
		NN_init_wvrange = 0.1;
	}
	//TODO
	void read(std::ifstream &fin){}
	void write(std::ofstream &fout){}
};

#endif /* CSNN_NN_OPTIONS_H_ */
