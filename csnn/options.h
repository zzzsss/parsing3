/*
 * options.h
 *
 *  Created on: Sep 25, 2015
 *      Author: zzs
 */

#ifndef CSNN_OPTIONS_H_
#define CSNN_OPTIONS_H_

#include <fstream>
//options related to nn
class nn_options{
public:
	int NN_wnum;	//number of words
	int NN_pnum;	//number of pos
	int NN_dnum;	//number of distances

	int NN_wsize;	//word embed size
	int NN_psize;	//pos embed size
	int NN_dsize;	//distance embed size

	int NN_win;		//window size

	int NN_add_average;		//whether add average feature
	int NN_add_sent;		//whether add sentence features

	int NN_out_size;		//output size
	int NN_hidden_size;		//hidden size(near output)
	int NN_wrsize;			//word repr size
	int NN_srsize;			//sentence repr size

	int NN_act;				//the activation function
	int NN_out_prob;		//whether add softmax

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
		return NN_wrsize+NN_srsize;
	}
	int get_NN_wv_wrsize(int order){	//the word vectors' size before rsize
		int basis = (NN_wsize+NN_psize)*NN_win+NN_dsize*(order-1);
		if(NN_add_average)
			basis += order*(NN_wsize+NN_psize);
		return basis;
	}

	//init and r/w
	nn_options(){default_init();}
	nn_options(std::ifstream fin){read(fin);}

	void default_init(){}
	void read(std::ifstream fin){}
	void write(std::ofstream fout){}
};

#endif /* CSNN_OPTIONS_H_ */
