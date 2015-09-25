/*
 * nn_input.h
 *
 *  Created on: Sep 17, 2015
 *      Author: zzs
 */

#ifndef CSNN_NN_INPUT_H_
#define CSNN_NN_INPUT_H_

/*
 * 	specified input for the nn
 * 	three vectors: word-form of sentence, pos of sentence, streams of inputs
 */

#include <vector>
using namespace std;

class nn_input{
public:
	int num_inst;		//total number of instance
	int num_width;		//width of one instance
	//these three vectors all allocated outside
	//!! must -- size(inputs) == num_inst*num_width
	vector<int>* inputs;
	vector<int>* wordl;
	vector<int>* posl;

	nn_input(int i,int w,vector<int>* il,vector<int>* wl,vector<int>* pl):
		num_inst(i),num_width(w),inputs(il),wordl(wl),posl(pl){}
	int get_numi(){return num_inst;}
	int get_numw(){return num_width;}
};



#endif /* CSNN_NN_INPUT_H_ */
