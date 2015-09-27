/*
 * nn_param.h
 *
 *  Created on: Sep 19, 2015
 *      Author: zzs
 */

#ifndef CSNN_NN_WB_H_
#define CSNN_NN_WB_H_

#include "nn_math.h"
#include <cstdlib>
#include <cstring>
#include <fstream>

//the linear parameter: w and b (weight and bias)
class nn_wb{
private:
	bool updating;
	int idim;
	int odim;

	REAL* w;	//o*i
	REAL* b;	//o
	//gradient calculated
	REAL* w_grad;
	REAL* b_grad;
	//gradient momentum
	REAL* w_moment;
	REAL* b_moment;
	//gradient accumulated square --- for AdaGrad
	REAL* w_square;
	REAL* b_square;

public:
	nn_wb(int i,int o):updating(true),idim(i),odim(o){
		int all = i*o;	//int is enough
		w = new REAL[all];
		b = new REAL[o];
		w_grad = new REAL[all];
		b_grad = new REAL[o];
		w_moment = new REAL[all];
		b_moment = new REAL[o];
		w_square = new REAL[all];
		b_square = new REAL[o];
	}
	void get_init(const REAL range){
		//fanio for weight and random for bias
		REAL c=2.0*range/sqrt((REAL) (idim+odim));
		for (int i=0; i<idim*odim; i++)
			w[i]=c*(drand48()-0.5);
		c=range*2.0;
		for (int i=0; i<odim; i++)
			b[i]=c*(drand48()-0.5);
		memset(w_grad,0,sizeof(REAL)*idim*odim);
		memset(w_moment,0,sizeof(REAL)*idim*odim);
		memset(w_square,0,sizeof(REAL)*idim*odim);
		memset(b_grad,0,sizeof(REAL)*odim);
		memset(b_moment,0,sizeof(REAL)*odim);
		memset(b_square,0,sizeof(REAL)*odim);
	}
	void clear_grad(){
		memset(w_grad,0,sizeof(REAL)*idim*odim);
		memset(b_grad,0,sizeof(REAL)*odim);
	}
	~nn_wb(){
		delete []w;
		delete []b;
		delete []w_grad;
		delete []b_grad;
		delete []w_moment;
		delete []b_moment;
		delete []w_square;
		delete []b_square;
	}
	bool need_updating(){return updating;}
	void set_updating(bool x){updating = x;}
	int geti(){return idim;}
	int geto(){return odim;}

	//three important operations
	//- the bsize of f/b only means the instances for one pass (one sentences)
	void forward(/*const*/REAL* in,REAL* out,int bsize);							//forward setting
	void backward(/*const*/REAL* ograd,REAL* igrad,/*const*/REAL* in,int bsize);	//backward accumulate
	void update(int way,REAL lrate,REAL wdecay,REAL m_alpha,REAL rms_smooth,int mbsize);

	//binary r/w
	nn_wb(std::ifstream fin):updating(true){
		fin.read((char*)&idim,sizeof(int));
		fin.read((char*)&odim,sizeof(int));
		int all = idim*odim;	//int is enough
		w = new REAL[all];
		b = new REAL[odim];
		w_grad = new REAL[all];
		b_grad = new REAL[odim];
		w_moment = new REAL[all];
		b_moment = new REAL[odim];
		w_square = new REAL[all];
		b_square = new REAL[odim];
		fin.read((char*)&w,sizeof(REAL)*all);
		fin.read((char*)&b,sizeof(REAL)*odim);
	}
	void write_params(std::ofstream fout){
		fout.write((char*)&idim,sizeof(int));
		fout.write((char*)&odim,sizeof(int));
		fout.write((char*)&w,sizeof(REAL)*idim*odim);
		fout.write((char*)&b,sizeof(REAL)*odim);
	}
};



#endif /* CSNN_NN_WB_H_ */
