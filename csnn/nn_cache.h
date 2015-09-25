/*
 * nn_cache.h
 *
 *  Created on: Sep 17, 2015
 *      Author: zzs
 */

#ifndef CSNN_NN_CACHE_H_
#define CSNN_NN_CACHE_H_

/*
 * the so-called layers, basically just two arrays of bsize*length
 * -- values and gradients
 */

#include "nn_math.h"
#include <cstdlib>

class nn_cache{
private:
	//values and gradients AT LEAST have size bsize*length
	long bsize;		//mini-batch size
	long length;	//length of one instance
	REAL* values;
	REAL* gradients;
	REAL* dropout;
public:
	nn_cache(long b,long l):bsize(b),length(l),values(0),gradients(0),dropout(0){
		long all = bsize*length;
		if(all > 0){
			values = new REAL[all];
			gradients = new REAL[all];
			dropout = new REAL[all];
		}
	}
	~nn_cache(){
		delete []values;
		delete []gradients;
		delete []dropout;
	}
	REAL* get_values(){return values;}
	REAL* get_gradients(){return gradients;}
	bool* get_dropout(){return dropout;}
	void gen_dropout(REAL rate){
		for(long i=0;i<bsize*length;i++)
			dropout[i] = (drand48()<rate) ? 1 : 0;
	}
	void clear_values(){
		for(long i=0;i<bsize*length;i++)
			values[i] = 0;
	}
	void clear_gradients(){
		for(long i=0;i<bsize*length;i++)
			gradients[i] = 0;
	}
	void clear_all(){
		clear_values();
		clear_gradients();
	}
	//special value of -1 means no change
	//!! delete old data when expand
	void resize(long b,long l=-1){
		long old_all = bsize*length;
		if(b >= 0)
			bsize = b;
		if(l >= 0)
			length = l;
		long all = bsize*length;
		if(all > old_all){
			delete []values;
			delete []gradients;
			delete []dropout;
			values = new REAL[all];
			gradients = new REAL[all];
			dropout = new REAL[all];
		}
	}
};

#endif /* CSNN_NN_CACHE_H_ */
