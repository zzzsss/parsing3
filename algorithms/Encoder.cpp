/*
 * Encoder.cpp
 *
 *  Created on: 2015Äê6ÔÂ8ÈÕ
 *      Author: zzs
 */

//inside-outside algorithm for order-1 model
// --- almost from MaxParser->DependendcyEncoder.cpp

#include "Eisner.h"
#include "Helper.h"

static double* calc_inside(const int length,const double *scores)
{
	double* inside = new double[length*length*2*2];
	for(int i=0;i<length*length*2*2;i++)
		inside[i] = 0.0;
	//loop
	for(int j = 1; j < length; j++){
		for(int s = 0; s + j < length; s++){
			int t = s + j;
			double prodProb_st = scores[get_index2(length,s,t)];
			double prodProb_ts = scores[get_index2(length,t,s)];
			//incomplete spans
			int key_st_0 = get_index(length,s,t,E_RIGHT,E_INCOM);
			int key_ts_0 = get_index(length,s,t,E_LEFT,E_INCOM);
			//complete spans
			int key_st_1 = get_index(length,s,t,E_RIGHT,E_COM);
			int key_ts_1 = get_index(length,s,t,E_LEFT,E_COM);
			//init-flags
			bool flg_st_0 = true, flg_ts_0 = true;
			bool flg_st_1 = true, flg_ts_1 = true;
			//incomplete spans
			for(int r = s; r < t; r++){
				int key1 = get_index(length,s,r,E_RIGHT,E_COM);
				int key2 = get_index(length,r+1,t,E_LEFT,E_COM);
				inside[key_st_0] = logsumexp(inside[key_st_0], inside[key1] + inside[key2] + prodProb_st, flg_st_0);
				flg_st_0 = false;
				inside[key_ts_0] = logsumexp(inside[key_ts_0], inside[key1] + inside[key2] + prodProb_ts, flg_ts_0);
				flg_ts_0 = false;
			}
			//complete spans
			for(int r = s; r <= t; r++){
				if(r != s){
					int key1 = get_index(length,s,r,E_RIGHT,E_INCOM);
					int key2 = get_index(length,r,t,E_RIGHT,E_COM);
					inside[key_st_1] = logsumexp(inside[key_st_1], inside[key1] + inside[key2], flg_st_1);
					flg_st_1 = false;
				}
				if(r != t){
					int key1 = get_index(length,s,r,E_LEFT,E_COM);
					int key2 = get_index(length,r,t,E_LEFT,E_INCOM);
					inside[key_ts_1] = logsumexp(inside[key_ts_1], inside[key1] + inside[key2], flg_ts_1);
					flg_ts_1 = false;
				}
			}
		}
	}
	return inside;
}

static double* calc_outside(const int length,const double *inside,const double *scores)
{
	double* outside = new double[length*length*2*2];
	for(int i=0;i<length*length*2*2;i++)
		outside[i] = 0.0;
	for(int j = length-1; j >= 1; j--){
		for(int s = 0; s + j < length; s++){
			int t = s + j;
			//incomplete spans
			int key_st_0 = get_index(length,s,t,E_RIGHT,E_INCOM);
			int key_ts_0 = get_index(length,s,t,E_LEFT,E_INCOM);
			//complete spans
			int key_st_1 = get_index(length,s,t,E_RIGHT,E_COM);
			int key_ts_1 = get_index(length,s,t,E_LEFT,E_COM);
			//init-flags
			bool flg_st_0 = true, flg_ts_0 = true;
			bool flg_st_1 = true, flg_ts_1 = true;
			//!!first complete spans
			//complete spans
			for(int r = 0; r < s; r++){
				double prodProb_rt = scores[get_index2(length,r,t)];
				double prodProb_tr = scores[get_index2(length,t,r)];
				//(+incomplete on the left) right one
				int key_b = get_index(length,r,s,E_RIGHT,E_INCOM);
				int key_a = get_index(length,r,t,E_RIGHT,E_COM);
				outside[key_st_1] = logsumexp(outside[key_st_1], inside[key_b] + outside[key_a], flg_st_1);
				flg_st_1 = false;
				//(+complete on the left) left one
				key_b = get_index(length,r,s-1,E_RIGHT,E_COM);
				key_a = get_index(length,r,t,E_RIGHT,E_INCOM);
				outside[key_ts_1] = logsumexp(outside[key_ts_1], inside[key_b] + outside[key_a] + prodProb_rt, flg_ts_1);
				flg_ts_1 = false;
				key_a = get_index(length,r,t,E_LEFT,E_INCOM);
				outside[key_ts_1] = logsumexp(outside[key_ts_1], inside[key_b] + outside[key_a] + prodProb_tr, flg_ts_1);
				flg_ts_1 = false;
			}
			for(int r = t + 1; r < length; r++){
				double prodProb_sr = scores[get_index2(length,s,r)];
				double prodProb_rs = scores[get_index2(length,r,s)];
				//(+complete on the right) right one
				int key_b = get_index(length,t+1,r,E_LEFT,E_COM);
				int key_a = get_index(length,s,r,E_RIGHT,E_INCOM);
				outside[key_st_1] = logsumexp(outside[key_st_1], inside[key_b] + outside[key_a] + prodProb_sr, flg_st_1);
				flg_st_1 = false;
				key_a = get_index(length,s,r,E_LEFT,E_INCOM);
				outside[key_st_1] = logsumexp(outside[key_st_1], inside[key_b] + outside[key_a] + prodProb_rs, flg_st_1);
				flg_st_1 = false;
				//(+incomplete on the right) left one
				key_b = get_index(length,t,r,E_LEFT,E_INCOM);
				key_a = get_index(length,s,r,E_LEFT,E_COM);
				outside[key_ts_1] = logsumexp(outside[key_ts_1], inside[key_b] + outside[key_a], flg_ts_1);
				flg_ts_1 = false;
			}
			//incomplete spans
			for(int r = t; r < length; r++){
				int key_b = get_index(length,t,r,E_RIGHT,E_COM);
				int key_a = get_index(length,s,r,E_RIGHT,E_COM);
				outside[key_st_0] = logsumexp(outside[key_st_0], inside[key_b] + outside[key_a], flg_st_0);
				flg_st_0 = false;
			}
			for(int r = 0; r <= s; r++){
				int key_b = get_index(length,r,s,E_LEFT,E_COM);
				int key_a = get_index(length,r,t,E_LEFT,E_COM);
				outside[key_ts_0] = logsumexp(outside[key_ts_0], inside[key_b] + outside[key_a], flg_ts_0);
				flg_ts_0 = false;
			}
		}
	}
	return outside;
}

double* encodeMarginals(int length,double* scores)
{
	double* marginals = new double[length*length];	//use get_index2
	double* inside = calc_inside(length,scores);
	double* outside = calc_outside(length,scores);
	int key1 = get_index(length,0,length-1,E_RIGHT,E_COM);
	int key2 = get_index(length,0,length-1,E_LEFT,E_COM);
	double z = logsumexp(inside[key1], inside[key2], false);
	for(int i=0;i<length;i++){
		for(int j=i+1;j<length;j++){
			//i->j
			int key_io = get_index(length,i,j,E_RIGHT,E_INCOM);
			int key_assign = get_index2(length,i,j);
			marginals[key_assign] = exp(inside[key_io]+outside[key_io]-z);
			//j->i
			key_io = get_index(length,i,j,E_LEFT,E_INCOM);
			key_assign = get_index2(length,j,i);
			marginals[key_assign] = exp(inside[key_io]+outside[key_io]-z);
		}
	}
	delete []inside;
	delete []outside;
	return marginals;
}
