/*
 * Encoder.cpp
 *
 *  Created on: 2015Äê6ÔÂ8ÈÕ
 *      Author: zzs
 */

//inside-outside algorithm for order-1 model
// --- according to MaxParser->DependendcyEncoder.cpp
// 	--- but does not count-in root as m

#include "Eisner.h"
#include "Helper.h"

static double* calc_inside(const int length,const double *probs)
{
	int all = length * length * 2 * 2;
	double* inside = new double[all];
	for(int i=0;i<all;i++)
		inside[i] = 0;
	//loop
	for(int k=1; k<length; k++){//the distance k
		for(int s=0; s<length; s++){
			//s--t
			int t=s+k;
			if(t>=length)
				break;
			int the_ind = -1;
			//1.incomplete
			//1.1 s->t
			the_ind = get_index(length,s,t,E_RIGHT,E_INCOM);
			inside[the_ind] = Negative_Infinity;
			for(int r = s; r < t; r++){
				inside[the_ind] = logsumexp(inside[the_ind],inside[get_index(length,s,r,E_RIGHT,E_COM)]
					+inside[get_index(length,r+1,t,E_LEFT,E_COM)]+probs[get_index2(length,s,t)]);
			}
			//1.2 t->s
			if(s!=0){
			the_ind = get_index(length,s,t,E_LEFT,E_INCOM);
			inside[the_ind] = Negative_Infinity;
			for(int r = s; r < t; r++){
				inside[the_ind] = logsumexp(inside[the_ind],inside[get_index(length,s,r,E_RIGHT,E_COM)]
					+inside[get_index(length,r+1,t,E_LEFT,E_COM)]+probs[get_index2(length,t,s)]);
			}
			}
			//2.complete
			//2.1 s->t
			the_ind = get_index(length,s,t,E_RIGHT,E_COM);
			inside[the_ind] = Negative_Infinity;
			for(int r = s+1; r <= t; r++){
				inside[the_ind] = logsumexp(inside[the_ind],inside[get_index(length,s,r,E_RIGHT,E_INCOM)]
					+inside[get_index(length,r,t,E_RIGHT,E_COM)]);
			}
			//2.2 t->s
			if(s!=0){
			the_ind = get_index(length,s,t,E_LEFT,E_COM);
			inside[the_ind] = Negative_Infinity;
			for(int r = s; r < t; r++){
				inside[the_ind] = logsumexp(inside[the_ind],inside[get_index(length,s,r,E_LEFT,E_COM)]
					+inside[get_index(length,r,t,E_LEFT,E_INCOM)]);
			}
			}
		}
	}
	return inside;
}

static double* calc_outside(const int length,const double *inside,const double *probs)
{
	int all = length * length * 2 * 2;
	double* outside = new double[all];
	for(int i=0;i<all;i++)
		outside[i] = 0;
	//loop
	for(int k=length-2;k>=1;k--){	//no need for the largest span
		for(int s=0; s<length; s++){
			//s--t
			int t=s+k;
			if(t>=length)
				break;
			int the_ind = -1;
			//1.complete
			//1.1 s->t
			the_ind = get_index(length,s,t,E_RIGHT,E_COM);
			outside[the_ind] = Negative_Infinity;
			for(int r=t+1;r<length;r++){
				outside[the_ind] = logsumexp(outside[the_ind],inside[get_index(length,t+1,r,E_LEFT,E_COM)]
					+outside[get_index(length,s,r,E_RIGHT,E_INCOM)]+probs[get_index2(length,s,r)]);
				if(s!=0){
					outside[the_ind] = logsumexp(outside[the_ind],inside[get_index(length,t+1,r,E_LEFT,E_COM)]
						+outside[get_index(length,s,r,E_LEFT,E_INCOM)]+probs[get_index2(length,r,s)]);
				}
			}
			for(int r=0;r<s;r++){
				outside[the_ind] = logsumexp(outside[the_ind],inside[get_index(length,r,s,E_RIGHT,E_INCOM)]
					+outside[get_index(length,r,t,E_RIGHT,E_COM)]);
			}
			//1.2 t->s
			if(s!=0){
			the_ind = get_index(length,s,t,E_LEFT,E_COM);
			outside[the_ind] = Negative_Infinity;
			for(int r=0;r<s;r++){
				outside[the_ind] = logsumexp(outside[the_ind],inside[get_index(length,r,s-1,E_RIGHT,E_COM)]
					+outside[get_index(length,r,t,E_RIGHT,E_INCOM)]+probs[get_index2(length,r,t)]);
				if(r!=0){
					outside[the_ind] = logsumexp(outside[the_ind],inside[get_index(length,r,s-1,E_RIGHT,E_COM)]
						+outside[get_index(length,r,t,E_LEFT,E_INCOM)]+probs[get_index2(length,t,r)]);
				}
			}
			for(int r=t+1;r<length;r++){
				outside[the_ind] = logsumexp(outside[the_ind],inside[get_index(length,t,r,E_LEFT,E_INCOM)]
					+outside[get_index(length,s,r,E_LEFT,E_COM)]);
			}
			}
			//2.incomplete
			//2.1 s->t
			the_ind = get_index(length,s,t,E_RIGHT,E_INCOM);
			outside[the_ind] = Negative_Infinity;
			for(int r=t;r<length;r++){
				outside[the_ind] = logsumexp(outside[the_ind],inside[get_index(length,t,r,E_RIGHT,E_COM)]
					+outside[get_index(length,s,r,E_RIGHT,E_COM)]);
			}
			//2.2 t->s
			if(s!=0){
			the_ind = get_index(length,s,t,E_LEFT,E_INCOM);
			outside[the_ind] = Negative_Infinity;
			for(int r=0+1;r<=s;r++){
				outside[the_ind] = logsumexp(outside[the_ind],inside[get_index(length,r,s,E_LEFT,E_COM)]
					+outside[get_index(length,r,t,E_LEFT,E_COM)]);
			}
			}
		}
	}
	return outside;
}

double* encodeMarginals(const int length,const double* scores)
{
	double* marginals = new double[length*length];	//use get_index2
	for(int i=0;i<length*length;i++)
		marginals[i] = 0;
	double* inside = calc_inside(length,scores);
	double* outside = calc_outside(length,inside,scores);
	double z = inside[get_index(length,0,length-1,E_RIGHT,E_COM)];

	for(int i=0;i<length;i++){
		for(int j=i+1;j<length;j++){
			//i->j
			int key_io = get_index(length,i,j,E_RIGHT,E_INCOM);
			int key_assign = get_index2(length,i,j);
			marginals[key_assign] = exp(inside[key_io]+outside[key_io]-z);
			//j->i
			if(i!=0){
			key_io = get_index(length,i,j,E_LEFT,E_INCOM);
			key_assign = get_index2(length,j,i);
			marginals[key_assign] = exp(inside[key_io]+outside[key_io]-z);
			}
		}
	}
	delete []inside;
	delete []outside;
	return marginals;
}
