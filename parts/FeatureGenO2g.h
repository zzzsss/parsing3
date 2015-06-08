/*
 * FeatureGenO2g.h
 *
 *  Created on: 2015��4��21��
 *      Author: zzs
 */

#ifndef PARTS_FEATUREGENO2G_H_
#define PARTS_FEATUREGENO2G_H_

#include "FeatureGen.h"
class FeatureGenO2g: public FeatureGen{
public:
	FeatureGenO2g(Dict* d,int w,int di,int apos,int dir);
	virtual int fill_one(REAL*,DependencyInstance*,int head,int mod,int g,int no_use=-1);
	virtual ~FeatureGenO2g(){}
	virtual int get_order(){return 2;}

	//no-use
	virtual void add_filter(vector<DependencyInstance*>*){}
	int allowed_pair(DependencyInstance* x,int head,int modif,int c){return -1;}
};



#endif /* PARTS_FEATUREGENO2G_H_ */
