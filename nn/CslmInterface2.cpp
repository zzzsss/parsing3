/*
 * CslmInterface2.cpp
 *
 *  Created on: 2015Äê6ÔÂ10ÈÕ
 *      Author: zzs
 */

#include "CslmInterface.h"

void CslmInterface::mach_backward(REAL* assign,REAL* gradient,const float lrate,const float wdecay,int all)
{
	if(all==0)
		return;
	//if bs small, maybe need forward
	Mach* m = mach;
	int idim = m->GetIdim();
	int odim = m->GetOdim();
	int bsize = m->GetBsize();

	int remain = all;
	REAL* setg = gradient + all*odim;
	REAL* setd = assign + all*idim;
	int this_num = all % bsize;
	if(this_num==0)
		this_num = bsize;
	while(remain > 0){
		setg -= this_num*odim;
		setd -= this_num*idim;
		if(remain != all){
			//need forward
			mach->SetDataIn(setd);
			mach->Forw(this_num);
		}
		mach->SetGradOut(setg);
		mach->Backw(lrate,wdecay,this_num);
		remain -= this_num;
		this_num = bsize;
	}
}

void CslmInterface::mach_forwback(REAL* assign,REAL* gradient,const float lrate,const float wdecay,int all)
{
	if(all==0)
		return;
	//gradient is given, all forward and backward, don't care about outputs
	Mach* m = mach;
	int idim = m->GetIdim();
	int odim = m->GetOdim();
	int bsize = m->GetBsize();

	int remain = all;
	int this_num = bsize;
	REAL* xx = assign;
	REAL* gg = gradient;
	while(remain > 0){
		if(remain < this_num)
			this_num = remain;
		mach->SetDataIn(xx);
		mach->Forw(this_num);
		mach->SetGradOut(gg);
		mach->Backw(lrate,wdecay,this_num);
		xx += this_num*idim;
		gg += this_num*odim;
		remain -= this_num;
	}
}

