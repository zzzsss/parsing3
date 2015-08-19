/*
 * FeatureGenO3g.cpp
 *
 *  Created on: 2015Äê4ÔÂ20ÈÕ
 *      Author: zzs
 */

#include "FeatureGenO3g.h"

FeatureGenO3g::FeatureGenO3g(Dict* d,int w,int di,int apos,int dir,int mv):FeatureGen(d,w,di,apos,dir,mv)
{
	xdim = 4*w;
	if(apos)
		xdim *= 2;
	if(di){
		xdim += 3;
	}
	if(dir)
		xdim ++;
}

int FeatureGenO3g::fill_one(REAL* to_fill,DependencyInstance* ins,int head,int mod,int mod_center,int g)
{
	int off_h=0,off_m=0,off_c=0,off_g=0;
	if(multi_vec){
		int count = dictionary->get_count();
		off_h = count * MVEC_H;
		off_m = count * MVEC_M;
		off_c = count * MVEC_C;
		off_g = count * MVEC_G;
	}
	//head-w,mod_center-w,mod-w,g-w,head-pos,mod_center-w,mod-pos,g-pos,distances
	int backward = window_size/2;	//window_size should be odd...
	int leng = ins->forms->size();

	//1.head
	for(int i=head-backward;i<=head+backward;i++){
		if(i<0)					*to_fill = off_h+dictionary->get_index(&dictionary->WORD_START,0);	//must exist
		else if(i>=leng)		*to_fill = off_h+dictionary->get_index(&dictionary->WORD_END,0);	//must exist
		else					*to_fill = off_h+ins->index_forms->at(i);
		to_fill ++;
	}
	if(pos_add){
		for(int i=head-backward;i<=head+backward;i++){
			if(i<0)				*to_fill = off_h+dictionary->get_index(&dictionary->POS_START,0);	//must exist
			else if(i>=leng)	*to_fill = off_h+dictionary->get_index(&dictionary->POS_END,0);	//must exist
			else				*to_fill = off_h+ins->index_pos->at(i);
			to_fill ++;
		}
	}
	if(distance){
		*to_fill = off_h+dictionary->get_index(g-head);
		to_fill ++;
	}

	//2.center
	if(mod_center<0){	//use dummy
		for(int i=mod_center-backward;i<=mod_center+backward;i++){	//**ONCE A BUG**
			if(head < mod)		*to_fill = off_c+dictionary->get_index(&dictionary->WORD_DUMMY_L,0);
			else				*to_fill = off_c+dictionary->get_index(&dictionary->WORD_DUMMY_R,0);
			to_fill ++;
		}
	}
	else{
		for(int i=mod_center-backward;i<=mod_center+backward;i++){
			if(i<0)				*to_fill = off_c+dictionary->get_index(&dictionary->WORD_START,0);	//must exist
			else if(i>=leng)	*to_fill = off_c+dictionary->get_index(&dictionary->WORD_END,0);	//must exist
			else				*to_fill = off_c+ins->index_forms->at(i);
			to_fill ++;
		}
	}
	if(pos_add){
		if(mod_center<0){	//use dummy
			for(int i=mod_center-backward;i<=mod_center+backward;i++){		//**ONCE A BUG**
				if(head < mod)	*to_fill = off_c+dictionary->get_index(&dictionary->POS_DUMMY_L,0);
				else			*to_fill = off_c+dictionary->get_index(&dictionary->POS_DUMMY_R,0);
				to_fill ++;
			}
		}
		else{
			for(int i=mod_center-backward;i<=mod_center+backward;i++){
				if(i<0)			*to_fill = off_c+dictionary->get_index(&dictionary->POS_START,0);	//must exist
				else if(i>=leng)*to_fill = off_c+dictionary->get_index(&dictionary->POS_END,0);	//must exist
				else			*to_fill = off_c+ins->index_pos->at(i);
				to_fill ++;
			}
		}
	}
	if(distance){
		if(mod_center<0){
			*to_fill = off_c+dictionary->get_index(&dictionary->DISTANCE_DUMMY,0);
			to_fill ++;
		}
		else{
			*to_fill = off_c+dictionary->get_index(head-mod_center);
			to_fill ++;
		}
	}

	//3.modifier
	for(int i=mod-backward;i<=mod+backward;i++){
		if(i<0)					*to_fill = off_m+dictionary->get_index(&dictionary->WORD_START,0);	//must exist
		else if(i>=leng)		*to_fill = off_m+dictionary->get_index(&dictionary->WORD_END,0);	//must exist
		else					*to_fill = off_m+ins->index_forms->at(i);
		to_fill ++;
	}
	if(pos_add){
		for(int i=mod-backward;i<=mod+backward;i++){
			if(i<0)				*to_fill = off_m+dictionary->get_index(&dictionary->POS_START,0);	//must exist
			else if(i>=leng)	*to_fill = off_m+dictionary->get_index(&dictionary->POS_END,0);	//must exist
			else				*to_fill = off_m+ins->index_pos->at(i);
			to_fill ++;
		}
	}
	if(distance){
		*to_fill = off_m+dictionary->get_index(head-mod);
		to_fill ++;
	}

	//4.grand-parent
	if(g==0 && head==0){
		for(int i=g-backward;i<=g+backward;i++){
			*to_fill = off_g+dictionary->get_index(&dictionary->WORD_ROOTG,0);
			to_fill++;
		}
		if(pos_add){
			for(int i=g-backward;i<=g+backward;i++){
				*to_fill = off_g+dictionary->get_index(&dictionary->POS_ROOTG,0);
				to_fill++;
			}
		}
	}
	else{
	for(int i=g-backward;i<=g+backward;i++){
		if(i<0)					*to_fill = off_g+dictionary->get_index(&dictionary->WORD_START,0);	//must exist
		else if(i>=leng)		*to_fill = off_g+dictionary->get_index(&dictionary->WORD_END,0);	//must exist
		else					*to_fill = off_g+ins->index_forms->at(i);
		to_fill ++;
	}
	if(pos_add){
		for(int i=g-backward;i<=g+backward;i++){
			if(i<0)				*to_fill = off_g+dictionary->get_index(&dictionary->POS_START,0);	//must exist
			else if(i>=leng)	*to_fill = off_g+dictionary->get_index(&dictionary->POS_END,0);	//must exist
			else				*to_fill = off_g+ins->index_pos->at(i);
			to_fill ++;
		}
	}
	}
	if(dir_add){
		*to_fill = off_g+dictionary->get_index(mod,head,g);
		to_fill ++;
	}

	return xdim;
}
