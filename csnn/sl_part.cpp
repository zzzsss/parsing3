/*
 * sl_part.cpp
 *
 *  Created on: Dec 4, 2015
 *      Author: zzs
 */

#include "sl_part.h"

void sl_part::forward(nn_input* inputs,REAL* out)
{
	this_input = inputs;
	const long this_size = inputs->num_inst;
	const long this_len = this_input->wordl->size();
	const int the_win = op->NN_sl_filter;	//should be odd-number

	const int idim_wp = p_main->geti();
	const int idim_dist = p_dist->geti();
	const int odim = p_main->geto();

	//1.allocate memory
	delete []c_input_wp;
	delete []c_input_dist;
	delete []c_output_wp;
	delete []c_output_dist;
	delete []c_output_tmp;
	delete []c_which;
	c_input_wp = new REAL[idim_wp*this_len];
	c_input_dist = new REAL[idim_dist*this_len*this_size];
	c_output_wp = new REAL[odim*this_len];
	c_output_dist = new REAL[odim*this_len*this_size];
	c_output_tmp = new REAL[odim*this_len*this_size];
	c_which = new int[odim*this_size];

	//2.fill inputs
	//2.1 sentence
	REAL* to_assign = c_input_wp;
	for(int i=0;i<this_len;i++){	//for each center word
		for(int w=i-the_win/2;w<=i+the_win/2;w++){
			int single_index,pos_index;
			if(w<0){
				single_index = this_input->helper->start_word;
				pos_index = this_input->helper->start_pos;
			}
			else if(w>=this_len){
				single_index = this_input->helper->end_word;
				pos_index = this_input->helper->end_pos;
			}
			else{
				single_index = this_input->wordl->at(w);
				pos_index = this_input->posl->at(w);
			}
			d_word->forward(single_index,to_assign);
			to_assign += d_word->getd();
			d_pos->forward(single_index,to_assign);
			to_assign += d_pos->getd();
		}
	}
	nn_math::CHECK_EQUAL(to_assign,c_input_wp+idim_wp*this_len,"Forward error of c_input_wp.");
	//2.2 sl distances
	to_assign = c_input_dist;
	for(vector<int>::const_iterator iter=this_input->inputs->begin();iter!=this_input->inputs->end();iter+=this_input->num_width){
		for(int i=0;i<this_len;i++){
			for(int ord=0;ord<this_input->num_width;ord++){
				int location = *(iter+ord);
				if(location < 0)
					d_ds->forward(this_input->helper->get_sd_dummy(),to_assign);
				else{
					location = this_input->helper->get_sd_index(location - i);
					d_ds->forward(location,to_assign);
				}
				to_assign += d_ds->getd();
			}
		}
	}
	nn_math::CHECK_EQUAL(to_assign,c_input_dist+idim_dist*this_len*this_size,"Forward error of c_input_dist.");

	//3.forward
	p_main->forward(c_input_wp,c_output_wp,this_len);
	p_dist->forward(c_input_dist,c_output_dist,this_len*this_size);

	//4.attach distance
	for(int i=0;i<this_size;i++)
		memcpy(c_output_tmp+i*odim*this_len,c_output_wp,sizeof(REAL)*odim*this_len);
	switch(op->NN_sl_way){
	case nn_options::NN_SL_ADDING:	//directly adding
		nn_math::op_y_plus_ax(odim*this_len*this_size,c_output_tmp,c_output_dist,1);
		break;
	case nn_options::NN_SL_TANHMUL:	//element-wise multiply tanh(dist)
		nn_math::act_f(nn_math::ACT_TANH,c_output_dist,odim*this_len*this_size,0);
		nn_math::op_y_elem_x(odim*this_len*this_size,c_output_tmp,c_output_dist);
		break;
	}

	//5.max-pooling
	REAL* mto_assign = out;
	int* mto_index = c_which;
	const REAL* mfrom_assign = c_output_tmp;
	for(int inst=0;inst<this_size;inst++){
		//(1).all set to 0
		memcpy(mto_assign,mfrom_assign,sizeof(REAL)*odim);
		for(int tmpi=0;tmpi<odim;tmpi++)
			mto_index[tmpi] = 0;
		//(2).update max one
		for(int i=1;i<this_len;i++){	//from next one
			for(int elem=0;elem<odim;elem++){
				REAL TMP_cur = *mfrom_assign++;	//add here
				if(TMP_cur > mto_assign[elem]){
					mto_assign[elem] = TMP_cur;
					mto_index[elem] = i;
				}
			}
		}
		mto_assign += odim;
		mto_index += odim;
	}
	nn_math::CHECK_EQUAL(mto_assign,out+odim*this_size,"Forward error of mto_assign.");
	nn_math::CHECK_EQUAL(mto_index,c_which+odim*this_size,"Forward error of mto_index.");
	nn_math::CHECK_EQUAL(mfrom_assign,(const REAL*)c_output_tmp+odim*this_len*this_size,"Forward error of mfrom_assign.");

	return;
}


void sl_part::backward(/*const*/REAL* ograd)
{

}
