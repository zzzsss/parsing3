/*
 * Csnn.cpp
 *
 *  Created on: Sep 25, 2015
 *      Author: zzs
 */

#include "Csnn.h"

//--prepares--
void Csnn::construct_caches(){
	//this is done when init and read
	c_allcaches = new vector<nn_cache*>();
	c_out = new nn_cache(0,the_option->NN_out_size);	c_allcaches->push_back(c_out);
	c_h = new nn_cache(0,the_option->NN_hidden_size);	c_allcaches->push_back(c_h);
	c_repr = new nn_cache(0,the_option->get_NN_rsize());	c_allcaches->push_back(c_repr);
	c_wrepr = new nn_cache(0,the_option->NN_wrsize);	c_allcaches->push_back(c_wrepr);
	c_srepr = new nn_cache(0,the_option->get_NN_srsize());	c_allcaches->push_back(c_srepr);
	c_wv = new nn_cache(0,the_option->get_NN_wv_wrsize(get_order()));	c_allcaches->push_back(c_wv);
}

void Csnn::prepare_caches(int bsize){
	//only need to clear all the gradients and get possible drop-out ready
	//this is done before each MiniBatch
	for(vector<nn_cache*>::iterator i=c_allcaches->begin();i!=c_allcaches->end();i++){
		(*i)->resize(bsize);
		(*i)->clear_gradients();
		if(the_option->NN_dropout > 0)
			(*i)->gen_dropout(the_option->NN_dropout);
	}
}

void Csnn::construct_params(){
	//init all the params
	p_out = new nn_wb(the_option->NN_hidden_size,the_option->NN_out_size);
	p_out->get_init(the_option->NN_init_wbrange);
	p_h = new nn_wb(the_option->get_NN_rsize(),the_option->NN_hidden_size);
	p_h->get_init(the_option->NN_init_wbrange);
	//special untied param
	int all = the_option->NN_pnum*the_option->NN_pnum+1;
	p_untied = new vector<nn_wb*>(all,0);
	p_untied->at(0) = new nn_wb(the_option->get_NN_wv_wrsize(get_order()),the_option->NN_wrsize);
	(p_untied->at(0))->get_init(the_option->NN_init_wbrange);
	//the embeddings
	p_word = new nn_wv(the_option->NN_wnum,the_option->NN_wsize);
	p_word->get_init(the_option->NN_init_wvrange);
	p_pos = new nn_wv(the_option->NN_pnum,the_option->NN_psize);
	p_pos->get_init(the_option->NN_init_wvrange);
	p_distance = new nn_wv(the_option->NN_dnum,the_option->NN_dsize);
	p_distance->get_init(the_option->NN_init_wvrange);
}

void Csnn::read_params(std::ifstream fin){
	p_out = new nn_wb(fin);
	p_h = new nn_wb(fin);
	//special untied param
	int un_num = 0;
	fin.read((char*)&un_num,sizeof(int));
	int all = the_option->NN_pnum*the_option->NN_pnum+1;
	p_untied = new vector<nn_wb*>(all,0);
	for(int i=0;i<un_num;i++){
		int tmp_index = 0;
		fin.read((char*)&tmp_index,sizeof(int));
		p_untied->at(tmp_index) = new nn_wb(fin);
	}
	//embeddings
	p_word = new nn_wv(fin);
	p_pos = new nn_wv(fin);
	p_distance = new nn_wv(fin);
}

void Csnn::write_params(std::ofstream fout){
	p_out->write_params(fout);
	p_h->write_params(fout);
	//special untied param
	int all = the_option->NN_pnum*the_option->NN_pnum+1;
	int un_num = 0;
	for(int i=0;i<all;i++){
		if(p_untied->at(i) != 0)
			un_num ++;
	}
	fout.write((char*)&un_num,sizeof(int));
	for(int i=0;i<all;i++){
		if(p_untied->at(i) != 0){
			fout.write((char*)&i,sizeof(int));
			p_untied->at(i)->write_params(fout);
		}
	}
	//embeddings
	p_word->write_params(fout);
	p_pos->write_params(fout);
	p_distance->write_params(fout);
}

//-----------main methods-------------------------------
//if testing, no adding untied-nnwb, untied_rate<=0 means no untied
REAL* Csnn::forward(nn_input* in,int testing,REAL untied_rate)
{
	//1.prepare inputs
	this_input = in;
	this_bsize = in->get_numi();
	this_mbsize += this_bsize;
	prepare_caches(this_bsize);
	f_inputs();		/**********VIRTUAL***********/	//now c_wv ready, and here also prepare this_untied_index

	//2.1:input->wrepr --- need take care of untied
	if(testing){

	}
	else{

	}
}

