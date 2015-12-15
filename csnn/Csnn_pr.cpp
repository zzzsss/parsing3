/*
 * Csnn_pr.cpp
 *
 *  Created on: Dec 10, 2015
 *      Author: zzs
 */

//the perceptron part of Csnn

#include "Csnn.h"

/*
 *  the "right" perceptron training:
 *  1. load existing csnn models and start_perceptron for them
 *  2. training: for each minibatch(for sentence{decode();update_pr();} adding()); finish(after several iters)
 */

void Csnn::update_pr(nn_input* good,nn_input* bad,REAL wdecay)
{
	int bsize = good->num_inst;
	nn_math::CHECK_EQUAL(good->num_inst,bad->num_inst,"No match for good and bad.");
	if(bsize <= 0)	//nope
		return;
	nn_cache* c_good;
	nn_cache* c_bad;
	forward(good,1,&c_good);	//testing == 1, fix the other parts (as in testing mode), and no return
	forward(bad,1,&c_bad);

	//perceptron update --- use goals to indicate the label
	int input_size = p_pr->geti();
	for(int i=0;i<bsize;i++){
		if(wdecay > 0)
			p_pr_all->div_w(1-wdecay);	//this is in fact multiply
		p_pr->update_pr(c_good->get_values()+i*input_size,good->goals->at(i),1);
		p_pr->update_pr(c_bad->get_values()+i*input_size,bad->goals->at(i),-1);
	}

	delete c_good;
	delete c_bad;
}

void Csnn::update_pr_adding()
{
	p_pr_all->add_w(p_pr);
	pr_count ++;
}

void Csnn::finish_perceptron()
{
	//maybe there is no way back
	cout << "Averaging perceptron parameters with " << pr_count << " of adding." << endl;
	p_pr_all->div_w(1.0/pr_count);
	p_pr = p_pr_all;
}

