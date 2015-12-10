/*
 * M3_pr.cpp
 *
 *  Created on: Dec 10, 2015
 *      Author: zzs
 */

#include "M3_pr.h"

//only before training (and after building dictionary)
void M3_pro2::each_create_machine()
{
	//mfo1 init at M3_pro2()
	mso1 = dynamic_cast<CsnnO1*>(Csnn::read(hp->CONF_score_mach_so1));
	mach = Csnn::read(hp->CONF_score_mach_so2sib);

	//enforcing labeling
	int class_num = dict->getnum_deprel();
	Process::CHECK_EQUAL(mso1->get_odim(),class_num,"Pr. so1 bad odim.");
	Process::CHECK_EQUAL(mach->get_odim(),class_num,"Pr. so2sib bad odim.");

	mso1->start_perceptron(class_num);
	mach->start_perceptron(class_num);
	//but not for fo1 !!
}

void M3_pro2::each_test_one(DependencyInstance* x,int dev)
{
	//Here ignore dev or not
	Process::parse_o2sib(x,mfo1,mso1);
}

void M3_pro2::each_train_one_iter()
{
	static bool** STA_noprobs = 0;	//static ine, init only once
	if(STA_noprobs==0 && !filter_read(STA_noprobs)){
		//init only once
		int all_tokens_train=0,all_token_filter_wrong=0;
		time_t now;
		time(&now);
		cout << "-Preparing no_probs at " << ctime(&now) << endl;
		STA_noprobs = new bool*[training_corpus->size()];
		for(unsigned int i=0;i<training_corpus->size();i++){
			DependencyInstance* x = training_corpus->at(i);
			STA_noprobs[i] = get_cut_o1(x,mfo1,dict,hp->CONF_score_o1filter_cut);
			all_tokens_train += x->length()-1;
			for(int m=1;m<x->length();m++)
				if(STA_noprobs[i][get_index2(x->length(),x->heads->at(m),m)])
					all_token_filter_wrong ++;
		}
		cout << "For o1 filter: all " << all_tokens_train << ";filter wrong " << all_token_filter_wrong << endl;
		filter_write(STA_noprobs);
	}

	//per-sentence approach
	int num_sentences = training_corpus->size();
	//statistics
	int skip_sent_num = 0;
	int all_forward_instance = 0;
	//some useful info
	int odim = mach->get_odim();
	//training
	time_t now;
	time(&now); //ctime is not rentrant ! use ctime_r() instead if needed
	cout << "##*** Start the o2 Perceptron training for iter " << cur_iter << " at " << ctime(&now) << endl;
	cout << "#Sentences is " << num_sentences << " and resample (about)" << num_sentences*hp->CONF_NN_resample << endl;

	vector<DependencyInstance*> xs;
	for(int i=0;i<num_sentences;){
		//random skip (instead of shuffling every time)
		if(drand48() > hp->CONF_NN_resample){
			skip_sent_num ++;
			i ++;
			continue;
		}
		//main batch
		for(;;){
			//forward
			DependencyInstance* x = training_corpus->at(i);
			xs.push_back(x);

			Process::parse_o2sib(x,mfo1,mso1);

			//out of the mini-batch
			if(i>=num_sentences)
				break;
			if(int(xs.size()) >= hp->CONF_minibatch)
				break;
		}

		//update

	}
	cout << "Iter done, skip " << skip_sent_num << " sentences." << endl;

}

void M3_pro2::train()
{
	Process::train();
	//averaging and saving
	cout << "--//don't care about the nn.mach.* files ..." << endl;
}

void M3_pro2::test(string)
{

}
