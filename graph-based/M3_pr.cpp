/*
 * M3_pr.cpp
 *
 *  Created on: Dec 10, 2015
 *      Author: zzs
 */

#include "M3_pr.h"
#include "../algorithms/Eisner.h"

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
	//training
	time_t now;
	time(&now); //ctime is not rentrant ! use ctime_r() instead if needed
	cout << "##*** Start the o2 Perceptron training for iter " << cur_iter << " at " << ctime(&now) << endl;
	cout << "#Sentences is " << num_sentences << " and resample (about)" << num_sentences*hp->CONF_NN_resample << endl;

	vector<DependencyInstance*> xs;
	for(int i=0;i<num_sentences;){
		//random skip (instead of shuffling every time)
		if(drand48() > hp->CONF_NN_resample || training_corpus->at(i)->length() >= hp->CONF_higho_toolong){
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
			while(training_corpus->at(i)->length() >= hp->CONF_higho_toolong){	//HAVE to compromise, bad choice
				skip_sent_num ++;
				i ++;
			}
			if(i>=num_sentences)
				break;
			if(int(xs.size()) >= hp->CONF_minibatch)
				break;
		}

		//update
		for(int ii=0;ii<xs.size();ii++){
			DependencyInstance* x = xs[ii];
			nn_input* good_o1,* good_o2;
			nn_input* bad_o1, * bad_o2;
			get_nninput_o1(x,&good_o1,&bad_o1);
			get_nninput_o2sib(x,&good_o2,&bad_o2);
			mso1->update_pr(good_o1,bad_o1);
			mach->update_pr(good_o2,bad_o2);
			delete good_o1;delete bad_o1;
			delete good_o2;delete bad_o2;
		}
		xs.clear();
		mso1->update_pr_adding();
		mach->update_pr_adding();
	}
	cout << "Iter done, skip " << skip_sent_num << " sentences." << endl;

}

void M3_pro2::train()
{
	Process::train();
	//averaging and saving
	cout << "--//don't care about the nn.mach.* files ..." << endl;
	mso1->finish_perceptron();
	mach->finish_perceptron();
	mso1->write(hp->CONF_pr_macho1);
	mach->write(hp->CONF_pr_macho2);
	cout << "--OK, final averaging and saving." << endl;
}

void M3_pro2::test(string x)
{
	cout << "!! Warning, the machine name should be specified in conf, no-use of cmd mach name." << endl;
	mso1 = dynamic_cast<CsnnO1*>(Csnn::read(hp->CONF_pr_macho1));
	//<not here>mach = Csnn::read(hp->CONF_pr_macho2);
	Process::test(hp->CONF_pr_macho2);
}

//---------------helpers---------------
// !! nn-input must be consistent with specified !!
void M3_pro2::get_nninput_o1(DependencyInstance* x,nn_input** good,nn_input**bad)
{
	vector<int>* ginput = new vector<int>();
	vector<int>* ggoals = new vector<int>();
	vector<int>* binput = new vector<int>();
	vector<int>* bgoals = new vector<int>();
	int len = x->length();
	for(int m=1;m<len;m++){
		ginput->push_back(x->heads->at(m));
		ginput->push_back(m);
		ggoals->push_back(x->index_deprels->at(m));
		binput->push_back(x->predict_heads->at(m));
		binput->push_back(m);
		bgoals->push_back(x->predict_deprels->at(m));
	}
	*good = new nn_input(ggoals->size(),2,ginput,ggoals,x->index_forms,x->index_pos,dict->get_helper(),0,0);
	*bad = new nn_input(bgoals->size(),2,binput,bgoals,x->index_forms,x->index_pos,dict->get_helper(),0,0);
}

static int TMP_get_sib(vector<int>* heads,int m)	//-1 for no-sib
{
	int h = heads->at(m);
	int step = (h>m)?1:-1;	//m->h
	int sib;
	for(sib=m+step;sib!=h;sib+=step){
		if(heads->at(sib)==h)
			break;
	}
	return (sib==h)?-1:sib;
}

void M3_pro2::get_nninput_o2sib(DependencyInstance* x,nn_input** good,nn_input**bad)
{
	vector<int>* ginput = new vector<int>();
	vector<int>* ggoals = new vector<int>();
	vector<int>* binput = new vector<int>();
	vector<int>* bgoals = new vector<int>();
	int len = x->length();
	for(int m=1;m<len;m++){
		ginput->push_back(x->heads->at(m));
		ginput->push_back(m);
		ginput->push_back(TMP_get_sib(x->heads,m));
		ggoals->push_back(x->index_deprels->at(m));
		binput->push_back(x->predict_heads->at(m));
		binput->push_back(m);
		binput->push_back(TMP_get_sib(x->predict_heads,m));
		bgoals->push_back(x->predict_deprels->at(m));
	}
	*good = new nn_input(ggoals->size(),3,ginput,ggoals,x->index_forms,x->index_pos,dict->get_helper(),0,0);
	*bad = new nn_input(bgoals->size(),3,binput,bgoals,x->index_forms,x->index_pos,dict->get_helper(),0,0);
}
