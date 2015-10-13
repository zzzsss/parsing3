#include<iostream>
#include"DependencyInstance.h"
using namespace std;

void DependencyInstance::init(){
	forms=0;
	postags=0;
	heads=0;
	deprels=0;
	index_forms=0;
	index_pos=0;
	index_deprels=0;
	predict_heads=0;
	predict_deprels=0;
	predict_deprels_str=0;
}
DependencyInstance::DependencyInstance(std::vector<string*> *forms,
		std::vector<string*> *postags,std::vector<string*> *deprels,std::vector<int> *heads){
	init();
	this->forms = forms;
	this->heads = heads;
	this->postags = postags;
	this->deprels = deprels;
}
int DependencyInstance::length(){
	return (int)(forms->size());
}

DependencyInstance::~DependencyInstance(){
	vector<string*>::iterator iter;
	for(iter = forms->begin(); iter != forms->end(); ++iter){
		delete (*iter);
	}
	for(iter = postags->begin(); iter != postags->end(); ++iter){
		delete (*iter);
	}
	for(iter = deprels->begin(); iter != deprels->end(); ++iter){
		delete (*iter);
	}
	delete(forms);
	delete(postags);
	delete(heads);
	delete(deprels);
	delete(index_forms);
	delete(index_pos);
	delete index_deprels;
	delete predict_heads;
	delete predict_deprels;
	delete predict_deprels_str;	//!!don't delete the string here, they are from dictionary
}

