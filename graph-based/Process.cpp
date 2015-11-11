/*
 * Process.cpp
 *
 *  Created on: Oct 10, 2015
 *      Author: zzs
 */

#include "Process.h"
#include "../algorithms/Eisner.h"

//---------------------------INIT-------------------------------//
Process::Process(string conf)
{
	//here only prepare all the HypherParameters
	cout << "1.configuration:" << endl;
	hp = new HypherParameters(conf);
}


//---------------------------LRATE-------------------------------//
int Process::set_lrate_one_iter()	//currently: return has no meaning
{
	if(!nn_math::opt_changelrate[hp->CONF_UPDATE_WAY])
		return 0;
	if(hp->CONF_NN_LMULT<0 && cur_iter>0){
		//special schedule in (-1,0)
		if(hp->CONF_NN_LMULT > -1){
			if(dev_results[cur_iter] < dev_results[cur_iter-1]){
				cur_lrate *= (-1 * hp->CONF_NN_LMULT);
				lrate_cut_times++;
				last_cut_iter = cur_iter;
				return 1;
			}
			else if((cur_iter-last_cut_iter) >= hp->CONF_NN_ITER_force_half){
				//force cut
				cur_lrate *= (-1 * hp->CONF_NN_LMULT);
				lrate_force_cut_times++;
				last_cut_iter = cur_iter;
				return 1;
			}
			else
				return 0;
		}
	}
	return 0;
}

int Process::whether_keep_trainning()
{	//DEBUG: !
	return nn_math::opt_changelrate[hp->CONF_UPDATE_WAY] && (hp->CONF_NN_LMULT<0)
			   && ((lrate_cut_times+lrate_force_cut_times) < hp->CONF_NN_ITER_decrease);
}


void Process::init_embed()
{
	if(!(hp->CONF_embed_WL.length()>0 && hp->CONF_embed_EM.length()>0))	//nothing to do if not setting
		return;

	//temps
	const int INIT_EM_MAX_SIZE = 1000000;
	HashMap t_maps(INIT_EM_MAX_SIZE);
	vector<REAL*> t_embeds;
	vector<string*> t_words;

	cout << "--Init embedding from " << hp->CONF_embed_WL << " and " << hp->CONF_embed_EM << "\n";
	ifstream fwl,fem;
	fwl.open(hp->CONF_embed_WL.c_str());
	fem.open(hp->CONF_embed_EM.c_str());
	if(!fwl || !fem)
		FatalError("Failed when opening embedding file.");
	int TMP_count = 0;
	while(fwl){
		if(!fem)
			FatalError("No match with embedding files.");
		string one_word;
		fwl >> one_word;
		REAL* temp = new REAL[hp->hp_nn.NN_wsize];
		for(int i=0;i<hp->hp_nn.NN_wsize;i++){
			fem >> temp[i];
			temp[i] *= hp->CONF_embed_ISCALE;
		}
		string* one_word_p = new string(one_word);
		t_maps.insert(pair<string*, int>(one_word_p,TMP_count++));
		t_embeds.push_back(temp);
		t_words.push_back(one_word_p);
	}
	fwl.close();
	fem.close();

	//start
	HashMap *inner_map = dict->get_wordmap();
	int n_all = dict->getnum_word();
	int n_check = 0;
	int n_icheck = 0;
	for(HashMap::iterator i = inner_map->begin();i!=inner_map->end();i++){
		string* one_str = i->first;
		int one_index = i->second;
		//serach here
		HashMap::iterator iter = t_maps.find(one_str);
		if(iter == t_maps.end()){
			//tolower
			string one_str_temp = (*one_str);
			for(int j=0;j<one_str_temp.size();j++)
				one_str_temp[j] = tolower(one_str_temp[j]);
			//find again
			iter = t_maps.find(&one_str_temp);
			if(iter != t_maps.end())
				n_icheck++;
		}
		else
			n_check++;

		if(iter != t_maps.end()){
			mach->init_word_tab(one_index,t_embeds[iter->second]);
		}
	}

	//clear
	for(int i=0;i<t_words.size();i++){
		delete []t_embeds[i];
		delete t_words[i];
	}
	cout << "-- Done, with " << n_all << "/" << n_check << "/" << n_icheck << '\n';
}

void Process::check_o1_filter(string m_name,string cutting)
{
	//MUST BE O1 MACH
	cout << "----- Check o1 filter(must be o1-mach)-----" << endl;
	hp->CONF_score_o1filter_cut = atof(cutting.c_str());
	dict = new Dictionary(hp->CONF_dict_file);
	mach = Csnn::read(m_name);
	dev_test_corpus = read_corpus(hp->CONF_test_file);
	dict->prepare_corpus(dev_test_corpus,1);

	int token_num = 0;	//token number
	int filter_wrong_count = 0;
	for(int ii=0;ii<dev_test_corpus->size();ii++){
		if(ii%100 == 0)
			cout << filter_wrong_count << "/" << token_num << endl;
		DependencyInstance* x = dev_test_corpus->at(ii);
		int length = x->forms->size();
		bool* tmp_cut = get_cut_o1(x,dynamic_cast<CsnnO1*>(mach),dict,hp->CONF_score_o1filter_cut);
		for(int i=1;i<length;i++){
			if(tmp_cut[get_index2(length,x->heads->at(i),i)])
				filter_wrong_count++;
			token_num++;
		}
		delete []tmp_cut;
	}
	cout << "FINAL:" << filter_wrong_count << "/" << token_num  << "=" << filter_wrong_count/(0.0+token_num) << endl;
}
