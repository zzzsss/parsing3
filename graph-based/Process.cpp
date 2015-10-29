/*
 * Process.cpp
 *
 *  Created on: Oct 10, 2015
 *      Author: zzs
 */

#include "Process.h"

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
{
	return !nn_math::opt_changelrate[hp->CONF_UPDATE_WAY] && (hp->CONF_NN_LMULT<0)
			   && ((lrate_cut_times+lrate_force_cut_times) < hp->CONF_NN_ITER_decrease);
}

