/*
 * nn_input_helper.h
 *
 *  Created on: Oct 3, 2015
 *      Author: zzs
 */

#ifndef CSNN_NN_INPUT_HELPER_H_
#define CSNN_NN_INPUT_HELPER_H_

//helper for extra special info
class nn_input_helper{
public:
	//the specified index
	int start_word;
	int end_word;
	int start_pos;
	int end_pos;

	static const int DIST_MAX=10,DIST_MIN=-10;
	static int get_distance_index(int distance){
		if(distance < DIST_MIN)
			distance = DIST_MIN;
		else if(distance > DIST_MAX)
			distance = DIST_MAX;
		return distance-DIST_MIN;
	}
	static int get_distance_num(){
		return DIST_MAX-DIST_MIN+1;
	}
};


#endif /* CSNN_NN_INPUT_HELPER_H_ */
