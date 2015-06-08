/*
 * Helper.h
 *
 *  Created on: 2015Äê6ÔÂ8ÈÕ
 *      Author: zzs
 */

#ifndef ALGORITHMS_HELPER_H_
#define ALGORITHMS_HELPER_H_

#include <cmath>
const double MINUS_LOG_EPSILON=50;

// log(exp(x) + exp(y));
inline double logsumexp(double x, double y) {
	const double vmin = (x>y)?y:x;
	const double vmax = (x<y)?y:x;
	if (vmax > vmin + MINUS_LOG_EPSILON) {
		return vmax;
	}else{
		return vmax + log(exp(vmin - vmax) + 1.0);
	}
}

#endif /* ALGORITHMS_HELPER_H_ */
