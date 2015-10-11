#ifndef DependencyInstance_H
#define DependencyInstance_H
#include<vector>
#include<string>
using namespace std;

/* An instance means one sentence
 * 	--- also simpilfied version
 */

class DependencyInstance{
private:
	void init();
public:
	vector<string*>* forms;
	vector<string*>* postags;
	vector<int>* heads;
	vector<string*>* deprels;

	//used when decoding --- set up after construction of dictionary
	vector<int>* index_forms;
	vector<int>* index_pos;
	vector<int>* index_deprels;

	//to-predict
	vector<int>* predict_heads;
	vector<int>* predict_deprels;

	DependencyInstance();
	DependencyInstance(vector<string*>* forms, vector<string*>* postags,
			std::vector<string*> *deprels,vector<int>* heads);
	~DependencyInstance();
	//void setFeatureVector(FeatureVector* fv);
	int length();
	string toString();
	//void writeObject(ObjectWriter &out);
	//void readObject(ObjectReader &in);
};
#endif
