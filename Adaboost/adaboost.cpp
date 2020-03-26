#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <locale> 
#include <bits/stdc++.h> 
#include <math.h> 
using namespace std; 

//prints input to insure it is split correclty 
void printinput( vector<vector<double>> a, vector<int> b){
    for (int i=0; i<a.size(); i++){
        for(int j=0; j<a[i].size(); j++){
            if (j==1){
                cout << b[i] << " ";
            }
            cout << a[i][j] << " "; 
        }
        cout << endl;
        cout << endl;
        cout << endl;
    }

}

//adds all values within a vector
int sum(vector<double> x){
	int temp =0;
	for(int i=0; i<x.size(); i++){
		temp+=x[i];
	}
	return temp;
}



//stump builder
class stump{
	public:
	stump(){

	}

	vector<double> build_stumnp(vector<double> x, vector<double> y, int w){
		int d = x.size();       
		int z = w/sum(y);

		vector<vector<double>> stump;
		vector<vector<double>> werr; //zeros
		vector<double> stumper;

		for(int i=0; i<d; i++ ){
			//stump[i] = build_onedim_stump(x,y,z);
			stump[i].push_back(i);
			werr[i].push_back(0);
			werr[i] = stump[i];
		}

		//min(werr);
		//set best stump

		return stumper;

	}


	vector<double> build_onedim_stump(vector<double> x, vector<double> y, int z){

		vector<double> asc = x;
		vector<double> desc = x;
		vector<double> stumper;

		sort(asc.begin(), asc.end());
		sort(desc.begin(), desc.end(), greater<int>());

		int left = 1;
		int right = 0;
		int score = left + right;

		vector<double> dist;

		if(dist.size() > 0){
			stumper.push_back(0);
		}else{
			stumper.push_back(1);
		}
		return stumper;

	}


 	double get_error(){
 		return 0.2;
 	}

 	double get_feature(){
 		return 0.2;
 	}

 	double get_split(){
 		return 0.5;
 	}

 	vector<int> get_stump(){

 	}


};


class AdaBoost{
    public: 
    vector<vector<double>> features;
    vector<int> diagnosis;
    vector<int> at;
    vector<int> dt;
    vector<int> ht;
    vector<int> splitter;

    AdaBoost(vector<vector<double>> a, vector<int> b){
        features = a;
        diagnosis =b;
        dt.push_back(1/b.size());
    }


    void train(){
    	double error = 0;
    	double aos = 0;
    	double d1= 0;
    	double d2= 0;
    	double weight = (1.0000/diagnosis.size()) ;
    	double f = 0 ;
    	int split = 0 ;
    	vector<vector<double>> temp = features;
    	for (int i=1; i<diagnosis.size(); i++){
    		for(int i=0; i<diagnosis.size(); i++){
    			temp[i][0] = (1.0000/diagnosis.size());
    		}

    		double norm = 0;
    		stump tree;
    		//tree.build_stumnp()

    		f = tree.get_feature();
    		split = tree.get_split();
    		splitter.push_back(split);

    		error = tree.get_error();
    		aos = AmountofSay(error);

    		d1 = distribution(weight, -1.0*aos);
    		d2 = distribution(weight, aos);

    		//sets new weight
    		for(int i=0; i<diagnosis.size(); i++){
    			if (temp[i][f]>split && diagnosis[i]==0){
    				temp[i][0] = d1;
    				norm+=d1;
    			}else{
    				temp[i][0] = d2;
    				norm+=d2;
    			}
    		}

    		//normalizes weight
    		for(int i=0; i<diagnosis.size(); i++){
    			temp[i][0] = temp[i][0]/norm;
    		}



    		vector<vector<double>> newtemp;
    		//make new sample collection
    		for(int i=0; i<diagnosis.size(); i++){
    			int rnd = rand() % 101;
    			double counter = 1;
    			for(int i=0; i<diagnosis.size(); i++){
    				if(counter>=rnd){
    					newtemp.push_back(temp[i]);
    					break;
    				}
    				 counter+=temp[i][0]*100;
    			}
    		}

    		temp = newtemp;

    	}


    }

    void test(vector<vector<double>> a, vector<int> b){
        //convert data
        for (int j=1; j<a[1].size(); j++){
	        vector<double> temp;
	        for (int i=0; i<a.size(); i++){
	            if(a[i][j] >= splitter[i]){
	               a[i][j] = 1;
	            }else{
	                a[i][j] = -1;
	           }
	       }
	   }

	   

    }




	//sets the amount of say at t and also returns value to trainer
	double AmountofSay(double error){
		
		double ln =  log((1.0-error)/error);
		double aos = (1.0/2.0)*ln;
		at.push_back(aos);
	    return aos;
	}

	//sets the new weight at t and also returns value to trainer
	double distribution(double weight, double aos){
		//needs normalization factor
		double d = (weight*exp((-1.0)*aos));
		dt.push_back(d);
	    return d; 
	}

};







int main (int argc, char **argv) {

	if (argc==1){
		cout << "Please include csv file in argument" << endl;
		return 0;
	}

    //segment 1
    //gets input file and splits into features and diagnosis

    ifstream ifs;
    ifs.open (argv[1]);
    string str, dub;                 
    vector<vector<double>> features;    
    vector<int> diagnosis;


    while (getline (ifs, str)){        
        vector<double> temp;                
        stringstream ss(str);         
        while (getline(ss, dub, ',')){  
            if (isalpha(dub[0])){
                if (dub[0]=='M'){
                    diagnosis.push_back(1);
                }else{
                    diagnosis.push_back(0);
                }
            } else{ 
                temp.push_back(stod(dub));  
            }
        }
        features.push_back(temp);             
    }

    //splits into training and test data
    vector<vector<double>> f_train(features.begin(), features.begin()+299);
    vector<int> d_train(diagnosis.begin(), diagnosis.begin()+299);
    vector<vector<double>> f_test(features.begin()+300, features.end());
    vector<int> d_test(diagnosis.begin(), diagnosis.begin()+299);
    //printinput(f_train, d_train);

    //segment 2
   	AdaBoost wbdc(f_train,d_train);
   	wbdc.train();
    //wbdc.test(f_test, d_test);

    return 0;

}