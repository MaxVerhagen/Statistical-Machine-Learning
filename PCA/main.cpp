#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <locale> 
#include <bits/stdc++.h> 
#include <math.h> 

#include <eigen3/Eigen/Eigenvalues>

//#include <opencv2/core/core.hpp>
//#include <opencv2/ml/ml.hpp>
//using namespace cv;

using namespace std; 





vector<double> meanV(vector<vector<double>> a){
	vector<double> temp;
	double mean = 0; 
	double sum = 0;
	for(int i=0; i<a[0].size(); i++){
		sum =0;
		for(int l=0; l<a.size(); l++){
			sum+=a[l][i];
		}
		mean = sum/a.size();
		temp.push_back(mean);
	}
	return temp;
}


vector<vector<double>> covariance(vector<vector<double>> a, vector<double> mean, int dim){
	
	double multi = 0;
	vector<double> temp;
	vector<vector<double>> varience;

	for(int i=0; i<a[0].size(); i++){
		for(int l=0; l<a.size(); l++){
			a[l][i]=a[l][i]-mean[i];
		}
	}

	for(int i=0; i<dim; i++){
		temp.clear();
		for(int k=0; k<dim; k++){
			multi = 0;
			for(int l=0; l<a.size(); l++){
				multi += a[l][i]*a[l][k];
			}
			multi = multi/(a.size()-1);
			temp.push_back(multi);
		}
		varience.push_back(temp);
	}
	

	return varience;

}

vector<vector<double>> eigen(vector<vector<double>> a, int dim){
	Eigen::MatrixXd m(dim, dim);
	for(int i=0; i<dim; i++){
		for(int k=0; k<dim; k++){
			m(i,k) = a[i][k];
		}

	}
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> sol(m);


	Eigen::VectorXd eigenValues = sol.eigenvalues();
   	Eigen::MatrixXd eigenVectors = sol.eigenvectors();

   	int sum = 0; 
   	for (int i = 0; i<dim; i++){
   		sum+=sol.eigenvalues()(i);
   	}
   	


	cout << "eigne 1 : " << sol.eigenvalues()(dim-1) << " , percentage of coverage: " << sol.eigenvalues()(dim-1)/sum*100 << "%" << endl;
	cout << "eigne 2 : " << sol.eigenvalues()(dim-2) << " , percentage of coverage: " << sol.eigenvalues()(dim-2)/sum*100 << "%" << endl;

	vector<double> temp1;
	vector<double> temp2;
	for (int i=0; i<dim; i++){
		temp1.push_back(eigenVectors(dim-1,i));
		temp2.push_back(eigenVectors(dim-2,i));
	}
	
	vector<vector<double>> maxEig;
	maxEig.push_back(temp1);
	maxEig.push_back(temp2);


    return maxEig;

}


vector<vector<double>> proj(vector<vector<double>> a, vector<vector<double>> b, int dim){
	double multi = 0;
	vector<double> temp;
	vector<vector<double>> proj;

	for(int i=0; i<a.size(); i++){
		temp.clear();
		for(int k=0; k<2; k++){
			multi = 0;
			for(int l=0; l<dim; l++){
				multi += b[k][l]*a[i][l];
			}
			multi = multi/(a.size()-1);
			temp.push_back(multi);
		}
		proj.push_back(temp);
	}

	return proj;

}

int Linear_K(int a, int b){
	//defult
	return 0;
}

vector<vector<double>> RBF_K( vector<vector<double>> a, int dim){
	int d = 2;
	int sum = 0; 
	for(int i=0; i<a[0].size(); i++){
		for(int l=0; l<a.size(); l++){
			sum+=a[l][i];
		}
	}


	vector<double> temp;
	vector<vector<double>> RBF;



	for(int i=0; i<dim; i++){
		temp.clear();
		for(int k=0; k<dim; k++){
			for(int l=0; l<a.size(); l++){
				temp.push_back(exp((-pow(abs(a[l][i]*a[l][k]),2))/(pow(2*sum,2))));
			}
		}
		RBF.push_back(temp);
	}


	return RBF;
}

vector<vector<double>> Poly_K(vector<vector<double>> a, int dim){
	double d = 2;
	vector<double> temp;
	vector<vector<double>> poly;
	for(int i=0; i<dim; i++){
		temp.clear();
		for(int k=0; k<dim; k++){
			for(int l=0; l<a.size(); l++){
				temp.push_back(pow(a[l][i]*a[l][k],d));
			}
		}
		poly.push_back(temp);
	}

	return poly;
}



int main (int argc, char **argv) {

	if (argc==1){
		cout << "Please include csv file in argument" << endl;
		return 0;
	}

	int dim = 256;
	int KPCA = 0;
	string str2, dub2;
	vector<vector<double>> tester;
	vector<int> tester_classes;
	bool first = true; 

	//loads arguments
	if (argc==3){
		dim = stod(argv[2]);
	}else if (argc==4){
		dim = stod(argv[2]);
		KPCA = stod(argv[3]);
	}else if (argc==5){
		dim = stod(argv[2]);
		KPCA = stod(argv[3]);
	}

	//creates train matrix
    ifstream ifs;
    ifs.open (argv[1]);
    string str, dub;                 
    vector<vector<double>> features;
    vector<int> classes;
    vector<vector<double>> fnew;    
    first = true; 
    while (getline (ifs, str)){        
        vector<double> temp;        
        stringstream ss(str);
        first = true;
        while (getline(ss, dub, ',')){  
        	if(first){
        		classes.push_back(stoi(dub));
        		first=false;
        	}else{
        		temp.push_back(stod(dub));
        	}

                  
        }
        features.push_back(temp);             
    }
    cout << "> TRAINING FILE LOADED" << endl;
    vector<vector<double>> covar;
    vector<double> mean;
    //runs PCA
    switch (KPCA){
    	case 0 :
    		cout << "> PCA SELECTED" << endl;
    	    mean = meanV(features);
		    cout << "> MEAN VECTOR CREATED" << endl;

		    covar = covariance(features, mean, dim);
		    cout << "> COVARIENCE MATRIX CREATED" << endl;
    		break;
    	case 1 :
    		cout << "> KPCA RBF SELECTED" << endl;
    		covar = RBF_K(features, dim);
    		cout << "> KERNEL MATRIX CREATED" << endl;
    		break;
    	case 2 :
    		cout << "> KPCA POLY SELECTED" << endl;
    		covar = Poly_K(features, dim);
    		cout << "> KERNEL MATRIX CREATED" << endl;
    		break;
    }

   	vector<vector<double>> projMAP =  eigen(covar, dim);
   	cout << "> EIGENVALUES FOUND" << endl;

   	fnew = proj(features, projMAP, dim);
   	cout << "> PROJECTION COMPLETED" << endl;

	cout << "> TRAINING ENDED" << endl;
   	//saves training outcome
   	int cls = 1;
   	string comm = ",";
   	ofstream outFile("Traing_Output" + to_string(dim) + 'K' + to_string(KPCA) + ".csv");
   	for(int i=0; i<fnew.size(); i++){
   		if(classes[i]==cls){
   			outFile << fnew[i][0] << comm << fnew[i][1] << endl;
   		}
		if(i==fnew.size()-1 && cls<10){
			cls++;
			comm = comm+" ,";
			i=0;
		}
	}   		

	cout << "> NEW CSV TRAINING FILE CREATED" << endl;


	if(argc==5){
		cout << "> TEST PHASE STARTED" << endl;
		ifstream test;
    	test.open (argv[4]);
    	while (getline (test, str2)){        
	        vector<double> temp2;        
	        stringstream ss2(str2);
	        first = true;
	        while (getline(ss2, dub2, ',')){  
	        	if(first){
	        		tester_classes.push_back(stoi(dub2));
	        		first=false;
	        	}else{
	        		temp2.push_back(stod(dub2));
	        	}
	        }
	        tester.push_back(temp2);             
	    }
	    cout << "> TEST FILE LOADED" << endl;
		switch (KPCA){
	    	case 0 :

	    		break;
	    	case 1 :

	    		break;
	    	case 2 :

	    		break;
   		}
   		cout << "> TEST ENDED" << endl;
   		cls = 1;
   		comm = ",";
	   	ofstream testFile("Test_Output" + to_string(dim) + 'K' + to_string(KPCA) + ".csv");
	   	for(int i=0; i<fnew.size(); i++){
	   		if(classes[i]==cls){
	   			//testFile << fnew[i][0] << comm << fnew[i][1] << endl;
	   		}
			if(i==fnew.size()-1 && cls<10){
				cls++;
				comm = comm+" ,";
				i=0;
			}
		}   
		cout << "> NEW CSV TEST FILE CREATED" << endl;		

		
	}

}
