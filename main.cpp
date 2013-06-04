#include "C_TrainMLP.h"
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <sstream>
#include <iomanip>
using namespace std;



int main(int argc, char *argv[]){
	ifstream loaddata,loadtest;
	ofstream savedata;
	double learnrate, decayrate;
	int nVars,nEpochs,nEvents,NumberOfLayers,i,j,l;
	int* NeuronsPerLayer;
	double*** Synweights;
	double** Neurons;
	int* bias;
	double** mean;
	double** varianz;
	double** testoutput;
	int* classevents;
	int nclasses;
	int nBins=20;
	int*** bins;
	double binsize=1.0/nBins;
	double*** histdata;
	CEvents* training=(CEvents*)malloc(sizeof(CEvents));;
	CEvents* testing=(CEvents*)malloc(sizeof(CEvents));;
	if (argc <=1){
		cout<<"Kein Ordner angegeben"<<endl;
	}
	cout<<"Loading data"<<endl;
	string folder=argv[1];
	string filename=folder+"/numbers.txt";
	loaddata.open(filename.c_str());
	if(!loaddata.is_open()){
		cout<<folder+"/numbers.txt kann nicht gefunden werden"<<endl;
	}
	loaddata>>learnrate;
	loaddata>>nVars;
	loaddata>>nEpochs;
	loaddata>>nEvents;
	loaddata>>NumberOfLayers;
	NeuronsPerLayer=(int*)malloc(NumberOfLayers*sizeof(int));
	for(i=0;i<NumberOfLayers;i++){
    loaddata>>NeuronsPerLayer[i];    
  }
  loaddata>>decayrate;
  loaddata.close();
  
  filename=folder+"/training.txt";
  loaddata.open(filename.c_str());
	if(!loaddata.is_open()){
		cout<<folder+"/training.txt kann nicht gefunden werden"<<endl;
	}
	filename=folder+"/testing.txt";
	loadtest.open(filename.c_str());
	if(!loadtest.is_open()){
		cout<<folder+"/testing.txt kann nicht gefunden werden"<<endl;
	}

	training->eventClass=(int*)malloc(nEvents*sizeof(int));
	training->eventWeights=(double*)malloc(nEvents*sizeof(double));
	training->eventValues=(double**)malloc(nEvents*sizeof(double*));
	testing->eventClass=(int*)malloc(nEvents*sizeof(int));
	testing->eventWeights=(double*)malloc(nEvents*sizeof(double));
	testing->eventValues=(double**)malloc(nEvents*sizeof(double*));

	for(i=0;i<nEvents;i++){
		training->eventValues[i]=(double*)malloc(nVars*sizeof(double));
		testing->eventValues[i]=(double*)malloc(nVars*sizeof(double));
		loaddata>>training->eventClass[i];
		loadtest>>testing->eventClass[i];
		loaddata>>training->eventWeights[i];
		loadtest>>testing->eventWeights[i];
		for(j=0;j<nVars;j++){
			loaddata>>training->eventValues[i][j];
			loadtest>>testing->eventValues[i][j];
		}
	}
	loaddata.close();
	loadtest.close();
	Neurons=(double**)malloc(NumberOfLayers*sizeof(double*));
	for(i=0;i<NumberOfLayers;i++){
		Neurons[i]=(double*)malloc(NeuronsPerLayer[i]*sizeof(double));
	}

	bias=(int*)calloc(NumberOfLayers,sizeof(int));
	for(i=0;i<NumberOfLayers-1;i++){
		bias[i]=1;
	}

	filename=folder+"/synapses.txt";
	loaddata.open(filename.c_str());
	if(!loaddata.is_open()){
		cout<<folder+"/synapses.txt kann nicht gefunden werden"<<endl;
	}
	Synweights=(double***)malloc((NumberOfLayers-1)*sizeof(double**));
	for(l=0;l<NumberOfLayers-1;l++){
		Synweights[l]=(double**)malloc(NeuronsPerLayer[l]*sizeof(double*));
		for(i=0;i<NeuronsPerLayer[l];i++){
			Synweights[l][i]=(double*)malloc((NeuronsPerLayer[l+1]-bias[l+1])*sizeof(double));
			for(j=0;j<NeuronsPerLayer[l+1]-bias[l+1];j++){
				loaddata>>Synweights[l][i][j];
			}
		}
	}
	cout<<"Train network"<<endl;
	Synweights=CTrainMLP(training, learnrate,nVars,nEpochs, 
		nEvents,Synweights,Neurons,NeuronsPerLayer,NumberOfLayers,decayrate,1.0,0.0);
	testoutput=(double**)malloc(nEvents*sizeof(double*));
	for(i=0;i<nEvents;i++){
		testoutput[i]=(double*)malloc(NeuronsPerLayer[NumberOfLayers-1]*sizeof(double));
	}
	cout<<"Test network"<<endl;
	testoutput=CTrainMLP_testing(testing,nEpochs,nEvents,Synweights,Neurons,NeuronsPerLayer,NumberOfLayers,testoutput);
	filename=folder+"/Outputvalues.txt";
	cout<<"Write Outputvalues in "<<filename<<endl;
	savedata.open(filename.c_str());
	for(i=0;i<nEvents;i++){
		for(j=0;j<NeuronsPerLayer[NumberOfLayers-1];j++){
			savedata<<setprecision(8)<<scientific<<testoutput[i][j]<<" ";
		}
		savedata<<endl;
	}
	savedata.close();
	if(NeuronsPerLayer[NumberOfLayers-1]==1){
		classevents=(int*)calloc(2,sizeof(int));
		nclasses=2;
		mean=(double**)calloc(2,sizeof(double*));
		mean[0]=(double*)calloc(1,sizeof(double));
		mean[1]=(double*)calloc(1,sizeof(double));
		varianz=(double**)calloc(2,sizeof(double*));
		varianz[0]=(double*)calloc(1,sizeof(double));
		varianz[1]=(double*)calloc(1,sizeof(double));
	}
	else{
		nclasses=NeuronsPerLayer[NumberOfLayers-1];
		classevents=(int*)calloc(NeuronsPerLayer[NumberOfLayers-1],sizeof(int));
		mean=(double**)calloc(NeuronsPerLayer[NumberOfLayers-1],sizeof(double*));
		for(i=0;i<NeuronsPerLayer[NumberOfLayers-1];i++){
			mean[i]=(double*)calloc(NeuronsPerLayer[NumberOfLayers-1],sizeof(double));
		}
		varianz=(double**)calloc(NeuronsPerLayer[NumberOfLayers-1],sizeof(double*));
		for(i=0;i<NeuronsPerLayer[NumberOfLayers-1];i++){
			varianz[i]=(double*)calloc(NeuronsPerLayer[NumberOfLayers-1],sizeof(double));
		}
	}

	for(i=0;i<nEvents;i++){
		classevents[testing->eventClass[i]]+=1;
		for(j=0;j<NeuronsPerLayer[NumberOfLayers-1];j++){
			mean[testing->eventClass[i]][j]+=testoutput[i][j];
		}
	}
	cout<<"Mittelwert"<<endl;
	for(i=0;i<nclasses;i++){
		cout<<"Event"<<i;
		for(j=0;j<NeuronsPerLayer[NumberOfLayers-1];j++){
			mean[i][j]=mean[i][j]/classevents[i];
			cout<<"\t"<<fixed<<mean[i][j];
		}
		cout<<endl;
	}
	cout<<"Varianz"<<endl;
	for(i=0;i<nEvents;i++){
		for(j=0;j<NeuronsPerLayer[NumberOfLayers-1];j++){
			varianz[testing->eventClass[i]][j]+=pow(testoutput[i][j]-mean[testing->eventClass[i]][j],2);
		}
	}
	for(i=0;i<nclasses;i++){
		cout<<"Event"<<i;
		for(j=0;j<NeuronsPerLayer[NumberOfLayers-1];j++){
			varianz[i][j]=varianz[i][j]/classevents[i];
			cout<<"\t"<<fixed<<varianz[i][j];
		}
		cout<<endl;
	}

	histdata=(double***)malloc(nclasses*sizeof(double**));
	for(i=0;i<nclasses;i++){
		histdata[i]=(double**)malloc(classevents[i]*sizeof(double*));
		for(j=0;j<classevents[i];j++){
			histdata[i][j]=(double*)malloc(NeuronsPerLayer[NumberOfLayers-1]*sizeof(double));
		}
	}
	for(i=0;i<nclasses;i++){
		classevents[i]=0;
	}
	for(i=0;i<nEvents;i++){
		int m=testing->eventClass[i];
		for(j=0;j<NeuronsPerLayer[NumberOfLayers-1];j++){
			//cout<<"i:"<<i<<" j:"<<j<<" m:"<<m<<" classevent:"<<classevents[m]<<endl;
			histdata[m][classevents[m]][j]=testoutput[i][j];
		}
		classevents[m]++;
	}
	bins=(int***)calloc(nclasses,sizeof(int**));
	for(i=0;i<nclasses;i++){
		bins[i]=(int**)calloc(NeuronsPerLayer[NumberOfLayers-1],sizeof(int*));
		for(j=0;j<NeuronsPerLayer[NumberOfLayers-1];j++){
			bins[i][j]=(int*)calloc(nBins,sizeof(int));
		}
	}

	double* max=(double*)calloc(NeuronsPerLayer[NumberOfLayers-1],sizeof(double));
	double* min=(double*)calloc(NeuronsPerLayer[NumberOfLayers-1],sizeof(double));
	for(int i=0;i<NeuronsPerLayer[NumberOfLayers-1];i++){
		min[i]=1;
	}
	for(i=0;i<nclasses;i++){
		for(j=0;j<classevents[i];j++){
			for(int k=0;k<NeuronsPerLayer[NumberOfLayers-1];k++){
				if(histdata[i][j][k]<min[k])
					min[k]=histdata[i][j][k];
				if(histdata[i][j][k]>max[k])
					max[k]=histdata[i][j][k];
			}
		}
		for(j=0;j<classevents[i];j++){
			for(int k=0;k<NeuronsPerLayer[NumberOfLayers-1];k++){
				int m;
				double interval = (double)(max[k] - min[k] ) / nBins;
				m=(int)((histdata[i][j][k]- min[k])/interval);
				bins[i][k][m]++;
			}
		}
		ostringstream convert;
			convert<<i;
		filename=folder+"/Class"+convert.str();
			savedata.open(filename.c_str());
		for(j=0;j<nBins;j++){
			savedata<<setprecision(4)<<fixed<<(binsize*j+binsize*(j+1))/2;
			for(int k=0;k<NeuronsPerLayer[NumberOfLayers-1];k++){
				savedata<<"\t"<<setw(6)<<bins[i][k][j];
			}
			savedata<<endl;
		}
		savedata.close();
	}
	return 0;
}