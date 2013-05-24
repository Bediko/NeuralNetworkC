#include "C_TrainMLP.h"
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>

using namespace std;

int main(int argc, char *argv[]){
	ifstream loaddata,loadtest;
	double learnrate, decayrate;
	int nVars,nEpochs,nEvents,NumberOfLayers,i,j;
	int* NeuronsPerLayer;
	CEvents* training=(CEvents*)malloc(sizeof(CEvents));;
	CEvents* testing=(CEvents*)malloc(sizeof(CEvents));;
	if (argc <=1){
		cout<<"Kein Ordner angegeben"<<endl;
	}

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
	cout<<testing->eventClass[nEvents-1]<<endl;
	cout<<testing->eventWeights[nEvents-1]<<endl;
	for(j=0;j<nVars;j++){
			cout<<testing->eventValues[nEvents-1][j]<<endl;
		}
	return 0;
}