#ifndef ROOT_TMVA_C_TRAINMLP
#define ROOT_TMVA_C_TRAINMLP



	struct CEvents{
		double** eventValues;
		double* eventWeights;
		int* eventClass;
	};
	double*** CTrainMLP(CEvents* ev, double learnRate, int nVars, int nEpochs, 
		int nEvents, double*** Synweights, double** Neurons,int* NeuronsPerLayer, int NumberOfLayers,double decayRate,double max,double min);

	double** CTrainMLP_testing(CEvents* ev, int nEpochs, 
		int nEvents, double*** Synweights, double** Neurons,int* NeuronsPerLayer, int NumberOfLayers, double** testout);

#endif
