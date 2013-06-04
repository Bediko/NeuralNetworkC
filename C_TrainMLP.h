#ifndef ROOT_TMVA_C_TRAINMLP
#define ROOT_TMVA_C_TRAINMLP


struct CEvents{
  double** eventValues;
  double* eventWeights;
  int* eventClass;
  double ** desired;
};

double***        CTrainMLP(CEvents* ev, double learnRate, int nVars, int nEpochs, 
		           int nEvents, double*** Synweights, double** Neurons,
		           int* NeuronsPerLayer, int NumberOfLayers, int * bias, 
			   double decayRate, double max, double min);
 
double***      CTrainMLP_b(CEvents* ev, double learnRate, int nVars, int nEpochs, 
		           int nEvents, double*** Synweights, double** Neurons,
		           int* NeuronsPerLayer, int NumberOfLayers, int * bias, 
			   double decayRate, double max, double min);
 
void CTrainMLP_testing(CEvents* ev, int nEpochs, int nEvents, double*** Synweights, 
			   double** Neurons, int* NeuronsPerLayer, int NumberOfLayers, 
			   int * bias, double** testout);

#endif
