#ifndef ROOT_TMVA_C_TRAINMLP
#define ROOT_TMVA_C_TRAINMLP


struct CEvents{
  double** eventValues;
  double* eventWeights;
  int* eventClass;
};

void             CTrainMLP  (CEvents* ev, double learnRate, int nVars, int nEpochs, 
			     int nEvents, double*** Synweights,
			     int* NeuronsPerLayer, int NumberOfLayers, int * bias, 
			     double decayRate, double max, double min);
 
void             CTrainMLP_b(CEvents* ev, double learnRate, int nVars, int nEpochs, 
			     int nEvents, double*** Synweights,
			     int* NeuronsPerLayer, int NumberOfLayers, int * bias, 
			     double decayRate);

void             CTrainMLP_m(CEvents* ev, double learnRate, int nVars, int nEpochs, 
			     int nEvents, double*** Synweights,
			     int* NeuronsPerLayer, int NumberOfLayers, int * bias, 
			     double decayRate, int events);
 
void CTrainMLP_testing(CEvents* ev, int nEpochs, int nEvents, double*** Synweights, 
			     double** Neurons, int* NeuronsPerLayer, int NumberOfLayers, 
			      int * bias, double** testout);

#endif
