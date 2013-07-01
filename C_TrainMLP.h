#ifndef ROOT_TMVA_C_TRAINMLP
#define ROOT_TMVA_C_TRAINMLP
#ifdef BLAS
#include <cblas.h>
#endif
#include <CL/cl.h>


struct CEvents {
    double **eventValues;
    double *eventWeights;
    int *eventClass;
};

void CTrainMLP  (CEvents *ev, double learnRate, int nVars, int nEpochs,
                 int nEvents, double *** Synweights,
                 int *NeuronsPerLayer, int NumberOfLayers, int *bias,
                 double decayRate, double max, double min);

void CTrainMLP_b(CEvents *ev, double learnRate, int nVars, int nEpochs,
                 int nEvents, double *** Synweights,
                 int *NeuronsPerLayer, int NumberOfLayers, int *bias,
                 double decayRate);

void CTrainMLP_m(CEvents *ev,  double *** Synweights, double learnRate, double decayRate,
                 int nVars, int nEpochs, int nEvents, int NumberOfLayers,
                 int *NeuronsPerLayer, int *bias, int events);

void CTrainMLP_testing(CEvents *ev, int nEpochs, int nEvents,
                       double *** Synweights, double **Neurons,
                       int *NeuronsPerLayer, int NumberOfLayers,
                       int *bias, double **testout);

void CTrainMLP_4(CEvents *ev,  double *** Synweights, double learnRate, double decayRate,
                 int nVars, int nEpochs, int nEvents, int NumberOfLayers,
                 int *NeuronsPerLayer, int *bias, int events);

void check_error(int error);

cl_context CTrainMLP_CreateContext();

cl_command_queue CTrainMLP_CreateCommandQueue(cl_context context, cl_device_id *device);

cl_program CTrainMLP_CreateProgram(cl_context context, cl_device_id device, const char *fileName);


#endif
