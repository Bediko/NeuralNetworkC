#include "C_TrainMLP.h"
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <sstream>
#include <iomanip>
using namespace std;
#include <CL/cl.h>


#include <sys/time.h>
#include <unistd.h>

double second()
{
    struct timeval tm;
    double t ;
    gettimeofday(&tm, NULL);
    t = (double) (tm.tv_sec) + ((double) (tm.tv_usec)) / 1.0e6;
    return t;
}

/**
 ******************************************************************************
 * main function
 ******************************************************************************/

int main(int argc, char *argv[])
{

    ifstream loaddata, loadtest;
    ofstream savedata;
    double learnrate, decayrate;
    int nVars, nEpochs, nEvents, NumberOfLayers, i, j, l;
    int *NeuronsPerLayer;
    double *** Synweights;
    double **Neurons;
    int *bias;
    double **mean;
    double **varianz;
    double **testoutput;
    int *classevents;
    int nclasses;
    int *** bins;
    double binsize = 0.05;
    double *** histdata;
    CEvents *training = (CEvents *)malloc(sizeof(CEvents));;
    CEvents *testing  = (CEvents *)malloc(sizeof(CEvents));;
    int type, events;
    double start_t, stop_t, duration;
    int *eventClass;
    double *eventWeights;
    double **eventValues;

    if (argc <= 1) {
        cout << "Kein Ordner angegeben" << endl;
    }

    // read simuation parameter from argv[1]+"/numbers.txt"
    cout << "Loading data" << endl;
    string folder = argv[1];
    string filename = folder + "/numbers.txt";
    loaddata.open(filename.c_str());
    if (!loaddata.is_open()) {
        cout << folder + "/numbers.txt kann nicht gefunden werden" << endl;
    }
    loaddata >> learnrate;
    cout << "learning rate: " << learnrate << endl;
    loaddata >> nVars;
    cout << "number of variables: " << nVars << endl;
    loaddata >> nEpochs;
    cout << "number of learning epochs: " << nEpochs << endl;
    loaddata >> nEvents;
    cout << "number of events: " << nEvents << endl;
    loaddata >> NumberOfLayers;
    NeuronsPerLayer = (int *)malloc(NumberOfLayers * sizeof(int));
    for (i = 0; i < NumberOfLayers; i++) {
        loaddata >> NeuronsPerLayer[i];
        cout << "number of nodes on layer " << i << " is "
             << NeuronsPerLayer[i] << endl;
    }
    loaddata >> decayrate;
    cout << "decay rate of the learning rate: " << decayrate << endl;
    loaddata >> type;
    cout << "learning type: " << type << endl;
    if (type == 3) {
        loaddata >> events;
        cout << "learning in batches of " << events << " events" << endl;
    }
    loaddata.close();

    // read trainig and testing data from argv[1]+"/training.txt
    // and argv[1]+"/testing.txt"
    filename = folder + "/training.txt";
    loaddata.open(filename.c_str());
    if (!loaddata.is_open()) {
        cout << folder + "/training.txt kann nicht gefunden werden" << endl;
    }
    filename = folder + "/testing.txt";
    loadtest.open(filename.c_str());
    if (!loadtest.is_open()) {
        cout << folder + "/testing.txt kann nicht gefunden werden" << endl;
    }

    training->eventClass  =  (int *)  malloc(nEvents * sizeof(int));
    training->eventWeights = (double *) malloc(nEvents * sizeof(double));
    training->eventValues = (double **)malloc(nEvents * sizeof(double *));

    eventClass  =  (int *)  malloc(nEvents * sizeof(int));
    eventWeights = (double *) malloc(nEvents * sizeof(double));
    eventValues = (double **)malloc(nEvents * sizeof(double *));

    testing->eventClass   =  (int *)  malloc(nEvents * sizeof(int));
    testing->eventWeights = (double *) malloc(nEvents * sizeof(double));
    testing->eventValues  = (double **)malloc(nEvents * sizeof(double *));

    training->eventValues[0] = (double *)malloc(nEvents * nVars * sizeof(double));
    testing->eventValues[0] = (double *)malloc(nEvents * nVars * sizeof(double));
    eventValues[0] = (double *)malloc(nEvents * nVars * sizeof(double));
    for (i = 1; i < nEvents; i++) {
        training->eventValues[i] = training->eventValues[0] + i * nVars;
        eventValues[i] = training->eventValues[0] + i * nVars;
        testing->eventValues[i] = testing->eventValues[0] + i * nVars;
    }

    for (i = 0; i < nEvents; i++) {
        loaddata >> training->eventClass[i];
        loadtest >> testing->eventClass[i];
        loaddata >> training->eventWeights[i];
        loadtest >> testing->eventWeights[i];
        for (j = 0; j < nVars; j++) {
            loaddata >> training->eventValues[i][j];
            loadtest >> testing->eventValues[i][j];
        }
    }
    loaddata.close();
    loadtest.close();

    // allocate neurons
    int totalNeurons = NeuronsPerLayer[0];
    for (l = 1; l < NumberOfLayers; l++) {
        totalNeurons += NeuronsPerLayer[l];
    }
    Neurons    = (double **)malloc(NumberOfLayers * sizeof(double *));
    Neurons[0] = (double *) malloc(totalNeurons * sizeof(double));
    for (l = 1; l < NumberOfLayers; l++) {
        Neurons[l] = Neurons[l - 1] + NeuronsPerLayer[l - 1];
    }

    // set bias neurons
    bias = (int *)malloc(NumberOfLayers * sizeof(int));
    for (i = 0; i < NumberOfLayers - 1; i++) {
        bias[i] = 1;       // gives the number of bias nodes for each layer
    }
    bias[NumberOfLayers - 1] = 0;

    // read initial synapses values from argv[1]+"/synapses.txt
    filename = folder + "/synapses.txt";
    loaddata.open(filename.c_str());
    if (!loaddata.is_open()) {
        cout << folder + "/synapses.txt kann nicht gefunden werden" << endl;
    }

    //int totalNeurons = NeuronsPerLayer[0];
    //for(l=1;l<NumberOfLayers;l++){
    //  totalNeurons +=NeuronsPerLayer[l];
    //}

    Synweights = (double ** *)malloc((NumberOfLayers - 1) * sizeof(double **));
    for (l = 0; l < NumberOfLayers - 1; l++) {
        Synweights[l] = (double **)malloc(NeuronsPerLayer[l] * sizeof(double *));
        for (i = 0; i < NeuronsPerLayer[l]; i++) {
            Synweights[l][i] = (double *)malloc((NeuronsPerLayer[l + 1] - bias[l + 1]) * sizeof(double));
        }
    }
    for (l = 0; l < NumberOfLayers - 1; l++) {
        for (i = 0; i < NeuronsPerLayer[l]; i++) {
            for (j = 0; j < NeuronsPerLayer[l + 1] - bias[l + 1]; j++) {
                loaddata >> Synweights[l][i][j];
            }
        }
    }
    for (l = 0; l < NumberOfLayers - 1; l++) {
        cout << "number of synapses between " << l << " and " << l + 1 << ": "
             << NeuronsPerLayer[l] << "X" << NeuronsPerLayer[l + 1] - bias[l + 1] << endl;

    }
    // all data read

    // now train network, CTrainMLP for online, CTrainMLP_b for learning in one batch,
    // CTrainMLP_m for learning in many batches
    cout << "Train network" << endl;

    start_t = second();
    if (type == 1) {
        cout << "online learning " << endl;
        CTrainMLP(training, learnrate, nVars, nEpochs,
                  nEvents, Synweights, NeuronsPerLayer,
                  NumberOfLayers, bias, decayrate, 1.0, 0.0);
    }
    if (type == 2) {
        cout << "batch learning" << endl;
        CTrainMLP_b(training, learnrate, nVars, nEpochs,
                    nEvents, Synweights, NeuronsPerLayer,
                    NumberOfLayers, bias, decayrate);
    }
    if (type == 3) {
        if (nEvents % events != 0) {
            cout << events << " events is not a factor of " << nEvents
                 << ". Choose different value" << endl;
            exit(1);
        }
        cout << "learning " << nEvents << " events in batches of " << events
             << " events" << endl;
        if (NumberOfLayers == 4)
            CTrainMLP_4(training, Synweights, learnrate, decayrate,
                        nVars, nEpochs, nEvents, NumberOfLayers,
                        NeuronsPerLayer, bias, events);
        else
            CTrainMLP_m(training, Synweights, learnrate, decayrate,
                        nVars, nEpochs, nEvents, NumberOfLayers,
                        NeuronsPerLayer, bias, events);
    }

    if (type == 4) {
        cl_context context = 0;
        cl_command_queue commandQueue = 0;
        cl_program program = 0;
        cl_device_id device = 0;
        cl_kernel kernel = 0;
        cl_mem memObjects[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        cl_int errNum;

        float *desired;  // desired output value for each output neuron.
        desired = (float *) malloc(NeuronsPerLayer[NumberOfLayers - 1] * sizeof(float));

        float **deltas;  // deltas, used in back propagation, index1: Layer index2: Neuron
        deltas = (float **) malloc(NumberOfLayers * sizeof(float *));
        for (l = 0; l < NumberOfLayers; l++) {
            deltas[l] = (float *)malloc(NeuronsPerLayer[l] * sizeof(float));
        }

        context = CTrainMLP_CreateContext();
        if (context == NULL) {
            cout << "unable to create context. Abort." << endl;
            return 1;
        }
        commandQueue = CTrainMLP_CreateCommandQueue(context, &device);
        if (commandQueue == NULL) {
            cout << "Unable to create command queue. Abort." << endl;
            return 1;
        }
        program = CTrainMLP_CreateProgram(context, device, "kernelonline.cl");
        cout << "batch learning on GPU" << endl;

    }


    duration = second() - start_t;

    // test network
    testoutput = (double **)malloc(nEvents * sizeof(double *));
    for (i = 0; i < nEvents; i++) {
        testoutput[i] = (double *)malloc(NeuronsPerLayer[NumberOfLayers - 1] * sizeof(double));
    }

    // output results of test events
    cout << "Test network" << endl;

    start_t = second();
    CTrainMLP_testing(testing, nEpochs, nEvents, Synweights, Neurons,
                      NeuronsPerLayer, NumberOfLayers, bias, testoutput);
    stop_t = second();
    cout << "time for learning: " << duration << endl;
    cout << "time for testing:  " << stop_t - start_t << endl;

    // write results of final synapses
    filename = folder + "/synapses_out.txt";
    savedata.open(filename.c_str());
    if (!savedata.is_open()) {
        cout << folder + "/synapses_out.txt kann nicht gefunden werden" << endl;
    }
    savedata << setprecision(8) << scientific;
    for (l = 0; l < NumberOfLayers - 1; l++) {
        for (i = 0; i < NeuronsPerLayer[l]; i++) {
            for (j = 0; j < NeuronsPerLayer[l + 1] - bias[l + 1]; j++) {
                savedata << Synweights[l][i][j] << endl;
            }
        }
    }
    savedata.close();

    // write results of the output layer
    filename = folder + "/Outputvalues.txt";
    savedata.open(filename.c_str());
    if (!savedata.is_open())
        cout << folder + "/Outputvalues.txt kann nicht gefunden werden" << endl;
    else
        cout << "Write Outputvalues in " << filename << endl;
    for (i = 0; i < nEvents; i++) {
        for (j = 0; j < NeuronsPerLayer[NumberOfLayers - 1]; j++) {
            savedata << setprecision(8) << scientific << testoutput[i][j] << endl;;
        }
    }
    savedata.close();

    // finally anaylse output values of the network for the test data
    if (NeuronsPerLayer[NumberOfLayers - 1] == 1) {
        classevents = (int *)calloc(2, sizeof(int));
        nclasses = 2;
        mean = (double **)calloc(2, sizeof(double *));
        mean[0] = (double *)calloc(1, sizeof(double));
        mean[1] = (double *)calloc(1, sizeof(double));
        varianz = (double **)calloc(2, sizeof(double *));
        varianz[0] = (double *)calloc(1, sizeof(double));
        varianz[1] = (double *)calloc(1, sizeof(double));
    } else {
        nclasses = NeuronsPerLayer[NumberOfLayers - 1];
        classevents = (int *)calloc(NeuronsPerLayer[NumberOfLayers - 1], sizeof(int));
        mean = (double **)calloc(NeuronsPerLayer[NumberOfLayers - 1], sizeof(double *));
        for (i = 0; i < NeuronsPerLayer[NumberOfLayers - 1]; i++) {
            mean[i] = (double *)calloc(NeuronsPerLayer[NumberOfLayers - 1], sizeof(double));
        }
        varianz = (double **)calloc(NeuronsPerLayer[NumberOfLayers - 1], sizeof(double *));
        for (i = 0; i < NeuronsPerLayer[NumberOfLayers - 1]; i++) {
            varianz[i] = (double *)calloc(NeuronsPerLayer[NumberOfLayers - 1], sizeof(double));
        }
    }

    for (i = 0; i < nEvents; i++) {
        classevents[testing->eventClass[i]] += 1;
        for (j = 0; j < NeuronsPerLayer[NumberOfLayers - 1]; j++) {
            mean[testing->eventClass[i]][j] += testoutput[i][j];
        }
    }
    cout << "Mittelwert" << endl;
    for (i = 0; i < nclasses; i++) {
        cout << "Event" << i;
        for (j = 0; j < NeuronsPerLayer[NumberOfLayers - 1]; j++) {
            mean[i][j] = mean[i][j] / classevents[i];
            cout << "\t" << fixed << mean[i][j];
        }
        cout << endl;
    }
    cout << "Varianz" << endl;
    for (i = 0; i < nEvents; i++) {
        for (j = 0; j < NeuronsPerLayer[NumberOfLayers - 1]; j++) {
            varianz[testing->eventClass[i]][j] += pow(testoutput[i][j] - mean[testing->eventClass[i]][j], 2);
        }
    }
    for (i = 0; i < nclasses; i++) {
        cout << "Event" << i;
        for (j = 0; j < NeuronsPerLayer[NumberOfLayers - 1]; j++) {
            varianz[i][j] = varianz[i][j] / classevents[i];
            cout << "\t" << fixed << varianz[i][j];
        }
        cout << endl;
    }

    // historgramme !!!
    int lastLayer   = NumberOfLayers - 1;
    int lastNeurons = NeuronsPerLayer[lastLayer];

    histdata = (double ** *)malloc(nclasses * sizeof(double **));
    for (i = 0; i < nclasses; i++) {
        histdata[i] = (double **)malloc(classevents[i] * sizeof(double *));
        for (j = 0; j < classevents[i]; j++) {
            histdata[i][j] = (double *)malloc(lastNeurons * sizeof(double));
        }
    }
    for (i = 0; i < nclasses; i++) {
        classevents[i] = 0;
    }
    for (i = 0; i < nEvents; i++) {
        int m = testing->eventClass[i];
        for (j = 0; j < lastNeurons; j++) {
            //cout<<"i:"<<i<<" j:"<<j<<" m:"<<m<<" classevent:"<<classevents[m]<<endl;
            histdata[m][classevents[m]][j] = testoutput[i][j];
        }
        classevents[m]++;
    }

    double *max = (double *)calloc(lastNeurons, sizeof(double));
    double *min = (double *)calloc(lastNeurons, sizeof(double));
    int nBins = 0;
    double minhisto = 100;
    for (int i = 0; i < lastNeurons; i++) {
        min[i] = 100.0;
        max[i] = -100.0;
    }
    for (i = 0; i < nclasses; i++) {
        for (j = 0; j < classevents[i]; j++) {
            for (int k = 0; k < lastNeurons; k++) {
                if (histdata[i][j][k] < min[k])
                    min[k] = histdata[i][j][k];
                if (histdata[i][j][k] > max[k])
                    max[k] = histdata[i][j][k];
            }
        }
        double gmax = -100;
        double gmin = 100;
        for (int k = 0; k < lastNeurons; k++) {
            if (min[k] < gmin)
                gmin = min[k];
            if (max[k] > gmax)
                gmax = max[k];
        }
        for (int k = 0; k < lastNeurons; k++) {
            int tmp = (int)(((-gmin - 0.025) / 0.05) + 1 + (gmax + 0.025) / 0.05);
            if (nBins < tmp)
                nBins = tmp;

            double tmp2 = (int)((gmin - 0.025) / 0.05) * 0.05 - 0.025;
            if (minhisto > tmp2)
                minhisto = tmp2;
        }
    }
    //nBins++;
    bins = (int ** *)malloc(nclasses * sizeof(int **));
    for (i = 0; i < nclasses; i++) {
        bins[i] = (int **)malloc(lastNeurons * sizeof(int *));
        for (j = 0; j < lastNeurons; j++) {
            bins[i][j] = (int *)malloc((nBins + 1) * sizeof(int));
            for (int k = 0; k <= nBins; k++) {
                bins[i][j][k] = 0;
            }
        }
    }
    for (i = 0; i < nclasses; i++) {
        for (j = 0; j < classevents[i]; j++) {
            for (int k = 0; k < lastNeurons; k++) {
                int m;
                m = (int)((histdata[i][j][k] - minhisto) / 0.05);
                if (m > nBins || m < 0) {
                    cout << "fehler: " << m << endl;
                    continue;
                }
                bins[i][k][m]++;
            }
        }
    }
    for (i = 0; i < nclasses; i++) {
        ostringstream convert;
        convert << i;
        filename = folder + "/Class" + convert.str();
        savedata.open(filename.c_str());
        for (j = 0; j <= nBins; j++) {
            savedata << setprecision(4) << fixed << (minhisto + 0.025) + 0.05 * j;
            for (int k = 0; k < lastNeurons; k++) {
                savedata << "\t" << setw(6) << bins[i][k][j];
            }
            savedata << endl;
        }
        savedata.close();
    }
    return 0;
}
/*



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
    for(j=0;j<=nBins;j++){
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
*/
