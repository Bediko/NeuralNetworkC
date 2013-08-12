#include "C_TrainMLP.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <CL/cl.h>
#include <sstream>
#include <string>

#define VALUEPLUS 1
#define VALUEMINUS 0



#define FUNCTION  tanh     //function
#define DERIVATE tanhd     //derivate of function

using namespace std;

double tanhd(double value)
{
    return 1.0 - value * value;
}

/**
 ******************************************************************************
 * function
 * void CTrainMLP(CEvents* ev, double learnRate, int nVars, int nEpochs,
            int nEvents, double*** Synweights, double** Neurons,
            int* NeuronsPerLayer, int NumberOfLayers,  int * bias,
            double decayRate, double max, double min)
 *
 *Trains the neural network with online back propagation learning
 *
 *@param[in] ev               Structure with data of the events for training.
                              Includes the class, the weight and the values of the events
 *@param[in] learnRate        The learning rate for the neural network.
 *@param[in] nVars            Number of Inputvariables (without Bias Node!)
 *@param[in] nEpochs          Number of Epochs to train.
 *@param[in] nEvents          Number of Events to train.
 *@param[in,out] Synweights:  3-Dimensional array for Synapseweights.
 *                    Index 1 is the layer starting at the weights between
 *                                 the inputlayer and the first hiddenlayer.
 *                Index 2 is the neuron of the lower layer.
 *                Index 3 is the neuron of the upper layer.
 *@param[in] Neurons:         Matrix for the outputvalue of each neuron.
 *                Index 1 is the layer.
 *                Index 2 is the neuron on the layer.
 *@param[in] NeuronsPerLayer: Array that contains the number of neurons per layer.
 *@param[in] NumberOfLayers:  Number of Layers in the neural network.
 *@param[in] bias:            Number of bias nodes per layer
 *@param[in] decayRat:        Parameter for changing the learing rate
 *@param[in] max:             set to 1.0, used for activation function on the last layer
 *@param[in] min:             set to 0.0, used for activation function on the last layer
 *
 ******************************************************************************/
void CTrainMLP(CEvents *ev, double learnRate, int nVars, int nEpochs,
               int nEvents, double *** Synweights,
               int *NeuronsPerLayer, int NumberOfLayers, int *bias,
               double decayRate, double max, double min)
{

    int l, i, j, k;       //indices in for loops
    int lateEpochs = (int)(nEpochs * 0.95) - 1; // taken from TMVA for better learning

    double sumdeltas = 0.0; // Sum of delta_k*weight_jk used in back propagation

    //Create Vector of desired Values for back propagation
    double *desired;  // desired output value for each output neuron.
    desired = (double *) malloc(NeuronsPerLayer[NumberOfLayers - 1] * sizeof(double));

    double **deltas;  // deltas, used in back propagation, index1: Layer index2: Neuron
    deltas = (double **) malloc(NumberOfLayers * sizeof(double *));
    for (l = 0; l < NumberOfLayers; l++) {
        deltas[l] = (double *)malloc(NeuronsPerLayer[l] * sizeof(double));
    }

    cout << "function CTrainMLP" << endl;

    // allocate neurons
    double **Neurons;
    int totalNeurons = NeuronsPerLayer[0];
    for (l = 1; l < NumberOfLayers; l++) {
        totalNeurons += NeuronsPerLayer[l];
    }
    Neurons = (double **)malloc(NumberOfLayers * sizeof(double *));
    Neurons[0] = (double *) malloc(totalNeurons * sizeof(double));
    for (l = 1; l < NumberOfLayers; l++) {
        Neurons[l] = Neurons[l - 1] + NeuronsPerLayer[l - 1];
    }

    // set neurons of bias nodes
    for (i = 0; i < NumberOfLayers - 1; i++) {
        if (bias[i] != 0) {
            Neurons[i][NeuronsPerLayer[i] - 1] = 1.0;
        }
    }

    // online learning
    for (int nEp = 0; nEp < nEpochs; nEp++) {     //for each epoch

        for (int nEv = 0; nEv < nEvents; nEv++) {   //for each event

            //Initialization for each event
            for (i = 0; i < NeuronsPerLayer[NumberOfLayers - 1]; i++) {
                //Set desired output to 1 if eventclass is the same
                // as the index of the output neuron
                if (ev->eventClass[nEv] == i) {
                    desired[i] = 1.0;
                } else {
                    desired[i] = 0.0;
                }
            }

            // aus eventValue bias-Knoten wieder raus und for-loop nur
            // bis < NeuronsPerLayer[0]-1
            for (i = 0; i < NeuronsPerLayer[0] - bias[0]; i++) {
                //Use Eventvalues as output of the input layer to make
                //the next step easier.
                Neurons[0][i] = ev->eventValues[nEv][i];
            }

            //forward propagation

            //For each layer except Input  Layer.
            for (l = 1; l < NumberOfLayers; l++) {

                // For each neuron on the next layer except bias node
                for (j = 0; j < NeuronsPerLayer[l] - bias[l]; j++) {

                    // Calculate the neuron input
                    Neurons[l][j] = 0.0;
                    // For each input coming from the lower layer
                    for (i = 0; i < NeuronsPerLayer[l - 1]; i++) {
                        // Calculate the neuron input
                        Neurons[l][j] += Neurons[l - 1][i] * Synweights[l - 1][i][j];
                    }

                    // decide if current layer is Output
                    if (l == NumberOfLayers - 1)
                        continue;
                    else
                        //Calculate the output as f_act(Input)
                        Neurons[l][j] = FUNCTION(Neurons[l][j]);
                }
            }


            // backward

            // output layer
            l = NumberOfLayers - 1;
            for (j = 0; j < NeuronsPerLayer[l]; j++) {
                // Calculate delta. Since the Output is linear, there is no need
                // to calculate the derivate of the activation function
                // here the deltas have to be multiplied with the weight of the event
                deltas[l][j] = Neurons[l][j] - desired[j];
                deltas[l][j] *= ev->eventWeights[nEv];
            }

            //Beginning from last layer where the next layer is hiden layer
            for (l = NumberOfLayers - 2; l >= 0; l--) {
                //for every Neuron on the current Layer
                for (j = 0; j < NeuronsPerLayer[l]; j++) {
                    sumdeltas = 0.0;
                    //for every Neuron on the next higher Layer
                    for (k = 0; k < NeuronsPerLayer[l + 1] - bias[l + 1]; k++) {
                        //Calculate delta_k*w_jk to calculate the new deltas
                        sumdeltas += deltas[l + 1][k] * Synweights[l][j][k];
                    }
                    //Calculate delta for current layer
                    deltas[l][j] = DERIVATE(Neurons[l][j]) * sumdeltas;
                } // end loop NeuronsPerLayer
            }   // end loop NumberOfLayers

            //For all Layers, upate Synapse weight
            for (l = 0; l < NumberOfLayers - 1; l++) {
                for (j = 0; j < NeuronsPerLayer[l + 1] - bias[l + 1]; j++) {
                    for (i = 0; i < NeuronsPerLayer[l]; i++) {
                        Synweights[l][i][j] += -learnRate * Neurons[l][i] * deltas[l + 1][j];
                    }
                }
            }

        } // end loop over events

        //reduce learnrate to get better minimum
        //check if we are in late epochs
        if (nEp >= lateEpochs) {
            // In order to lower the learning rate even more,
            learnRate *= (1.0 - sqrt(decayRate));
        } else {
            learnRate *= (1.0 - decayRate);
        }

    } // end loop over epochs


    /*ofstream file;
      file.open("eventvalues");
      for(i=0;i<100;i++){
      file<<ev->eventWeights[i]<<endl;
      for(j=0;j<nVars;j++){
      file<<ev->eventValues[i][j]<<endl;
      }
      file<<endl;
      }*/
    /*for(l=0;l<NumberOfLayers;l++){
      cout<<"Layer "<<l<<endl;
      for(i=0;i<NeuronsPerLayer[l];i++){
      cout<<"\tNeuron"<<i<<": "<<Neurons[l][i]<<endl;
      }
      }*/

    return;
}

/**
 ******************************************************************************
 * function
 * double*** CTrainMLP_b(CEvents* ev, double learnRate, int nVars, int nEpochs,
            int nEvents, double*** Synweights,
            int* NeuronsPerLayer, int NumberOfLayers,  int * bias,
            double decayRate, double max, double min)
 *
 *Trains the neural network with batch back propagation learning
 *
 *@param[in] ev               Structure with data of the events for training.
                              Includes the class, the weight and the values of the events
 *@param[in] learnRate        The learning rate for the neural network.
 *@param[in] nVars            Number of Inputvariables (without Bias Node!)
 *@param[in] nEpochs          Number of Epochs to train.
 *@param[in] nEvents          Number of Events to train.
 *@param[in,out] Synweights:  3-Dimensional array for Synapseweights.
 *                    Index 1 is the layer starting at the weights between
 *                                 the inputlayer and the first hiddenlayer.
 *                Index 2 is the neuron of the lower layer.
 *                Index 3 is the neuron of the upper layer.
 *@param[in] NeuronsPerLayer: Array that contains the number of neurons per layer.
 *@param[in] NumberOfLayers:  Number of Layers in the neural network.
 *@param[in] bias:            Number of bias nodes per layer
 *@param[in] decayRat:        Parameter for changing the learing rate
 *
 *@return
 ******************************************************************************/
void CTrainMLP_b(CEvents *ev, double learnRate, int nVars, int nEpochs,
                 int nEvents, double *** Synweights,
                 int *NeuronsPerLayer, int NumberOfLayers, int *bias,
                 double decayRate)
{

    int l, i, j, k;       //indices in for loops
    int lateEpochs = (int)(nEpochs * 0.95) - 1; // taken from TMVA for better learning

    double sumdeltas = 0.0; // Sum of delta_k*weight_jk used in back propagation

    int totalNeurons = NeuronsPerLayer[0];
    for (l = 1; l < NumberOfLayers; l++) {
        totalNeurons += NeuronsPerLayer[l];
    }
    cout << "total number of neurons:" << totalNeurons << endl;

    cout << "function CTrainMLP_b" << endl;

    double *** deltasEvents; // deltas, used in back propagation,
    // index1: Layer index2: Neuron per layer, index3: event
    deltasEvents = (double ** *) malloc(NumberOfLayers * sizeof(double **));
    for (l = 0; l < NumberOfLayers; l++) {
        deltasEvents[l] =  (double **) malloc(NeuronsPerLayer[l] * sizeof(double *));
    }
    for (l = 0; l < NumberOfLayers; l++) {
        for (i = 0; i < NeuronsPerLayer[l]; i++) {
            deltasEvents[l][i] = (double *) malloc(nEvents * sizeof(double));
        }
    }

    double *** neuronsEvents; // value of the neurons
    // index1: Layer, index2: Neuron per layer, index3: event
    neuronsEvents = (double ** *) malloc(NumberOfLayers * sizeof(double **));
    for (l = 0; l < NumberOfLayers; l++) {
        neuronsEvents[l] =  (double **) malloc(NeuronsPerLayer[l] * sizeof(double *));
    }
    for (l = 0; l < NumberOfLayers; l++) {
        for (i = 0; i < NeuronsPerLayer[l]; i++) {
            neuronsEvents[l][i] = (double *) malloc(nEvents * sizeof(double));
        }
    }


    // set neurons of bias nodes
    for (i = 0; i < NumberOfLayers - 1; i++) {
        if (bias[i] != 0) {
            for (int nEv = 0; nEv < nEvents; nEv++) {
                neuronsEvents[i][NeuronsPerLayer[i] - 1][nEv] = 1.0;
            }
        }
    }

    // initialization for each event the input layer for batch learning
    for (i = 0; i < NeuronsPerLayer[0] - bias[0]; i++) {
        //Use Eventvalues as output of the input layer to make the next step easier.
        for (int nEv = 0; nEv < nEvents; nEv++) {
            neuronsEvents[0][i][nEv] = ev->eventValues[nEv][i];
        }
    }

    // restore the desired output values
    int lastLayer   = NumberOfLayers - 1;
    int lastNeurons = NeuronsPerLayer[lastLayer];
    double desired[nEvents][lastNeurons];
    for (int nEv = 0; nEv < nEvents; nEv++) {
        for (i = 0; i < lastNeurons; i++) {
            if (ev->eventClass[nEv] == i) {
                desired[nEv][i] = 1.0;
            } else {
                desired[nEv][i] = 0.0;
            }
        }
    }



    for (int nEp = 0; nEp < nEpochs; nEp++) {     //for each epoch

        // batch learning

        //for all events do forward propagation
        for (int nEv = 0; nEv < nEvents; nEv++) {   //for each event

            //For each layer except Input and Output Layer.
            for (l = 1; l < NumberOfLayers - 1; l++) {

                // For each neuron on the next layer except bias node
                for (j = 0; j < NeuronsPerLayer[l] - bias[l]; j++) {

                    // Calculate the neuron input
                    neuronsEvents[l][j][nEv] = 0.0;
                    // For each input coming from the lower layer
                    for (i = 0; i < NeuronsPerLayer[l - 1]; i++)
                        // Calculate the neuron input
                        neuronsEvents[l][j][nEv] += neuronsEvents[l - 1][i][nEv] * Synweights[l - 1][i][j];

                    //Calculate the output as f_act(Input)
                    neuronsEvents[l][j][nEv] = FUNCTION( neuronsEvents[l][j][nEv]);
                }
            }

            // for the output layer, no activation funktion is applied
            l = NumberOfLayers - 1;
            for (j = 0; j < NeuronsPerLayer[l] - bias[l]; j++) {
                neuronsEvents[l][j][nEv] = 0.0;
                for (i = 0; i < NeuronsPerLayer[l - 1]; i++) {
                    neuronsEvents[l][j][nEv] += neuronsEvents[l - 1][i][nEv] * Synweights[l - 1][i][j];
                }
            }

        }

        // now, for all events do backward propagation

        // first compute all deltas
        for (int nEv = 0; nEv < nEvents; nEv++) {   //for each event

            // output layer
            l = NumberOfLayers - 1;
            for (j = 0; j < NeuronsPerLayer[l]; j++) {
                // Calculate delta. Since the Output is linear, there is no need
                // to calculate the derivate of the activation function
                // here the deltas have to be multiplied with the weight of the event
                deltasEvents[l][j][nEv] = neuronsEvents[l][j][nEv] - desired[nEv][j];
                deltasEvents[l][j][nEv] *= ev->eventWeights[nEv];
            }

            //Beginning from last layer where the next layer is hiden layer
            for (l = NumberOfLayers - 2; l >= 0; l--) {
                //for every Neuron on the current Layer
                for (j = 0; j < NeuronsPerLayer[l]; j++) {
                    sumdeltas = 0.0;
                    //for every Neuron on the next higher Layer
                    for (k = 0; k < NeuronsPerLayer[l + 1] - bias[l + 1]; k++) {
                        //Calculate delta_k*w_jk to calculate the new deltas
                        sumdeltas += deltasEvents[l + 1][k][nEv] * Synweights[l][j][k];
                    }
                    //Calculate delta for current layer
                    deltasEvents[l][j][nEv] = DERIVATE(neuronsEvents[l][j][nEv]) * sumdeltas;
                } // end loop NeuronsPerLayer
            }   // end loop NumberOfLayers

        }

        // secondly update weights

        //For all Layers, upate Synapse weight
        for (l = 0; l < NumberOfLayers - 1; l++) {
            for (j = 0; j < NeuronsPerLayer[l + 1] - bias[l + 1]; j++) {
                for (i = 0; i < NeuronsPerLayer[l]; i++) {
                    sumdeltas = 0.0;
                    for (int nEv = 0; nEv < nEvents; nEv++) { //for each event
                        sumdeltas += neuronsEvents[l][i][nEv] * deltasEvents[l + 1][j][nEv];
                    }
                    Synweights[l][i][j] += -learnRate * sumdeltas / (double)nEvents;
                }
            }

        } // end loop over events

        //reduce learnrate to get better minimum
        //check if we are in late epochs
        if (nEp >= lateEpochs) {
            // In order to lower the learning rate even more,
            learnRate *= (1.0 - sqrt(decayRate));
        } else {
            learnRate *= (1.0 - decayRate);
        }

    } // end loop over epochs

    return;
}

/**
 ******************************************************************************
 * function
 * double*** CTrainMLP_m(CEvents* ev,  double*** Synweights, double learnRate, double decayRate,
 *               int nVars, int nEpochs, int nEvents, int NumberOfLayers,
 *               int* NeuronsPerLayer, int * bias, int events)
 *
 *Trains the neural network with batch back propagation learning
 *
 *@param[in] ev               Structure with data of the events for training.
                              Includes the class, the weight and the values of the events
 *@param[in] learnRate        The learning rate for the neural network.
 *@param[in] nVars            Number of Inputvariables (without Bias Node!)
 *@param[in] nEpochs          Number of Epochs to train.
 *@param[in] nEvents          Number of Events to train.
 *@param[in,out] Synweights:  3-Dimensional array for Synapseweights.
 *                    Index 1 is the layer starting at the weights between
 *                                 the inputlayer and the first hiddenlayer.
 *                Index 2 is the neuron of the lower layer.
 *                Index 3 is the neuron of the upper layer.
 *@param[in] NeuronsPerLayer: Array that contains the number of neurons per layer.
 *@param[in] NumberOfLayers:  Number of Layers in the neural network.
 *@param[in] bias:            Number of bias nodes per layer
 *@param[in] decayRat:        Parameter for changing the learing rate
 *@param[in] events:          Parameter for number of events in one batch for
 *                            learing,
 *                            events=nEvents: batch, events=1: online
 *
 *@return
 ******************************************************************************/
void CTrainMLP_m(CEvents *ev,  double *** Synweights, double learnRate, double decayRate,
                 int nVars, int nEpochs, int nEvents, int NumberOfLayers,
                 int *NeuronsPerLayer, int *bias, int events)
{

    int l, i, j, k;
    int nEv;
    int lateEpochs = (int)(nEpochs * 0.95) - 1; // from TMVA for better learning

    double sumdeltas; // Sum of delta_k*weight_jk used in back propagation
    double tmp;           // temporary storage for multiplications

    int totalNeurons = NeuronsPerLayer[0];
    for (l = 1; l < NumberOfLayers; l++) {
        totalNeurons += NeuronsPerLayer[l];
    }
    cout << "total number of neurons:" << totalNeurons << endl;
    cout << "function CTrainMLP_m" << endl;

    double *** deltasEvents; // deltas, used in back propagation,
    // index 1: Layer, 2: Neuron per layer, 3: event
    deltasEvents = (double ** *) malloc(NumberOfLayers * sizeof(double **));
    for (l = 0; l < NumberOfLayers; l++) {
        deltasEvents[l] =  (double **) malloc(NeuronsPerLayer[l] * sizeof(double *));
    }
    for (l = 0; l < NumberOfLayers; l++) {
        for (i = 0; i < NeuronsPerLayer[l]; i++) {
            deltasEvents[l][i] = (double *) malloc(nEvents * sizeof(double));
        }
    }

    double *** neuronsEvents; // value of the neurons
    // index 1: Layer, 2: Neuron per layer, 3: event
    neuronsEvents = (double ** *) malloc(NumberOfLayers * sizeof(double **));
    for (l = 0; l < NumberOfLayers; l++) {
        neuronsEvents[l] =  (double **) malloc(NeuronsPerLayer[l] * sizeof(double *));
    }
    for (l = 0; l < NumberOfLayers; l++) {
        for (i = 0; i < NeuronsPerLayer[l]; i++) {
            neuronsEvents[l][i] = (double *) malloc(nEvents * sizeof(double));
        }
    }

    // set neurons of bias nodes
    for (i = 0; i < NumberOfLayers - 1; i++) {
        if (bias[i] != 0) {
            for (nEv = 0; nEv < nEvents; nEv++) {
                neuronsEvents[i][NeuronsPerLayer[i] - 1][nEv] = 1.0;
            }
        }
    }

    //Initialization for each event the input layer for batch learning
    for (i = 0; i < NeuronsPerLayer[0] - bias[0]; i++) {
        for (nEv = 0; nEv < nEvents; nEv++) {
            neuronsEvents[0][i][nEv] = ev->eventValues[nEv][i];
        }
    }

    // restore the desired output values
    int lastLayer   = NumberOfLayers - 1;
    int lastNeurons = NeuronsPerLayer[lastLayer];
    double desired[lastNeurons][nEvents];
    for (i = 0; i < lastNeurons; i++) {
        for (int nEv = 0; nEv < nEvents; nEv++) {
            if (ev->eventClass[nEv] == i) {
                desired[i][nEv] = 1.0;
            } else {
                desired[i][nEv] = 0.0;
            }
        }
    }

    // number of loops for mixed batch - online learning
    int parts = nEvents / events;

    for (int nEp = 0; nEp < nEpochs; nEp++) {         //for each epoch

        // do mixed learning, i.e. learn in iparts batches of size events
        int nEv_start = nEv = 0;
        int nEv_stop = events;
        for (int iparts = 0; iparts < parts; iparts++) {

            //for one batch of events do forward propagation
            //here: calculate the output as f_act(Input) not for the output layer
            for (l = 1; l < lastLayer; l++) {
                for (j = 0; j < NeuronsPerLayer[l] - bias[l]; j++) {
                    for (nEv = nEv_start; nEv < nEv_stop; nEv++) {
                        tmp = 0.0;
                        neuronsEvents[l][j][nEv] = 0.0;
                        for (i = 0; i < NeuronsPerLayer[l - 1]; i++) {
                            tmp += Synweights[l - 1][i][j] * neuronsEvents[l - 1][i][nEv];
                        }
                        neuronsEvents[l][j][nEv] = FUNCTION(tmp);
                    }
                }
            }

            l = NumberOfLayers - 1;
            for (j = 0; j < NeuronsPerLayer[l] - bias[l]; j++) {
                for (nEv = nEv_start; nEv < nEv_stop; nEv++) {
                    tmp = 0.0;
                    for (i = 0; i < NeuronsPerLayer[l - 1]; i++) {
                        tmp += Synweights[l - 1][i][j] * neuronsEvents[l - 1][i][nEv];

                    }
                    neuronsEvents[l][j][nEv] = tmp;
                }
            } // end loop over one batch

            // now, for one batch of events do backward propagation

            // first compute all deltas
            // output layer
            for (j = 0; j < NeuronsPerLayer[lastLayer]; j++) {
                for ( nEv = nEv_start; nEv < nEv_stop; nEv++) {
                    deltasEvents[lastLayer][j][nEv] = (neuronsEvents[lastLayer][j][nEv] -
                                                       desired[j][nEv]) * ev->eventWeights[nEv];
                }
            }

            //Beginning from last layer where the next layer is hiden layer
            for (l = NumberOfLayers - 2; l >= 0; l--) {
                //for every Neuron on the current Layer
                for (j = 0; j < NeuronsPerLayer[l]; j++) {
                    for ( nEv = nEv_start; nEv < nEv_stop; nEv++) {
                        sumdeltas = 0.0;
                        //for every Neuron on the next higher Layer
                        for (k = 0; k < NeuronsPerLayer[l + 1] - bias[l + 1]; k++) {
                            sumdeltas += Synweights[l][j][k] * deltasEvents[l + 1][k][nEv];
                        }
                        deltasEvents[l][j][nEv] = DERIVATE(neuronsEvents[l][j][nEv]) * sumdeltas;
                    } // end loop over one batch
                }   // end loop NeuronsPerLayer
            }     // end loop NumberOfLayers

            // secondly update weights

            //For all Layers, upate Synapse weight
            for (l = 0; l < NumberOfLayers - 1; l++) {
                for (j = 0; j < NeuronsPerLayer[l + 1] - bias[l + 1]; j++) {
                    for (i = 0; i < NeuronsPerLayer[l]; i++) {
                        sumdeltas = 0.0;
                        for (nEv = nEv_start; nEv < nEv_stop; nEv++) {
                            sumdeltas += neuronsEvents[l][i][nEv] * deltasEvents[l + 1][j][nEv];
                        }
                        Synweights[l][i][j] += -learnRate * sumdeltas / (double)events;
                    }
                }
            } // end loop over layers

            nEv_start += events;
            nEv_stop  += events;

        } // end loop over parts

        //reduce learnrate to get better minimum
        //check if we are in late epochs
        if (nEp >= lateEpochs) {
            // In order to lower the learning rate even more,
            learnRate *= (1.0 - sqrt(decayRate));
        } else {
            learnRate *= (1.0 - decayRate);
        }

    } // end loop over epochs

    return;
}

/**
 ******************************************************************************
 * function
 * void CTrainMLP_4(CEvents* ev,  double*** Synweights, double learnRate, double decayRate,
 *                  int nVars, int nEpochs, int nEvents, int NumberOfLayers,
 *                  int* NeuronsPerLayer, int * bias, int events)
 *
 *Trains the neural network with batch back propagation learning with 2 hidden layers
 *
 *@param[in] ev               Structure with data of the events for training.
                              Includes the class, the weight and the values of the events
 *@param[in,out] Synweights:  3-Dimensional array for Synapseweights.
 *                    Index 1 is the layer starting at the weights between
 *                                 the inputlayer and the first hiddenlayer.
 *                Index 2 is the neuron of the lower layer.
 *                Index 3 is the neuron of the upper layer.
 *@param[in] learnRate        The learning rate for the neural network.
 *@param[in] decayRat:        Parameter for changing the learing rate
 *@param[in] nVars            Number of Inputvariables (without Bias Node!)
 *@param[in] nEpochs          Number of Epochs to train.
 *@param[in] nEvents          Number of Events to train.
 *@param[in] NumberOfLayers:  Number of Layers in the neural network.
 *@param[in] NeuronsPerLayer: Array that contains the number of neurons per layer.
 *@param[in] bias:            Number of bias nodes per layer
 *@param[in] events:          Parameter for number of events in one batch for
 *                            learing,
 *                            events=nEvents: batch, events=1: online
 *
 *@return
 ******************************************************************************/
void CTrainMLP_4(CEvents *ev,  double *** Synweights, double learnRate, double decayRate,
                 int nVars, int nEpochs, int nEvents, int NumberOfLayers,
                 int *NeuronsPerLayer, int *bias, int events)
{

#ifdef BLAS
    if (nEvents != events) {
        printf("blas routines are only in bach mode implemented, i.e. nEvents= events\n");
        exit(1);
    } else {
        printf("running with blas routines\n");
    }
#else
    double sumdeltas; // Sum of delta_k*weight_jk used in back propagation
    double tmp;           // temporary storage for multiplications
#endif

    int i, j, k, m, n;
    int nEv;
    int lateEpochs = (int)(nEpochs * 0.95) - 1; // from TMVA for better learning


    cout << "function CTrainMLP_4" << endl;

    double deltas1[NeuronsPerLayer[1]][nEvents];
    double deltas2[NeuronsPerLayer[2]][nEvents];
    double deltas3[NeuronsPerLayer[3]][nEvents];

    double neurons0[NeuronsPerLayer[0]][nEvents];
    double neurons1[NeuronsPerLayer[1]][nEvents];
    double neurons2[NeuronsPerLayer[2]][nEvents];
    double neurons3[NeuronsPerLayer[3]][nEvents];

    if (bias[0] != 0) {
        for (nEv = 0; nEv < nEvents; nEv++) {
            neurons0[NeuronsPerLayer[0] - 1][nEv] = 1.0;
        }
    }
    if (bias[1] != 0) {
        for (nEv = 0; nEv < nEvents; nEv++) {
            neurons1[NeuronsPerLayer[1] - 1][nEv] = 1.0;
        }
    }
    if (bias[2] != 0) {
        for (nEv = 0; nEv < nEvents; nEv++) {
            neurons2[NeuronsPerLayer[2] - 1][nEv] = 1.0;
        }
    }
    if (bias[3] != 0) {
        for (nEv = 0; nEv < nEvents; nEv++) {
            neurons3[NeuronsPerLayer[3] - 1][nEv] = 1.0;
        }
    }

    //Initialization for each event the input layer for batch learning
    for (i = 0; i < NeuronsPerLayer[0] - bias[0]; i++) {
        for (nEv = 0; nEv < nEvents; nEv++) {
            neurons0[i][nEv] = ev->eventValues[nEv][i];
        }
    }

    // restore the desired output values
    int lastLayer   = NumberOfLayers - 1;
    int lastNeurons = NeuronsPerLayer[lastLayer];
    double desired[lastNeurons][nEvents];
    for (i = 0; i < lastNeurons; i++) {
        for (int nEv = 0; nEv < nEvents; nEv++) {
            if (ev->eventClass[nEv] == i) {
                desired[i][nEv] = 1.0;
            } else {
                desired[i][nEv] = 0.0;
            }
        }
    }

    // restore synweights
    double synapses0[NeuronsPerLayer[0]][NeuronsPerLayer[1] - bias[1]];
    double synapses1[NeuronsPerLayer[1]][NeuronsPerLayer[2] - bias[2]];
    double synapses2[NeuronsPerLayer[2]][NeuronsPerLayer[3] - bias[3]];

    for (i = 0; i < NeuronsPerLayer[0]; i++)
        for (j = 0; j < NeuronsPerLayer[1] - bias[1]; j++)
            synapses0[i][j] = Synweights[0][i][j];

    for (i = 0; i < NeuronsPerLayer[1]; i++)
        for (j = 0; j < NeuronsPerLayer[2] - bias[2]; j++)
            synapses1[i][j] = Synweights[1][i][j];

    for (i = 0; i < NeuronsPerLayer[2]; i++)
        for (j = 0; j < NeuronsPerLayer[3] - bias[3]; j++)
            synapses2[i][j] = Synweights[2][i][j];




    // number of loops for mixed batch - online learning
    int parts = nEvents / events;

    for (int nEp = 0; nEp < nEpochs; nEp++) {         //for each epoch

        // do mixed learning, i.e. learn in iparts batches of size events
        int nEv_start = nEv = 0;
        int nEv_stop = events;
        for (int iparts = 0; iparts < parts; iparts++) {

            //for one batch of events do forward propagation
            //here: calculate the output as f_act(Input) not for the output layer

#ifdef BLAS
            // compute c[m][n] = a[k][m] * b[k][n];
            n = events;
            m = NeuronsPerLayer[1] - bias[1];
            k = NeuronsPerLayer[0];
            cblas_dgemm (CblasRowMajor, CblasTrans, CblasNoTrans,
                         m, n, k, 1.0, synapses0[0], m, neurons0[0], n,
                         0.0, neurons1[0], n);
            for (j = 0; j < NeuronsPerLayer[1] - bias[1]; j++) {
                for ( nEv = nEv_start; nEv < nEv_stop; nEv++) {
                    neurons1[j][nEv] = FUNCTION(neurons1[j][nEv]);
                }
            }
            m = NeuronsPerLayer[2] - bias[2];
            k = NeuronsPerLayer[1];
            cblas_dgemm (CblasRowMajor, CblasTrans, CblasNoTrans,
                         m, n, k, 1.0, synapses1[0], m, neurons1[0], n,
                         0.0, neurons2[0], n);
            for (j = 0; j < NeuronsPerLayer[2] - bias[2]; j++) {
                for ( nEv = nEv_start; nEv < nEv_stop; nEv++) {
                    neurons2[j][nEv] = FUNCTION(neurons2[j][nEv]);
                }
            }
            m = NeuronsPerLayer[3] - bias[3];
            k = NeuronsPerLayer[2];
            cblas_dgemm (CblasRowMajor, CblasTrans, CblasNoTrans,
                         m, n, k, 1.0, synapses2[0], m, neurons2[0], n,
                         0.0, neurons3[0], n);
#else

            for (j = 0; j < NeuronsPerLayer[1] - bias[1]; j++) {
                for (nEv = nEv_start; nEv < nEv_stop; nEv++) {
                    tmp = 0.0;
                    for (i = 0; i < NeuronsPerLayer[0]; i++) {
                        tmp += synapses0[i][j] * neurons0[i][nEv];
                    }
                    neurons1[j][nEv] = FUNCTION(tmp);
                }
            }
            for (j = 0; j < NeuronsPerLayer[2] - bias[2]; j++) {
                for (nEv = nEv_start; nEv < nEv_stop; nEv++) {
                    tmp = 0.0;
                    for (i = 0; i < NeuronsPerLayer[1]; i++) {
                        tmp += synapses1[i][j] * neurons1[i][nEv];
                    }
                    neurons2[j][nEv] = FUNCTION(tmp);
                }
            }
            for (j = 0; j < NeuronsPerLayer[3] - bias[3]; j++) {
                for (nEv = nEv_start; nEv < nEv_stop; nEv++) {
                    tmp = 0.0;
                    for (i = 0; i < NeuronsPerLayer[2]; i++) {
                        tmp += synapses2[i][j] * neurons2[i][nEv];

                    }
                    neurons3[j][nEv] = tmp;
                }
            } // end loop over one batch
#endif

            // now, for one batch of events do backward propagation

            // first compute all deltas
            // output layer
            for (j = 0; j < lastNeurons; j++) {
                for ( nEv = nEv_start; nEv < nEv_stop; nEv++) {
                    deltas3[j][nEv] = (neurons3[j][nEv] - desired[j][nEv]) * ev->eventWeights[nEv];
                }
            }

            //Now the hidden layers
            //ACHTUNG; HIER WAR DIE "0" ZU VIEL
            //for (l=NumberOfLayers-2;l>=0;l--){

            // compute c[m][n] = a[m][k] * b[k][n];
#ifdef BLAS
            m = NeuronsPerLayer[2];
            k = NeuronsPerLayer[3] - bias[3];
            cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans,
                         m, events, k, 1.0, synapses2[0], k, deltas3[0], events,
                         0.0, deltas2[0], events);
            for (j = 0; j < NeuronsPerLayer[1]; j++) {
                for ( nEv = nEv_start; nEv < nEv_stop; nEv++) {
                    deltas2[j][nEv] *= DERIVATE(neurons2[j][nEv]);
                }
            }

            m = NeuronsPerLayer[1];
            k = NeuronsPerLayer[2] - bias[2];
            cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans,
                         m, events, k, 1.0, synapses1[0], k, deltas2[0], events,
                         0.0, deltas1[0], events);
            for (j = 0; j < NeuronsPerLayer[1]; j++) {
                for ( nEv = nEv_start; nEv < nEv_stop; nEv++) {
                    deltas1[j][nEv] *= DERIVATE(neurons1[j][nEv]);
                }
            }
#else


            for (j = 0; j < NeuronsPerLayer[2]; j++) {
                for ( nEv = nEv_start; nEv < nEv_stop; nEv++) {
                    sumdeltas = 0.0;
                    for (k = 0; k < NeuronsPerLayer[3] - bias[3]; k++) {
                        sumdeltas += synapses2[j][k] * deltas3[k][nEv];
                    }
                    deltas2[j][nEv] = DERIVATE(neurons2[j][nEv]) * sumdeltas;
                }
            }
            for (j = 0; j < NeuronsPerLayer[1]; j++) {
                for ( nEv = nEv_start; nEv < nEv_stop; nEv++) {
                    sumdeltas = 0.0;
                    for (k = 0; k < NeuronsPerLayer[2] - bias[2]; k++) {
                        sumdeltas += synapses1[j][k] * deltas2[k][nEv];
                    }
                    deltas1[j][nEv] = DERIVATE(neurons1[j][nEv]) * sumdeltas;
                }
            }
#endif

            // secondly update weights

            //For all Layers, upate Synapse weight

            double rate = -learnRate / (double)events;

            // compute c[m][n] = a[m][k] * b[n][k];
#ifdef BLAS
            k = events;
            m = NeuronsPerLayer[0];
            n = NeuronsPerLayer[1] - bias[1];
            cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasTrans,
                         m, n, k, rate, neurons0[0], k, deltas1[0], k,
                         1.0, synapses0[0], n);
            m = NeuronsPerLayer[1];
            n = NeuronsPerLayer[2] - bias[2];
            cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasTrans,
                         m, n, k, rate, neurons1[0], k, deltas2[0], k,
                         1.0, synapses1[0], n);
            m = NeuronsPerLayer[2];
            n = NeuronsPerLayer[3] - bias[3];
            cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasTrans,
                         m, n, k, rate, neurons2[0], k, deltas3[0], k,
                         1.0, synapses2[0], n);
#else

            for (j = 0; j < NeuronsPerLayer[1] - bias[1]; j++) {
                for (i = 0; i < NeuronsPerLayer[0]; i++) {
                    sumdeltas = 0.0;
                    for (nEv = nEv_start; nEv < nEv_stop; nEv++) {
                        sumdeltas += neurons0[i][nEv] * deltas1[j][nEv];
                    }
                    synapses0[i][j] += rate * sumdeltas;
                }
            }
            for (j = 0; j < NeuronsPerLayer[2] - bias[2]; j++) {
                for (i = 0; i < NeuronsPerLayer[1]; i++) {
                    sumdeltas = 0.0;
                    for (nEv = nEv_start; nEv < nEv_stop; nEv++) {
                        sumdeltas += neurons1[i][nEv] * deltas2[j][nEv];
                    }
                    synapses1[i][j] += rate * sumdeltas;
                }
            }
            for (j = 0; j < NeuronsPerLayer[3] - bias[3]; j++) {
                for (i = 0; i < NeuronsPerLayer[2]; i++) {
                    sumdeltas = 0.0;
                    for (nEv = nEv_start; nEv < nEv_stop; nEv++) {
                        sumdeltas += neurons2[i][nEv] * deltas3[j][nEv];
                    }
                    synapses2[i][j] += rate * sumdeltas;
                }
            }
#endif


            nEv_start += events;
            nEv_stop  += events;

        } // end loop over parts

        //reduce learnrate to get better minimum
        //check if we are in late epochs
        if (nEp >= lateEpochs) {
            // In order to lower the learning rate even more,
            learnRate *= (1.0 - sqrt(decayRate));
        } else {
            learnRate *= (1.0 - decayRate);
        }

    } // end loop over epochs

    // restore Synweights
    for (i = 0; i < NeuronsPerLayer[0]; i++)
        for (j = 0; j < NeuronsPerLayer[1] - bias[1]; j++)
            Synweights[0][i][j] = synapses0[i][j];

    for (i = 0; i < NeuronsPerLayer[1]; i++)
        for (j = 0; j < NeuronsPerLayer[2] - bias[2]; j++)
            Synweights[1][i][j] = synapses1[i][j];

    for (i = 0; i < NeuronsPerLayer[2]; i++)
        for (j = 0; j < NeuronsPerLayer[3] - bias[3]; j++)
            Synweights[2][i][j] = synapses2[i][j];


    return;
}

/**
 ******************************************************************************
 * function
 * double** CTrainMLP_testing(CEvents* ev,int nEpochs,int nEvents, double*** Synweights,
 *             double** Neurons, int* NeuronsPerLayer, int NumberOfLayers,
 *             int * bias, double** testout){
 *
 *Test the neural network
 *
 *@param[in] ev               Structure with data of the events for training.
                              Includes the class, the weight and the values of the events
 *@param[in] nEpochs          Number of Epochs to train.
 *@param[in] nEvents          Number of Events to train.
 *@param[in] Synweights:      3-Dimensional array for Synapseweights.
 *                    Index 1 is the layer starting at the weights
 *                            between the inputlayer
 *                                 and the first hiddenlayer.
 *                Index 2 is the neuron of the lower layer.
 *                Index 3 is the neuron of the upper layer.
 *@param[in] Neurons:         Matrix for the outputvalue of each neuron.
 *                Index 1 is the layer.
 *                Index 2 is the neuron on the layer.
 *@param[in] NeuronsPerLayer: Array that contains the number of neurons per layer.
 *@param[in] NumberOfLayers:  Number of Layers in the neural network.
 *@param[in] bias:            Number of bias nodes per layer
 *@param[out] testout:        Results of the test pattern
 *
 ******************************************************************************/
void CTrainMLP_testing(CEvents *ev, int nEpochs, int nEvents, double *** Synweights,
                       double **Neurons, int *NeuronsPerLayer, int NumberOfLayers,
                       int *bias, double **testout)
{

    int l, i, j; //indices in for loops

    // set neurons of bias nodes
    for (i = 0; i < NumberOfLayers - 1; i++) {
        if (bias[i] != 0) {
            Neurons[i][NeuronsPerLayer[i] - 1] = 1.0;
        }
    }

    //for each event
    for (int nEv = 0; nEv < nEvents; nEv++) {

        // aus eventValue bias-Knoten wieder raus und for-loop nur bis < NeuronsPerLayer[0]-1
        for (i = 0; i < NeuronsPerLayer[0] - bias[0]; i++) {
            //Use Eventvalues as output of the input layer to make the next step easier.
            Neurons[0][i] = ev->eventValues[nEv][i];
        }

        //forward propagation

        //For each layer except Input and Output Layer.
        for (l = 1; l < NumberOfLayers; l++) {
            //For each neuron on the next layer except bias node
            for (j = 0; j < NeuronsPerLayer[l] - bias[l]; j++) {
                Neurons[l][j] = 0.0;
                //For each input coming from the lower layer
                for (i = 0; i < NeuronsPerLayer[l - 1]; i++) {
                    //Calculate the neuron input
                    Neurons[l][j] += Neurons[l - 1][i] * Synweights[l - 1][i][j];
                }

                if (l == NumberOfLayers - 1)
                    //decide if current layer is Output
                    continue;
                else
                    //Calculate the output as f_act(Input)
                    Neurons[l][j] = FUNCTION(Neurons[l][j]);
            }
        }
        for (i = 0; i < NeuronsPerLayer[NumberOfLayers - 1]; i++) {
            testout[nEv][i] = Neurons[NumberOfLayers - 1][i];
        }
    }
    return;
}






void CTrainMLP_opencl(CEvents *ev,  double *** Synweights, double learnRate, double decayRate,
                      int nVars, int nEpochs, int nEvents, int NumberOfLayers,
                      int *NeuronsPerLayer, int *bias, int events)
{
    double sumdeltas;   // Sum of delta_k*weight_jk used in back propagation
    double tmp;           // temporary storage for multiplications

    int i, j, k;
    int nEv;
    int lateEpochs = (int)(nEpochs * 0.95) - 1;



    double **deltas1 = (double **) malloc(nEvents * sizeof(double *));
    double **deltas2 = (double **) malloc(nEvents * sizeof(double *));
    double **deltas3 = (double **) malloc(nEvents * sizeof(double *));
    deltas1[0] = (double *) malloc(nEvents * NeuronsPerLayer[1] * sizeof(double));
    deltas2[0] = (double *) malloc(nEvents * NeuronsPerLayer[2] * sizeof(double));
    deltas3[0] = (double *) malloc(nEvents * NeuronsPerLayer[3] * sizeof(double));

    for (i = 1; i < nEvents; i++) {
        deltas1[i] = deltas1[0] + i * NeuronsPerLayer[1];
        deltas2[i] = deltas2[0] + i * NeuronsPerLayer[2];
        deltas3[i] = deltas3[0] + i * NeuronsPerLayer[3];
    }
    double **neurons0 = (double **) malloc(nEvents * sizeof(double *));
    double **neurons1 = (double **) malloc(nEvents * sizeof(double *));
    double **neurons2 = (double **) malloc(nEvents * sizeof(double *));
    double **neurons3 = (double **) malloc(nEvents * sizeof(double *));
    neurons0[0] = (double *) malloc(nEvents * NeuronsPerLayer[0] * sizeof(double));
    neurons1[0] = (double *) malloc(nEvents * NeuronsPerLayer[1] * sizeof(double));
    neurons2[0] = (double *) malloc(nEvents * NeuronsPerLayer[2] * sizeof(double));
    neurons3[0] = (double *) malloc(nEvents * NeuronsPerLayer[3] * sizeof(double));

    for (i = 1; i < nEvents; i++) {
        neurons0[i] = neurons0[0] + i * NeuronsPerLayer[0];
        neurons1[i] = neurons1[0] + i * NeuronsPerLayer[1];
        neurons2[i] = neurons2[0] + i * NeuronsPerLayer[2];
        neurons3[i] = neurons3[0] + i * NeuronsPerLayer[3];
    }
    // double neurons0[nEvents][NeuronsPerLayer[0]];
    // double neurons1[nEvents][NeuronsPerLayer[1]];
    // double neurons2[nEvents][NeuronsPerLayer[2]];
    // double neurons3[nEvents][NeuronsPerLayer[3]];

    if (bias[0] != 0) {
        for (nEv = 0; nEv < nEvents; nEv++) {
            neurons0[nEv][NeuronsPerLayer[0] - 1] = 1.0;
        }
    }
    if (bias[1] != 0) {
        for (nEv = 0; nEv < nEvents; nEv++) {
            neurons1[nEv][NeuronsPerLayer[1] - 1] = 1.0;
        }
    }
    if (bias[2] != 0) {
        for (nEv = 0; nEv < nEvents; nEv++) {
            neurons2[nEv][NeuronsPerLayer[2] - 1] = 1.0;
        }
    }
    if (bias[3] != 0) {
        for (nEv = 0; nEv < nEvents; nEv++) {
            neurons3[nEv][NeuronsPerLayer[3] - 1] = 1.0;
        }
    }

    //Initialization for each event the input layer for batch learning
    for (i = 0; i < NeuronsPerLayer[0] - bias[0]; i++) {
        for (nEv = 0; nEv < nEvents; nEv++) {
            neurons0[nEv][i] = ev->eventValues[nEv][i];
        }
    }

    // restore event weights
    double *weights = (double *) malloc(nEvents * sizeof(double));
    for (nEv = 0; nEv < nEvents; nEv++) {
        weights[nEv] = ev->eventWeights[nEv];
    }

    // restore the desired output values
    int lastLayer   = NumberOfLayers - 1;
    int lastNeurons = NeuronsPerLayer[lastLayer];
    double desired[nEvents][lastNeurons];
    for (i = 0; i < lastNeurons; i++) {

        for (nEv = 0; nEv < nEvents; nEv++) {
            if (ev->eventClass[nEv] == i) {
                desired[nEv][i] = VALUEPLUS * weights[nEv];
            } else {
                desired[nEv][i] = VALUEMINUS * weights[nEv];
            }
        }
    }

    // restore synweights
    double synapses0[NeuronsPerLayer[0]][NeuronsPerLayer[1] - bias[1]];
    double synapses1[NeuronsPerLayer[1]][NeuronsPerLayer[2] - bias[2]];
    double synapses2[NeuronsPerLayer[2]][NeuronsPerLayer[3] - bias[3]];

    for (i = 0; i < NeuronsPerLayer[0]; i++)
        for (j = 0; j < NeuronsPerLayer[1] - bias[1]; j++)
            synapses0[i][j] = Synweights[0][i][j];

    for (i = 0; i < NeuronsPerLayer[1]; i++)
        for (j = 0; j < NeuronsPerLayer[2] - bias[2]; j++)
            synapses1[i][j] = Synweights[1][i][j];

    for (i = 0; i < NeuronsPerLayer[2]; i++)
        for (j = 0; j < NeuronsPerLayer[3] - bias[3]; j++)
            synapses2[i][j] = Synweights[2][i][j];

    // number of loops for mixed batch - online learning
    //Create everything needed for OpenCL




    int errNum;
    cl_context Context;
    cl_device_id device;
    cl_command_queue CommandQueue;
    Context = CTrainMLP_CreateContext();
    CommandQueue = CTrainMLP_CreateCommandQueue(Context, &device);
    stringstream ss;
    for (i = 0; i < NumberOfLayers; i++) {
        ss << "#define NEURONS" << i << " " << NeuronsPerLayer[i] << endl;
        ss << "#define NEURONSB" << i << " " << NeuronsPerLayer[i] - bias[i] << endl;
    }

    cl_program program = CTrainMLP_CreateProgram(Context, device, "kernelonline.cl", ss.str());
    cout << "create program" << endl;
    cl_kernel kernel_tanh = clCreateKernel(program,
                                           "CTrainMLP_forward_tanh", &errNum);
    cout << "create kernel tanh" << endl;
    if (errNum != CL_SUCCESS) {
        check_error(errNum);
        exit(0);
    }




    // number of loops for mixed batch - online learning
    int parts = nEvents / events;
    if (nEvents % events != 0) parts++;
    cl_mem memNeurons0, memNeurons1, memNeurons2, memNeurons3, memSynapses0, memSynapses1, memSynapses2, memNeuronsPerLayer, memBias;
            cl_mem memdeltas1, memdeltas2, memdeltas3, memdesired, memweights;
            size_t localWorkSize[2];
            size_t globalWorkSize[2];
            localWorkSize[0] = 8;
            localWorkSize[1] = 64;
            globalWorkSize[0] = 8;
            globalWorkSize[1] = 64;

        memNeurons0 = clCreateBuffer(Context,
                                         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         nEvents * NeuronsPerLayer[0]  * sizeof(double), neurons0[0], &errNum);

            memNeurons1 = clCreateBuffer(Context,
                                         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         nEvents * NeuronsPerLayer[1]  * sizeof(double), neurons1[0], &errNum);

            memNeurons2 = clCreateBuffer(Context,
                                         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         nEvents * NeuronsPerLayer[2]  * sizeof(double), neurons2[0], &errNum);
            memNeurons3 = clCreateBuffer(Context,
                                         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         nEvents * NeuronsPerLayer[3]  * sizeof(double), neurons3[0], &errNum);


            memSynapses0 = clCreateBuffer(Context,
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          sizeof(double) * NeuronsPerLayer[0] * (NeuronsPerLayer[1] - bias[1]), synapses0[0], &errNum);

            memSynapses1 = clCreateBuffer(Context,
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          sizeof(double) * NeuronsPerLayer[1] * (NeuronsPerLayer[2] - bias[2]), synapses1[0], &errNum);
            memSynapses2 = clCreateBuffer(Context,
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          sizeof(double) * NeuronsPerLayer[2] * (NeuronsPerLayer[3] - bias[3]), synapses2[0], &errNum);
            memdeltas1 = clCreateBuffer(Context,
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        sizeof(double) * nEvents * NeuronsPerLayer[1], deltas1[0], &errNum);
            memdeltas2 = clCreateBuffer(Context,
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        sizeof(double) * nEvents * NeuronsPerLayer[2], deltas2[0], &errNum);
            memdeltas3 = clCreateBuffer(Context,
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        sizeof(double) * nEvents * NeuronsPerLayer[3], deltas3[0], &errNum);
            memweights = clCreateBuffer(Context,
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        sizeof(double) * nEvents, weights, &errNum);

            memdesired = clCreateBuffer(Context,
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        sizeof(double) * nEvents * NeuronsPerLayer[3], desired[0], &errNum);

            memNeuronsPerLayer = clCreateBuffer(Context,
                                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                sizeof(int) * NumberOfLayers, NeuronsPerLayer, &errNum);

            memBias = clCreateBuffer(Context,
                                     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     sizeof(int) * NumberOfLayers, bias, &errNum);

            errNum = clSetKernelArg(kernel_tanh, 0,
                                    sizeof(cl_mem), (void *)&memNeurons0);
            errNum |= clSetKernelArg(kernel_tanh, 1,
                                     sizeof(cl_mem), (void *)&memNeurons1);
            errNum |= clSetKernelArg(kernel_tanh, 2,
                                     sizeof(cl_mem), (void *)&memNeurons2);
            errNum |= clSetKernelArg(kernel_tanh, 3,
                                     sizeof(cl_mem), (void *)&memNeurons3);
            errNum |= clSetKernelArg(kernel_tanh, 4,
                                     sizeof(cl_mem), (void *)&memSynapses0);
            errNum |= clSetKernelArg(kernel_tanh, 5,
                                     sizeof(cl_mem), (void *)&memSynapses1);
            errNum |= clSetKernelArg(kernel_tanh, 6,
                                     sizeof(cl_mem), (void *)&memSynapses2);
            errNum |= clSetKernelArg(kernel_tanh, 7,
                                     sizeof(cl_mem), (void *)&memdeltas1);
            errNum |= clSetKernelArg(kernel_tanh, 8,
                                     sizeof(cl_mem), (void *)&memdeltas2);
            errNum |= clSetKernelArg(kernel_tanh, 9,
                                     sizeof(cl_mem), (void *)&memdeltas3);
            errNum |= clSetKernelArg(kernel_tanh, 10,
                                     sizeof(cl_mem), (void *)&memdesired);
            errNum |= clSetKernelArg(kernel_tanh, 11,
                                     sizeof(cl_mem), (void *)&memweights);

            errNum |= clSetKernelArg(kernel_tanh, 12,
                                     sizeof(cl_mem), (void *)&memNeuronsPerLayer);
            errNum |= clSetKernelArg(kernel_tanh, 13,
                                     sizeof(cl_mem), (void *)&memBias);
            // errNum |= clSetKernelArg(kernel_tanh, 16,
            //                          sizeof(int), &events);

    for (int nEp = 0; nEp < nEpochs; nEp++) { //for each epoch

        //-------------------------------------------------------------------
        // do mixed learning, i.e. learn in iparts batches of size events
        int nEv_start = nEv = 0;
        int nEv_stop = events;
        for (int iparts = 0; iparts < parts; iparts++) {
            double rate = -learnRate / (double)events;
            cout<<nEv_stop-nEv_start<<endl;
            cout<<events<<endl;
            //for one batch of events do forward propagation
            //here: calculate the output as f_act(Input) not for the output layer
            
            // cout << "part: " << iparts << endl;
            // cout << "start: " << nEv_start;
            // cout << "stop: " << nEv_stop;
            // cout << "stop-start: " << nEv_stop - nEv_start << endl;
            


            
            errNum |= clSetKernelArg(kernel_tanh, 14,
                                     sizeof(int), &nEv_start);
            errNum |= clSetKernelArg(kernel_tanh, 15,
                                     sizeof(int), &nEv_stop);
            errNum |= clSetKernelArg(kernel_tanh, 16,
                                      sizeof(double), (void *)&rate);
            if (errNum != CL_SUCCESS) {
                cout << "set kernel arguments" << endl;
                check_error(errNum);
                exit(0);
            }
            errNum = clEnqueueNDRangeKernel(CommandQueue,
                                            kernel_tanh, 2, NULL, globalWorkSize,
                                            localWorkSize, 0, NULL, NULL);
            if (errNum != CL_SUCCESS) {
                cout << "start kernel" << endl;
                check_error(errNum);
                exit(0);
            }
            errNum = clFinish(CommandQueue);
            //errNum = clWaitForEvents(1, &event);
            if (errNum != CL_SUCCESS) {
                cout << "wait to calculate" << endl;
                check_error(errNum);
                exit(0);
            }

            
            

            //cout<<NeuronsPerLayer[0]<<endl;
            //cout<<NeuronsPerLayer[1]<<endl;
            //cout << "Read Buffer" << endl;
            if (errNum != CL_SUCCESS) {
                cout << "Read buffer back" << endl;
                check_error(errNum);
                exit(0);
            }

            nEv_start += events;
            nEv_stop  += events;
            if (nEv_stop > nEvents) nEv_stop = nEvents;

        } // end loop over parts

        //reduce learnrate to get better minimum
        //check if we are in late epochs
        if (nEp >= lateEpochs) {
            // In order to lower the learning rate even more,
            learnRate *= (1.0 - sqrt(decayRate));
        } else {
            learnRate *= (1.0 - decayRate);
        }

    } // end loop over epochs
            errNum = clEnqueueReadBuffer(CommandQueue,
                                         memSynapses0, CL_TRUE, 0, sizeof(double) * NeuronsPerLayer[0] * (NeuronsPerLayer[1] - bias[1]),
                                         synapses0[0], 0, NULL, NULL);
            errNum = clEnqueueReadBuffer(CommandQueue,
                                         memSynapses1, CL_TRUE, 0, sizeof(double) * NeuronsPerLayer[1] * (NeuronsPerLayer[2] - bias[2]),
                                         synapses1[0], 0, NULL, NULL);
            errNum = clEnqueueReadBuffer(CommandQueue,
                                         memSynapses2, CL_TRUE, 0, sizeof(double) * NeuronsPerLayer[2] * (NeuronsPerLayer[3] - bias[3]),
                                         synapses2[0], 0, NULL, NULL);
    // restore Synweights
    for (i = 0; i < NeuronsPerLayer[0]; i++)
        for (j = 0; j < NeuronsPerLayer[1] - bias[1]; j++)
            Synweights[0][i][j] = synapses0[i][j];

    for (i = 0; i < NeuronsPerLayer[1]; i++)
        for (j = 0; j < NeuronsPerLayer[2] - bias[2]; j++)
            Synweights[1][i][j] = synapses1[i][j];

    for (i = 0; i < NeuronsPerLayer[2]; i++)
        for (j = 0; j < NeuronsPerLayer[3] - bias[3]; j++)
            Synweights[2][i][j] = synapses2[i][j];


     clReleaseMemObject(memNeurons0);
            clReleaseMemObject(memNeurons1);
            clReleaseMemObject(memNeurons2);
            clReleaseMemObject(memNeurons3);
            clReleaseMemObject(memSynapses0);
            clReleaseMemObject(memSynapses1);
            clReleaseMemObject(memSynapses2);
            clReleaseMemObject(memdeltas1);
            clReleaseMemObject(memdeltas2);
            clReleaseMemObject(memdeltas3);
            clReleaseMemObject(memdesired);
            clReleaseMemObject(memweights);
            clReleaseMemObject(memNeuronsPerLayer);
            clReleaseMemObject(memBias);
    clReleaseContext(Context);
    clReleaseKernel(kernel_tanh);
    clReleaseProgram(program);
    clReleaseCommandQueue(CommandQueue);
    return;
}
/**
 ***************************************************************************************************
 *Creates context for openCL
 *@param[out] context: Created context
 ***************************************************************************************************/

cl_context CTrainMLP_CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;
    cl_device_id device;
    //get Platform and choose first one
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0) {
        cerr << "No OpenCL platforum found!" << endl;
        return NULL;
    }

    char buffer[10240];
    printf("=====  Platform 0 =====\n");
    clGetPlatformInfo(firstPlatformId, CL_PLATFORM_PROFILE, 10240, buffer, NULL);
    printf("  PROFILE = %s\n", buffer);
    clGetPlatformInfo(firstPlatformId, CL_PLATFORM_VERSION, 10240, buffer, NULL);
    printf("  VERSION = %s\n", buffer);
    clGetPlatformInfo(firstPlatformId, CL_PLATFORM_NAME, 10240, buffer, NULL);
    printf("  NAME = %s\n", buffer);
    clGetPlatformInfo(firstPlatformId, CL_PLATFORM_VENDOR, 10240, buffer, NULL);
    printf("  VENDOR = %s\n", buffer);
    //  clGetPlatformInfo(platforms[i],CL_PLATFORM_EXTENSIONS,10240,buffer,NULL);
    //  printf("  EXTENSIONS = %s\n", buffer);

    cl_uint devices_n;

    // get the GPU-devices of platform i, print details of the device
    errNum = clGetDeviceIDs( firstPlatformId, CL_DEVICE_TYPE_ALL, 1, &device,
                             &devices_n);
    if (errNum != CL_SUCCESS)
        printf("error getting device IDS\n");
    printf("  === %d OpenCL device(s) found on platform: 0\n\n", devices_n);
    for (unsigned int d = 0; d < devices_n; d++) {
        char buffer[10240];
        cl_uint buf_uint;
        cl_ulong buf_ulong;
        size_t buf[3];
        printf("  === --- Device -- %d \n", d);
        (clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer),
                         buffer, NULL));
        printf("    DEVICE_NAME = %s\n", buffer);
        (clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(buffer),
                         buffer, NULL));
        printf("    DEVICE_VENDOR = %s\n", buffer);
        (clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(buffer),
                         buffer, NULL));
        printf("    DEVICE_VERSION = %s\n", buffer);
        (clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(buffer),
                         buffer, NULL));
        printf("    DRIVER_VERSION = %s\n", buffer);
        (clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                         sizeof(buf_uint), &buf_uint, NULL));
        printf("    DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
        (clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                         sizeof(buf_uint), &buf_uint, NULL));
        printf("    DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
        (clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                         sizeof(buf_ulong), &buf_ulong, NULL));
        printf("    DEVICE_GLOBAL_MEM_SIZE = %u\n\n", (unsigned int)buf_ulong);
        (clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
                         sizeof(buf_ulong), &buf_ulong, NULL));
        printf("    DEVICE_LOCAL_MEM_SIZE = %u\n\n", (unsigned int)buf_ulong);
        (clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                         sizeof(buf_ulong), &buf_ulong, NULL));
        printf("    DEVICE_MAX_WORK_GROUP_SIZE = %u\n\n", (unsigned int)buf_ulong);
        (clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                         sizeof(buf_ulong), &buf_ulong, NULL));
        printf("    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = %u\n\n", (unsigned int)buf_ulong);

        (clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                         sizeof(buf), buf, NULL));
        printf("    CL_DEVICE_MAX_WORK_ITEM_SIZES1 = %zu\n\n", buf[0]);
        printf("    CL_DEVICE_MAX_WORK_ITEM_SIZES2 = %zu\n\n", buf[1]);
        printf("    CL_DEVICE_MAX_WORK_ITEM_SIZES3 = %zu\n\n", buf[2]);


    }
    if (devices_n == 0) {
        printf("error, on platform 0, there is no GPU device\n");
    }

    cl_context_properties contextProperties[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContext(contextProperties, devices_n, &device, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS) {
        cout << "Unable to create GPU context, try CPU..." << endl;
        check_error(errNum);
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_ALL, NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS) {
            cerr << "Unable to create GPU or CPU context" << endl;
            check_error(errNum);
            return NULL;
        }
    }
    cout << "Created GPU context" << endl;
    return context;
}

/**
***********************************************************************************************
*Creates Command queue for opencl
*@param[in] context: Context needed for Comannd queue
*@param[in] device: Device ID
*param[out] comandQueue: Created Command queue for specified context
***********************************************************************************************/
cl_command_queue CTrainMLP_CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS) {
        cerr << "Failed to get size of device buffer";
        return NULL;
    }
    if (deviceBufferSize <= 0) {
        cerr << "No devices available";
        return NULL;
    }
    //Allocate memory for device buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS) {
        cerr << "Failed to get device ID";
        return NULL;
    }
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL) {
        cerr << "Failed to create command queue";
        return NULL;
    }
    *device = devices[0];
    delete [] devices;
    return commandQueue;
}


cl_program CTrainMLP_CreateProgram(cl_context context, cl_device_id device, const char *fileName, string constants)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << constants;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char **)&srcStr,
                                        NULL, NULL);
    if (program == NULL) {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[10])
{
    for (int i = 0; i < 10; i++) {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}



void check_error(int error)
{
    if (error == CL_SUCCESS)
        printf("SUCCESS\n");
    else if (error == CL_INVALID_PLATFORM)
        printf("CL_INVALID_PLATFORM\n");
    else if (error == CL_INVALID_DEVICE_TYPE)
        printf("CL_INVALID_DEVICE_TYPE\n");
    else if (error == CL_INVALID_VALUE)
        printf("CL_INVALID_VALUE\n");
    else if (error == CL_DEVICE_NOT_FOUND)
        printf("CL_DEVICE_NOT_FOUND\n");
    else if (error == CL_DEVICE_NOT_AVAILABLE)
        printf("CL_DEVICE_NOT_AVAILABLE\n");
    else if (error == CL_INVALID_HOST_PTR)
        printf("CL_INVALID_HOST_PTR\n");
    else if (error == CL_INVALID_OPERATION)
        printf("CL_INVALID_OPERATION\n");
    else if (error == CL_INVALID_VALUE)
        printf("CL_INVALID_VALUE\n");
    else if (error == CL_INVALID_CONTEXT)
        printf("CL_INVALID_CONTEXT\n");
    else if (error == CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR\n");
    else if (error == CL_OUT_OF_HOST_MEMORY)
        printf("CL_OUT_OF_HOST_MEMORY\n");
    else if (error == CL_INVALID_OPERATION)
        printf("CL_INVALID_OPERATION\n");
    else if (error == CL_INVALID_BUFFER_SIZE)
        printf("CL_INVALID_BUFFER_SIZE\n");
    else if (error == CL_INVALID_PROGRAM)
        printf("CL_INVALID_PROGRAM\n");
    else if (error == CL_INVALID_PROGRAM_EXECUTABLE)
        printf("CL_INVALID_PROGRAM_EXECUTABLE\n");
    else if (error == CL_INVALID_KERNEL_NAME)
        printf("CL_INVALID_KERNEL_NAME\n");
    else if (error == CL_INVALID_KERNEL_DEFINITION)
        printf("CL_INVALID_KERNEL_DEFINITION\n");
    else if (error == CL_INVALID_KERNEL)
        printf("CL_INVALID_KERNEL\n");
    else if (error == CL_INVALID_ARG_INDEX)
        printf("CL_INVALID_ARG_INDEX\n");
    else if (error == CL_INVALID_ARG_VALUE)
        printf("CL_INVALID_ARG_VALUE\n");
    else if (error == CL_INVALID_MEM_OBJECT)
        printf("CL_INVALID_MEM_OBJECT\n");
    else if (error == CL_INVALID_SAMPLER)
        printf("CL_INVALID_SAMPLER\n");
    else if (error == CL_INVALID_ARG_SIZE)
        printf("CL_INVALID_ARG_SIZE\n");
    else if (error == CL_INVALID_VALUE)
        printf("CL_INVALID_VALUE\n");
    else if (error == CL_MEM_OBJECT_ALLOCATION_FAILURE)
        printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n");
    else if (error == CL_INVALID_COMMAND_QUEUE)
        printf("CL_INVALID_COMMAND_QUEUE\n");
    else if (error == CL_INVALID_CONTEXT)
        printf("CL_INVALID_CONTEXT\n");
    else if (error == CL_INVALID_EVENT_WAIT_LIST)
        printf("CL_INVALID_EVENT_WAIT_LIST\n");
    else if (error == CL_OUT_OF_HOST_MEMORY)
        printf("CL_OUT_OF_HOST_MEMORY\n");
    else if (error == CL_INVALID_PROGRAM_EXECUTABLE)
        printf("CL_INVALID_PROGRAM_EXECUTABLE\n");
    else if (error == CL_INVALID_COMMAND_QUEUE )
        printf("CL_INVALID_COMMAND_QUEUE \n");
    else if (error == CL_INVALID_KERNEL)
        printf("CL_INVALID_KERNEL\n");
    else if (error == CL_INVALID_CONTEXT)
        printf("CL_INVALID_CONTEXT\n");
    else if (error == CL_INVALID_KERNEL_ARGS)
        printf("CL_INVALID_KERNEL_ARGS\n");
    else if (error == CL_INVALID_WORK_DIMENSION)
        printf("CL_INVALID_WORK_DIMENSION\n");
    else if (error == CL_INVALID_WORK_GROUP_SIZE)
        printf("CL_INVALID_WORK_GROUP_SIZE\n");
    else if (error == CL_INVALID_GLOBAL_OFFSET )
        printf("CL_INVALID_GLOBAL_OFFSET \n");
    else if (error == CL_OUT_OF_RESOURCES )
        printf("CL_OUT_OF_RESOURCES \n");
    else if (error == CL_MEM_OBJECT_ALLOCATION_FAILURE)
        printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n");
    else if (error == CL_INVALID_EVENT_WAIT_LIST)
        printf("CL_INVALID_EVENT_WAIT_LIST\n");
    else if (error == CL_OUT_OF_HOST_MEMORY)
        printf("CL_OUT_OF_HOST_MEMORY\n");
    else
        printf("unbekannter ERROR\n");

}