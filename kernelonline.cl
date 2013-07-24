
#define FUNCTION  tanh     //function
#define DERIVATE tanhd     //derivate of function
#pragma OPENCL EXTENSION cl_khr_fp64: enable


double tanhd(double value)
{
    return 1.0 - value * value;
}


// for (nEv = nEv_start; nEv < nEv_stop; nEv++) {
//     // do forward propagation
//     for (j = 0; j < NeuronsPerLayer[1] - bias[1]; j++) {
//         tmp = 0.0;
//         for (i = 0; i < NeuronsPerLayer[0]; i++) {
//             tmp += neurons0[nEv][i] * synapses0[i][j];
//         }
//         neurons1[nEv][j] = FUNCTION(tmp);
//     }
// }
//          for (j = 0; j < NeuronsPerLayer[2] - bias[2]; j++) {
//     tmp = 0.0;
//     for (i = 0; i < NeuronsPerLayer[1]; i++) {
//         tmp += synapses1[i][j] * neurons1[nEv][i];
//     }

//     neurons2[nEv][j] = FUNCTION(tmp);

// }
__kernel void CTrainMLP_forward_tanh(__global double *Neurons0, __global double *Neurons1, __global double *Neurons2, __global double *Neurons3
                                     , __global double *Synapses0, __global double *Synapses1, __global double *Synapses2,
                                     __global int *NeuronsPerLayer, __global int *bias,
                                     int nEv_begin, int nEv_end, int nEvents)
{

    int nEv = get_global_id(0);
    int j = get_global_id(1);
    double tmp;
    if (nEv < nEv_end - nEv_begin && j < (NeuronsPerLayer[1] - bias[1])) {
        tmp = 0.0;
        for (int i = 0; i < NeuronsPerLayer[0]; i++) {
            tmp += Neurons0[(nEv + nEv_begin) * NeuronsPerLayer[0] + i] * Synapses0[i * (NeuronsPerLayer[1] - bias[1]) + j];
        }
        Neurons1[(nEv + nEv_begin) * (NeuronsPerLayer[1]) + j] = tanh(tmp); //BIAS!!!!
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (nEv < nEv_end - nEv_begin && j < (NeuronsPerLayer[2] - bias[2])) {
        tmp = 0.0;
        for (int i = 0; i < NeuronsPerLayer[1]; i++) {
            tmp += Synapses1[i * (NeuronsPerLayer[2] - bias[2]) + j] * Neurons1[(nEv + nEv_begin) * NeuronsPerLayer[1] + i] ;
        }
        Neurons2[(nEv + nEv_begin) * (NeuronsPerLayer[2]) + j] = tanh(tmp); //BIAS!!!!
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (nEv < nEv_end - nEv_begin && j < (NeuronsPerLayer[3] - bias[3])) {
        tmp = 0.0;
        for (int i = 0; i < NeuronsPerLayer[2]; i++) {
            tmp += Synapses2[i * (NeuronsPerLayer[3] - bias[3]) + j] * Neurons2[(nEv + nEv_begin) * NeuronsPerLayer[2] + i] ;
        }
        Neurons3[(nEv + nEv_begin) * (NeuronsPerLayer[3]) + j] = tmp; //BIAS!!!!
    }
    return;
}
__kernel void CTrainMLP_forward_linear(__global double *Neurons, __global double *weights, __global double *out, int uplayer, int downlayer,
                                       int nEv_begin, int nEv_end, int nEvents)
{
    int nEv = get_global_id(0);
    int j = get_global_id(1);
    double tmp;
    if (nEv < nEv_end - nEv_begin && j < uplayer) {
        tmp = 0.0;
        for (int i = 0; i < downlayer; i++) {
            tmp += Neurons[(nEv + nEv_begin) * downlayer + i] * weights[i * uplayer + j];
        }
        out[(nEv + nEv_begin) * (uplayer) + j] = tanh(tmp);//KEIN BIAS
    }
    return;
}


/*__kernel void CTrainMLP_kernel(__global int *eventClass, __global double *eventWeights, __global double *eventValues,  double learnRate, int nVars,
                               __global double *Synweightsout,
                               __global int *NeuronsPerLayer, __global int *bias,
                               double decayRate)
{
    int l, i, j, k;       //indices in for loops
    int nEpochs = NEPOCHS;
    int lateEpochs = (int)(nEpochs * 0.95) - 1; // taken from TMVA for better learning

    int NumberOfLayers = NUMBEROFLAYERS;
    int nEvents = NEVENTS;

    int lastLayer   = NumberOfLayers - 1;
    int lastNeurons = LASTNEURONS;

    double Neurons[NUMBEROFLAYERS][TOTALNEURONS];
    double deltas[NUMBEROFLAYERS][TOTALNEURONS];
    __local double Synweights[NUMBEROFLAYERS][TOTALNEURONS][TOTALNEURONS];

    double desired[LASTNEURONS];
    for (i = 0; i < lastNeurons; i++) {
        for (int nEv = 0; nEv < nEvents; nEv++) {
            if (eventClass[nEv] == i) {
                desired[i] = 1.0;
            } else {
                desired[i] = 0.0;
            }
        }
    }

    for (l = 0; l < NumberOfLayers - 1; l++) {
        for (i = 0; i < NeuronsPerLayer[l]; i++) {
            for (j = 0; j < NeuronsPerLayer[l + 1] - bias[l + 1]; j++) {
                Synweights[l][i][j] = Synweightsout[l+i+j];
            }
        }
    }
    double sumdeltas = 0.0; // Sum of delta_k*weight_jk used in back propagation

    //Create Vector of desired Values for back propagation

    int sum = 0;
    // allocate neurons


    // set neurons of bias nodes
    for (i = 0; i < NumberOfLayers - 1; i++) {
        if (bias[i] != 0) {
            Neurons[i][NeuronsPerLayer[i] - 1] = 1.0;
        }
    }
    sum += NeuronsPerLayer[NumberOfLayers - 1];

    // online learning
    for (int nEp = 0; nEp < nEpochs; nEp++) {     //for each epoch

        for (int nEv = 0; nEv < nEvents; nEv++) {   //for each event

            //Initialization for each event
            for (i = 0; i < NeuronsPerLayer[NumberOfLayers - 1]; i++) {
                //Set desired output to 1 if eventclass is the same
                // as the index of the output neuron
                if (eventClass[nEv] == i) {
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
                Neurons[0][i] = eventValues[nEv + i];
            }

            //forward propagation

            //For each layer except Input and Output Layer.
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
                deltas[l][j] *= eventWeights[nEv];
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

    }  // end loop over epochs

    for (l = 0; l < NumberOfLayers - 1; l++) {
        for (i = 0; i < NeuronsPerLayer[l]; i++) {
            for (j = 0; j < NeuronsPerLayer[l + 1] - bias[l + 1]; j++) {
                Synweightsout[l+i+j] = Synweights[l][i][j];
            }
        }
    }
    return;
}*/