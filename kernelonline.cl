#pragma OPENCL EXTENSION cl_khr_fp64: enable


kernel void CTrainMLP(global int* eventClass,global double* eventWeights,global double* eventValues,  double learnRate, int nVars, int nEpochs,
              int nEvents, global double* Synweights,
               global int *NeuronsPerLayer,  int NumberOfLayers, global int *bias,
               double decayRate,  double max,  double min, global double* desired, global double *deltas,global double *Neurons)
{
    int l, i, j, k;       //indices in for loops
    int lateEpochs = (int)(nEpochs * 0.95) - 1; // taken from TMVA for better learning

    double sumdeltas = 0.0; // Sum of delta_k*weight_jk used in back propagation

    //Create Vector of desired Values for back propagation
    
    int totalneurons[4];
    int sum=0;
    // allocate neurons
    

    // set neurons of bias nodes
    for (i = 0; i < NumberOfLayers - 1; i++) {
        sum+=NeuronsPerLayer[i];
        totalneurons[i]=sum;
        if (bias[i] != 0) {
            Neurons[totalneurons[i] - 1] = 1.0;
        }
    }
    sum+=NeuronsPerLayer[NumberOfLayers-1];
    totalneurons[4]=sum;

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
                Neurons[0][i] = eventValues[nEv][i];
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

    } // end loop over epochs
    return;
}