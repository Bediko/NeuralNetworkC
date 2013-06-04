#include "C_TrainMLP.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>



#define FUNCTION  tanh     //function
#define DERIVATE tanhd     //derivate of function

using namespace std;

double tanhd(double value){
	return 1.0-(pow(value,2)); 
}

/**
 ******************************************************************************
 * function 
 * double*** CTrainMLP(CEvents* ev, double learnRate, int nVars, int nEpochs, 
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
 *		              Index 1 is the layer starting at the weights between 
 *                                 the inputlayer and the first hiddenlayer.
 *			      Index 2 is the neuron of the lower layer.
 *			      Index 3 is the neuron of the upper layer.
 *@param[in] Neurons:         Matrix for the outputvalue of each neuron. 
 *			      Index 1 is the layer. 
 *			      Index 2 is the neuron on the layer.
 *@param[in] NeuronsPerLayer: Array that contains the number of neurons per layer.
 *@param[in] NumberOfLayers:  Number of Layers in the neural network.
 *@param[in] bias:            Number of bias nodes per layer
 *@param[in] decayRat:        Parameter for changing the learing rate
 *@param[in] max:             set to 1.0, used for activation function on the last layer
 *@param[in] min:             set to 0.0, used for activation function on the last layer
 *
 *@return Synweights 
 ******************************************************************************/
double*** CTrainMLP(CEvents* ev, double learnRate, int nVars, int nEpochs, 
		    int nEvents, double*** Synweights, double** Neurons,
		    int* NeuronsPerLayer, int NumberOfLayers, int * bias, 
		    double decayRate, double max, double min){
	
  int l,i,j,k;          //indices in for loops
  int lateEpochs= (int)(nEpochs*0.95) - 1;  // taken from TMVA for better learning

  double sumdeltas=0.0;	// Sum of delta_k*weight_jk used in back propagation

  //Create Vector of desired Values for back propagation
  double* desired;	// desired output value for each output neuron.
  desired = (double*) malloc(NeuronsPerLayer[NumberOfLayers-1]*sizeof(double));

  double** deltas;	// deltas, used in back propagation, index1: Layer index2: Neuron
  deltas = (double**) malloc(NumberOfLayers*sizeof(double*));
  for(l=0;l<NumberOfLayers;l++){
    deltas[l]=(double*)malloc(NeuronsPerLayer[l]*sizeof(double));
  }

  // set neurons of bias nodes
  for(i=0;i<NumberOfLayers-1;i++){
    if (bias[i] != 0) {
      Neurons[i][NeuronsPerLayer[i]-1]=1.0;
    }
  }

  // online learning
  for(int nEp=0;nEp<nEpochs;nEp++){    			//for each epoch

    for(int nEv=0;nEv<nEvents;nEv++){			//for each event

      //Initialization for each event     
      for(i=0;i<NeuronsPerLayer[NumberOfLayers-1];i++){ 

	//Set desired output to 1 if eventclass is the same 
	// as the index of the output neuron
	if(ev->eventClass[nEv]==i){
	  desired[i]=1.0;
	}
	else{
	  desired[i]=0.0;
	}
	
      }
	
      // aus eventValue bias-Knoten wieder raus und for-loop nur 
      // bis < NeuronsPerLayer[0]-1
      for(i=0;i<NeuronsPerLayer[0]-bias[0];i++){ 
	//Use Eventvalues as output of the input layer to make the next step easier.
	Neurons[0][i]=ev->eventValues[nEv][i];
      }
      
      //forward propagation	
      
      //For each layer except Input and Output Layer. 
      for(l=1;l<NumberOfLayers;l++){

	// For each neuron on the next layer except bias node
	for(j=0;j<NeuronsPerLayer[l]-bias[l];j++){ 

	  // Calculate the neuron input
	  Neurons[l][j]=0.0;
	  // For each input coming from the lower layer
	  for(i=0;i<NeuronsPerLayer[l-1];i++){
	    // Calculate the neuron input
	    Neurons[l][j]+=Neurons[l-1][i]*Synweights[l-1][i][j];
	  }

	  // decide if current layer is Output
	  if(l==NumberOfLayers-1)
	    continue;
	  else
	    Neurons[l][j]=FUNCTION(Neurons[l][j]); //Calculate the output as f_act(Input)
	}
      }
		

      // backward

      // output layer
      l=NumberOfLayers-1;
      for (j=0;j<NeuronsPerLayer[l];j++){    
	// Calculate delta. Since the Output is linear, there is no need 
	// to calculate the derivate of the activation function
	// here the deltas have to be multiplied with the weight of the event
	deltas[l][j]=Neurons[l][j]-desired[j];	
	deltas[l][j]*=ev->eventWeights[nEv];
      }

      //Beginning from last layer where the next layer is hiden layer
      for (l=NumberOfLayers-2;l>=0;l--){
	//for every Neuron on the current Layer
	for(j=0;j<NeuronsPerLayer[l];j++){
	  sumdeltas=0.0;
	  //for every Neuron on the next higher Layer
	  for(k=0;k<NeuronsPerLayer[l+1]-bias[l+1];k++){
	    //Calculate delta_k*w_jk to calculate the new deltas
	    sumdeltas+=deltas[l+1][k]*Synweights[l][j][k];
	  }
	  //Calculate delta for current layer
	  deltas[l][j]=DERIVATE(Neurons[l][j])*sumdeltas;
	} // end loop NeuronsPerLayer
      }	  // end loop NumberOfLayers
    
      //For all Layers, upate Synapse weight
      for(l=0;l<NumberOfLayers-1;l++){
	for(j=0;j<NeuronsPerLayer[l+1]-bias[l+1];j++){
	  for(i=0;i<NeuronsPerLayer[l];i++){
	    Synweights[l][i][j]+=-learnRate*Neurons[l][i]*deltas[l+1][j];     
	  }
	}
      }
		
    } // end loop over events

    //reduce learnrate to get better minimum
    //check if we are in late epochs
    if(nEp>=lateEpochs){
      // In order to lower the learning rate even more, 
      learnRate *= (1.0 - sqrt(decayRate));
    }
    else{
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

  return Synweights;
}

/**
 ******************************************************************************
 * function 
 * double*** CTrainMLP_b(CEvents* ev, double learnRate, int nVars, int nEpochs, 
		    int nEvents, double*** Synweights, double** Neurons,
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
 *		              Index 1 is the layer starting at the weights between 
 *                                 the inputlayer and the first hiddenlayer.
 *			      Index 2 is the neuron of the lower layer.
 *			      Index 3 is the neuron of the upper layer.
 *@param[in] Neurons:         Matrix for the outputvalue of each neuron. 
 *			      Index 1 is the layer. 
 *			      Index 2 is the neuron on the layer.
 *@param[in] NeuronsPerLayer: Array that contains the number of neurons per layer.
 *@param[in] NumberOfLayers:  Number of Layers in the neural network.
 *@param[in] bias:            Number of bias nodes per layer
 *@param[in] decayRat:        Parameter for changing the learing rate
 *@param[in] max:             set to 1.0, used for activation function on the last layer
 *@param[in] min:             set to 0.0, used for activation function on the last layer
 *
 *@return Synweights 
 ******************************************************************************/
double*** CTrainMLP_b(CEvents* ev, double learnRate, int nVars, int nEpochs, 
		    int nEvents, double*** Synweights, double** Neurons,
		    int* NeuronsPerLayer, int NumberOfLayers, int * bias, 
		    double decayRate, double max, double min){
	
  int l,i,j,k;          //indices in for loops
  int lateEpochs= (int)(nEpochs*0.95) - 1;  // taken from TMVA for better learning

  double sumdeltas=0.0;	// Sum of delta_k*weight_jk used in back propagation

  double** deltas;	// deltas, used in back propagation, 
                        // index1: Layer index2: Neuron
  deltas = (double**) malloc(NumberOfLayers*sizeof(double*));
  int totalNeurons = NeuronsPerLayer[0];
  for(l=1;l<NumberOfLayers;l++){
    totalNeurons +=NeuronsPerLayer[l];
  }
  cout << "total number of neurons:" << totalNeurons << endl;
  deltas[0] = (double*) malloc(totalNeurons*sizeof(double));

  for(l=1;l<NumberOfLayers;l++){
    deltas[l]= deltas[l-1] + NeuronsPerLayer[l-1];
  }

  // set neurons of bias nodes
  for(i=0;i<NumberOfLayers-1;i++){
    if (bias[i] != 0) {
      Neurons[i][NeuronsPerLayer[i]-1]=1.0;
    }
  }

  // for batch learning: integrate as second index the epoch number
  // in the delta values

  double*** deltasEpoch;// deltas, used in back propagation, 
                        // index1: Layer index2: Neuron per layer, index3: epochs
  deltasEpoch = (double***) malloc(NumberOfLayers*sizeof(double**));

  for (l=0;l<NumberOfLayers;l++){
    deltasEpoch[l] =  (double**) malloc(NeuronsPerLayer[l]*sizeof(double*));
  }

  for (l=0;l<NumberOfLayers;l++){
    for(i=0;i<NeuronsPerLayer[l];i++){
      deltasEpoch[l][i] = (double*) malloc(nEpochs*sizeof(double));
    }
  }


  // batch learning
  for(int nEp=0;nEp<nEpochs;nEp++){    			//for each epoch

    for(int nEv=0;nEv<nEvents;nEv++){			//for each event

      //Initialization for each event the input layer
      for(i=0;i<NeuronsPerLayer[0]-bias[0];i++){ 
	//Use Eventvalues as output of the input layer to make the next step easier.
	Neurons[0][i]=ev->eventValues[nEv][i];
      }
      
      //forward propagation	
      
      //For each layer except Input and Output Layer. 
      for(l=1;l<NumberOfLayers;l++){

	// For each neuron on the next layer except bias node
	for(j=0;j<NeuronsPerLayer[l]-bias[l];j++){ 

	  // Calculate the neuron input
	  Neurons[l][j]=0.0;
	  // For each input coming from the lower layer
	  for(i=0;i<NeuronsPerLayer[l-1];i++){
	    // Calculate the neuron input
	    Neurons[l][j]+=Neurons[l-1][i]*Synweights[l-1][i][j];
	  }

	  // decide if current layer is Output
	  if(l==NumberOfLayers-1)
	    continue;
	  else
	    Neurons[l][j]=FUNCTION(Neurons[l][j]); //Calculate the output as f_act(Input)
	}
      }
		

      // backward

      // output layer
      l=NumberOfLayers-1;
      for (j=0;j<NeuronsPerLayer[l];j++){    
	// Calculate delta. Since the Output is linear, there is no need 
	// to calculate the derivate of the activation function
	// here the deltas have to be multiplied with the weight of the event
	deltasEpoch[l][j][nEv]=Neurons[l][j]-ev->desired[nEv][j];	
	deltasEpoch[l][j][nEv]*=ev->eventWeights[nEv];
      }

      //Beginning from last layer where the next layer is hiden layer
      for (l=NumberOfLayers-2;l>=0;l--){
	//for every Neuron on the current Layer
	for(j=0;j<NeuronsPerLayer[l];j++){
	  sumdeltas=0.0;
	  //for every Neuron on the next higher Layer
	  for(k=0;k<NeuronsPerLayer[l+1]-bias[l+1];k++){
	    //Calculate delta_k*w_jk to calculate the new deltas
	    sumdeltas+=deltasEpoch[l+1][k][nEv]*Synweights[l][j][k];
	  }
	  //Calculate delta for current layer
	  deltasEpoch[l][j][nEv]=DERIVATE(Neurons[l][j])*sumdeltas;
	} // end loop NeuronsPerLayer
      }	  // end loop NumberOfLayers
    
      //For all Layers, upate Synapse weight
      for(l=0;l<NumberOfLayers-1;l++){
	for(j=0;j<NeuronsPerLayer[l+1]-bias[l+1];j++){
	  for(i=0;i<NeuronsPerLayer[l];i++){
	    Synweights[l][i][j]+=-learnRate*Neurons[l][i]*deltasEpoch[l+1][j][nEv];     
	  }
	}
      }
		
    } // end loop over events

    //reduce learnrate to get better minimum
    //check if we are in late epochs
    if(nEp>=lateEpochs){
      // In order to lower the learning rate even more, 
      learnRate *= (1.0 - sqrt(decayRate));
    }
    else{
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

  return Synweights;
}

/**
 ******************************************************************************
 * function 
 * double** CTrainMLP_testing(CEvents* ev,int nEpochs,int nEvents, double*** Synweights, 
 *			   double** Neurons, int* NeuronsPerLayer, int NumberOfLayers,
 *			   int * bias, double** testout){
 *
 *Test the neural network 
 *
 *@param[in] ev               Structure with data of the events for training. 
                              Includes the class, the weight and the values of the events
 *@param[in] nEpochs          Number of Epochs to train.
 *@param[in] nEvents          Number of Events to train.
 *@param[in] Synweights:      3-Dimensional array for Synapseweights. 
 *		              Index 1 is the layer starting at the weights 
 *                            between the inputlayer 
 *                                 and the first hiddenlayer.
 *			      Index 2 is the neuron of the lower layer.
 *			      Index 3 is the neuron of the upper layer.
 *@param[in] Neurons:         Matrix for the outputvalue of each neuron. 
 *			      Index 1 is the layer. 
 *			      Index 2 is the neuron on the layer.
 *@param[in] NeuronsPerLayer: Array that contains the number of neurons per layer.
 *@param[in] NumberOfLayers:  Number of Layers in the neural network.
 *@param[in] bias:            Number of bias nodes per layer
 *@param[out] testout:        Results of the test pattern
 *
 ******************************************************************************/
void CTrainMLP_testing(CEvents* ev, int nEpochs, int nEvents, double*** Synweights, 
			   double** Neurons, int* NeuronsPerLayer, int NumberOfLayers,
			   int * bias, double** testout){
	
  int l,i,j;   //indices in for loops
  
  // set neurons of bias nodes
  for(i=0;i<NumberOfLayers-1;i++){
    if (bias[i] != 0) {
      Neurons[i][NeuronsPerLayer[i]-1]=1.0;
    }
  }

  //for each event
  for(int nEv=0;nEv<nEvents;nEv++){
			
    // aus eventValue bias-Knoten wieder raus und for-loop nur bis < NeuronsPerLayer[0]-1
    for(i=0;i<NeuronsPerLayer[0]-bias[0];i++){	
      //Use Eventvalues as output of the input layer to make the next step easier.
      Neurons[0][i]=ev->eventValues[nEv][i];
    }
			
    //forward propagation	

    //For each layer except Input and Output Layer. 
    for(l=1;l<NumberOfLayers;l++){ 
      //For each neuron on the next layer except bias node
      for(j=0;j<NeuronsPerLayer[l]-bias[l];j++){
	Neurons[l][j]=0.0;
	//For each input coming from the lower layer
	for(i=0;i<NeuronsPerLayer[l-1];i++){
	  //Calculate the neuron input							
	  Neurons[l][j]+=Neurons[l-1][i]*Synweights[l-1][i][j];
	}
	
	if(l==NumberOfLayers-1)
	  //decide if current layer is Output
	  continue;
	else
	  //Calculate the output as f_act(Input)				
	  Neurons[l][j]=FUNCTION(Neurons[l][j]);
      }
    }
    for(i=0;i<NeuronsPerLayer[NumberOfLayers-1];i++){
      testout[nEv][i]=Neurons[NumberOfLayers-1][i];
    }
  }		
  return;
}
