#include "C_TrainMLP.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <CL/cl.h>
#include <sstream>




#define FUNCTION  tanh     //function
#define DERIVATE tanhd     //derivate of function

using namespace std;

double tanhd(double value){
	return 1.0-value*value; 
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
 ******************************************************************************/
void CTrainMLP(CEvents* ev, double learnRate, int nVars, int nEpochs, 
		    int nEvents, double*** Synweights,
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

  cout << "function CTrainMLP" << endl;

  // allocate neurons
  double** Neurons;
  int totalNeurons = NeuronsPerLayer[0];
  for(l=1;l<NumberOfLayers;l++){
    totalNeurons +=NeuronsPerLayer[l];
  }
  Neurons=(double**)malloc(NumberOfLayers*sizeof(double*));
  Neurons[0] = (double*) malloc(totalNeurons*sizeof(double));
  for(l=1;l<NumberOfLayers;l++){
    Neurons[l]= Neurons[l-1] + NeuronsPerLayer[l-1];
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
	//Use Eventvalues as output of the input layer to make 
	//the next step easier.
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
	    //Calculate the output as f_act(Input)
	    Neurons[l][j]=FUNCTION(Neurons[l][j]); 
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
 *		              Index 1 is the layer starting at the weights between 
 *                                 the inputlayer and the first hiddenlayer.
 *			      Index 2 is the neuron of the lower layer.
 *			      Index 3 is the neuron of the upper layer.
 *@param[in] NeuronsPerLayer: Array that contains the number of neurons per layer.
 *@param[in] NumberOfLayers:  Number of Layers in the neural network.
 *@param[in] bias:            Number of bias nodes per layer
 *@param[in] decayRat:        Parameter for changing the learing rate
 *
 *@return 
 ******************************************************************************/
void CTrainMLP_b(CEvents* ev, double learnRate, int nVars, int nEpochs, 
		    int nEvents, double*** Synweights,
		    int* NeuronsPerLayer, int NumberOfLayers, int * bias, 
		    double decayRate){
	
  int l,i,j,k;          //indices in for loops
  int lateEpochs= (int)(nEpochs*0.95) - 1;  // taken from TMVA for better learning

  double sumdeltas=0.0;	// Sum of delta_k*weight_jk used in back propagation

  int totalNeurons = NeuronsPerLayer[0];
  for(l=1;l<NumberOfLayers;l++){
    totalNeurons +=NeuronsPerLayer[l];
  }
  cout << "total number of neurons:" << totalNeurons << endl;

  cout << "function CTrainMLP_b" << endl;

  double*** deltasEvents;// deltas, used in back propagation, 
                         // index1: Layer index2: Neuron per layer, index3: event
  deltasEvents = (double***) malloc(NumberOfLayers*sizeof(double**));
  for (l=0;l<NumberOfLayers;l++){
    deltasEvents[l] =  (double**) malloc(NeuronsPerLayer[l]*sizeof(double*));
  }
  for (l=0;l<NumberOfLayers;l++){
    for(i=0;i<NeuronsPerLayer[l];i++){
      deltasEvents[l][i] = (double*) malloc(nEvents*sizeof(double));
    }
  }

  double*** neuronsEvents;// value of the neurons
                          // index1: Layer index2: Neuron per layer, index3: event
  neuronsEvents = (double***) malloc(NumberOfLayers*sizeof(double**));
  for (l=0;l<NumberOfLayers;l++){
    neuronsEvents[l] =  (double**) malloc(NeuronsPerLayer[l]*sizeof(double*));
  }
  for (l=0;l<NumberOfLayers;l++){
    for(i=0;i<NeuronsPerLayer[l];i++){
      neuronsEvents[l][i] = (double*) malloc(nEvents*sizeof(double));
    }
  }

  // set neurons of bias nodes
  for (i=0;i<NumberOfLayers-1;i++){
    if (bias[i] != 0) {
      for(int nEv=0;nEv<nEvents;nEv++){
 	neuronsEvents[i][NeuronsPerLayer[i]-1][nEv]=1.0;
      }
    }
  }
  
  // initialization for each event the input layer for batch learning
  for(i=0;i<NeuronsPerLayer[0]-bias[0];i++){ 
    //Use Eventvalues as output of the input layer to make the next step easier.
    for(int nEv=0;nEv<nEvents;nEv++) {
      neuronsEvents[0][i][nEv]=ev->eventValues[nEv][i];
    }
  }
  
  // restore the desired output values
  int lastLayer   = NumberOfLayers-1;
  int lastNeurons = NeuronsPerLayer[lastLayer];
  double desired[nEvents][lastNeurons];
  for (int nEv=0; nEv<nEvents; nEv++) {
    for(i=0; i<lastNeurons; i++){ 
      if(ev->eventClass[nEv]==i){
	desired[nEv][i]=1.0;
      }
      else{
	desired[nEv][i]=0.0;
      }
    }
  }


  for(int nEp=0;nEp<nEpochs;nEp++){    			//for each epoch

    // batch learning

    //for all events do forward propagation	
    for(int nEv=0;nEv<nEvents;nEv++){			//for each event

      //For each layer except Input and Output Layer. 
      for(l=1;l<NumberOfLayers-1;l++){ 

	// For each neuron on the next layer except bias node
	for(j=0;j<NeuronsPerLayer[l]-bias[l];j++){ 

	  // Calculate the neuron input
	  neuronsEvents[l][j][nEv]=0.0;
	  // For each input coming from the lower layer
	  for(i=0;i<NeuronsPerLayer[l-1];i++)
	    // Calculate the neuron input
	    neuronsEvents[l][j][nEv]+=neuronsEvents[l-1][i][nEv]*Synweights[l-1][i][j];

	  //Calculate the output as f_act(Input)
	  neuronsEvents[l][j][nEv]=FUNCTION( neuronsEvents[l][j][nEv]);
	}
      }
	
      // for the output layer, no activation funktion is applied
      l = NumberOfLayers-1;
      for(j=0;j<NeuronsPerLayer[l]-bias[l];j++){ 
	neuronsEvents[l][j][nEv]=0.0;
	for(i=0;i<NeuronsPerLayer[l-1];i++){
	  neuronsEvents[l][j][nEv]+=neuronsEvents[l-1][i][nEv]*Synweights[l-1][i][j];
	}
      }

      }

    // now, for all events do backward propagation	
    
    // first compute all deltas
    for(int nEv=0;nEv<nEvents;nEv++){			//for each event

      // output layer
      l=NumberOfLayers-1;
      for (j=0;j<NeuronsPerLayer[l];j++){    
	// Calculate delta. Since the Output is linear, there is no need 
	// to calculate the derivate of the activation function
	// here the deltas have to be multiplied with the weight of the event
	deltasEvents[l][j][nEv] = neuronsEvents[l][j][nEv] - desired[nEv][j];	
	deltasEvents[l][j][nEv]*= ev->eventWeights[nEv];
      }

      //Beginning from last layer where the next layer is hiden layer
      for (l=NumberOfLayers-2;l>=0;l--){
	//for every Neuron on the current Layer
	for(j=0;j<NeuronsPerLayer[l];j++){
	  sumdeltas=0.0;
	  //for every Neuron on the next higher Layer
	  for(k=0;k<NeuronsPerLayer[l+1]-bias[l+1];k++){
	    //Calculate delta_k*w_jk to calculate the new deltas
	    sumdeltas+=deltasEvents[l+1][k][nEv]*Synweights[l][j][k];
	  }
	  //Calculate delta for current layer
	  deltasEvents[l][j][nEv]=DERIVATE(neuronsEvents[l][j][nEv])*sumdeltas;
	} // end loop NeuronsPerLayer
      }	  // end loop NumberOfLayers

    }

    // secondly update weights
    
    //For all Layers, upate Synapse weight
    for(l=0;l<NumberOfLayers-1;l++){
      for(j=0;j<NeuronsPerLayer[l+1]-bias[l+1];j++){
	for(i=0;i<NeuronsPerLayer[l];i++){
	  sumdeltas=0.0;
	  for(int nEv=0;nEv<nEvents;nEv++){			//for each event
	    sumdeltas+=neuronsEvents[l][i][nEv]*deltasEvents[l+1][j][nEv];
	  }
	  Synweights[l][i][j]+=-learnRate*sumdeltas/(double)nEvents;     
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

  return;
}

/**
 ******************************************************************************
 * function 
 * double*** CTrainMLP_m(CEvents* ev, double learnRate, int nVars, int nEpochs, 
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
 *		              Index 1 is the layer starting at the weights between 
 *                                 the inputlayer and the first hiddenlayer.
 *			      Index 2 is the neuron of the lower layer.
 *			      Index 3 is the neuron of the upper layer.
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
void CTrainMLP_m(CEvents* ev, double learnRate, int nVars, int nEpochs, 
		 int nEvents, double*** Synweights,
		 int* NeuronsPerLayer, int NumberOfLayers, int * bias, 
		 double decayRate, int events){
	
  int l,i,j,k; 
  int nEv;
  int lateEpochs= (int)(nEpochs*0.95) - 1;  // from TMVA for better learning

  double sumdeltas;	// Sum of delta_k*weight_jk used in back propagation

  int totalNeurons = NeuronsPerLayer[0];
  for(l=1;l<NumberOfLayers;l++){
    totalNeurons +=NeuronsPerLayer[l];
  }
  cout << "total number of neurons:" << totalNeurons << endl;
  cout << "function CTrainMLP_m" << endl;

  double*** deltasEvents;// deltas, used in back propagation, 
                         // index 1: Layer, 2: Neuron per layer, 3: event
  deltasEvents = (double***) malloc(NumberOfLayers*sizeof(double**));
  for (l=0;l<NumberOfLayers;l++){
    deltasEvents[l] =  (double**) malloc(NeuronsPerLayer[l]*sizeof(double*));
  }
  for (l=0; l<NumberOfLayers; l++){
    for (i=0; i<NeuronsPerLayer[l]; i++){
      deltasEvents[l][i] = (double*) malloc(nEvents*sizeof(double));
    }
  }

  double*** neuronsEvents;// value of the neurons
                          // index 1: Layer, 2: Neuron per layer, 3: event
  neuronsEvents = (double***) malloc(NumberOfLayers*sizeof(double**));
  for (l=0; l<NumberOfLayers; l++){
    neuronsEvents[l] =  (double**) malloc(NeuronsPerLayer[l]*sizeof(double*));
  }
  for (l=0; l<NumberOfLayers; l++){
    for (i=0; i<NeuronsPerLayer[l]; i++){
      neuronsEvents[l][i] = (double*) malloc(nEvents*sizeof(double));
    }
  }

  // set neurons of bias nodes
  for (i=0; i<NumberOfLayers-1; i++){
    if (bias[i] != 0) {
      for(nEv=0; nEv<nEvents; nEv++){
 	neuronsEvents[i][NeuronsPerLayer[i]-1][nEv]=1.0;
      }
    }
  }
  
  //Initialization for each event the input layer for batch learning
  for (i=0; i<NeuronsPerLayer[0]-bias[0]; i++){ 
    for (nEv=0; nEv<nEvents; nEv++) {
      neuronsEvents[0][i][nEv]=ev->eventValues[nEv][i];
    }
  }

  // restore the desired output values
  int lastLayer   = NumberOfLayers-1;
  int lastNeurons = NeuronsPerLayer[lastLayer];
  double desired[nEvents][lastNeurons];
  for (int nEv=0; nEv<nEvents; nEv++) {
    for(i=0; i<lastNeurons; i++){ 
      if(ev->eventClass[nEv]==i){
	desired[nEv][i]=1.0;
      }
      else{
	desired[nEv][i]=0.0;
      }
    }
  }
  
  // number of loops for mixed batch - online learning
  int parts = nEvents/events;

  for(int nEp=0; nEp<nEpochs; nEp++){    			//for each epoch

    // do mixed learning, i.e. learn in iparts batches of size events
    int nEv_start = nEv = 0; 
    int nEv_stop = events;
    for (int iparts = 0; iparts < parts; iparts++) {

      //for one batch of events do forward propagation	
      //here: calculate the output as f_act(Input) not for the output layer
      for (nEv=nEv_start; nEv<nEv_stop; nEv++) {
	for(l=1;l<NumberOfLayers-1;l++){ 
	  for(j=0;j<NeuronsPerLayer[l]-bias[l];j++){ 
	    neuronsEvents[l][j][nEv]=0.0;
	    for(i=0;i<NeuronsPerLayer[l-1];i++)
	      neuronsEvents[l][j][nEv] += neuronsEvents[l-1][i][nEv]
		*Synweights[l-1][i][j];
	    neuronsEvents[l][j][nEv]=FUNCTION( neuronsEvents[l][j][nEv]);
	  }
	}

	l = NumberOfLayers-1;
	for(j=0;j<NeuronsPerLayer[l]-bias[l];j++){ 
	  neuronsEvents[l][j][nEv]=0.0;
	  for(i=0;i<NeuronsPerLayer[l-1];i++){
	    neuronsEvents[l][j][nEv]+=neuronsEvents[l-1][i][nEv]
	      *Synweights[l-1][i][j];
	  }
	}
      }  // end loop over one batch

      // now, for one batch of events do backward propagation	
      for ( nEv=nEv_start; nEv<nEv_stop; nEv++) {
	
	// first compute all deltas

	// output layer
	l=NumberOfLayers-1;
	for (j=0;j<NeuronsPerLayer[l];j++){    
	  // Calculate delta. Since the Output is linear, there is no need 
	  // to calculate the derivate of the activation function
	  // here the deltas have to be multiplied with the weight of the event
	  deltasEvents[l][j][nEv] = neuronsEvents[l][j][nEv] - desired[nEv][j];	
	  deltasEvents[l][j][nEv]*= ev->eventWeights[nEv];
	}
	
	//Beginning from last layer where the next layer is hiden layer
	for (l=NumberOfLayers-2;l>=0;l--){
	  //for every Neuron on the current Layer
	  for(j=0;j<NeuronsPerLayer[l];j++){
	    sumdeltas=0.0;
	    //for every Neuron on the next higher Layer
	    for(k=0;k<NeuronsPerLayer[l+1]-bias[l+1];k++){
	      //Calculate delta_k*w_jk to calculate the new deltas
	      sumdeltas+=deltasEvents[l+1][k][nEv]*Synweights[l][j][k];
	    }
	    //Calculate delta for current layer
	    deltasEvents[l][j][nEv]=DERIVATE(neuronsEvents[l][j][nEv])*sumdeltas;
	  } // end loop NeuronsPerLayer
	}   // end loop NumberOfLayers	
      }     // end loop over one batch

      // secondly update weights
    
      //For all Layers, upate Synapse weight
      for(l=0;l<NumberOfLayers-1;l++){
	for(j=0;j<NeuronsPerLayer[l+1]-bias[l+1];j++){
	  for(i=0;i<NeuronsPerLayer[l];i++){
	    sumdeltas=0.0;
	    for (nEv=nEv_start; nEv<nEv_stop; nEv++) {
	      sumdeltas+=neuronsEvents[l][i][nEv]*deltasEvents[l+1][j][nEv];
	    }
	    Synweights[l][i][j]+=-learnRate*sumdeltas/(double)events;     
	  }
	}
      } // end loop over layers
      
      nEv_start += events;
      nEv_stop  += events;

    } // end loop over parts
    
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

  return;
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
/**
 ***************************************************************************************************
 *Creates context for openCL
 *@param[out] context: Created context
 ***************************************************************************************************/

cl_context CTrainMLP_CreateContext(){
  cl_int errNum;
  cl_uint numPlatforms;
  cl_platform_id firstPlatformId;
  cl_context context=NULL;
  //get Platform and choose first one
  errNum = clGetPlatformIDs(1,&firstPlatformId, &numPlatforms);
  if(errNum != CL_SUCCESS || numPlatforms<=0){
    cerr<<"No OpenCL platforum found!"<<endl;
    return NULL;
  }
  cl_context_properties contextProperties[]={
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)firstPlatformId,
    0
  };
  context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,NULL,NULL,&errNum);
  if (errNum!= CL_SUCCESS){
    cout<<"Unable to create GPU context, try CPU..."<<endl;
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,NULL,NULL,&errNum);
     if (errNum!= CL_SUCCESS){
      cerr<<"Unable to create GPU or CPU context"<<endl;
      return NULL;
     }
  }
  cout<<"Created GPU context"<<endl;
  return context;
}

/**
***********************************************************************************************
*Creates Command queue for opencl
*@param[in] context: Context needed for Comannd queue
*@param[in] device: Device ID
*param[out] comandQueue: Created Command queue for specified context
***********************************************************************************************/
cl_command_queue CTrainMLP_CreateCommandQueue(cl_context context, cl_device_id *device){
  cl_int errNum;
  cl_device_id *devices;
  cl_command_queue commandQueue = NULL;
  size_t deviceBufferSize = -1;

  errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES,0,NULL,&deviceBufferSize);
  if(errNum!=CL_SUCCESS){
    cerr<<"Failed to get size of device buffer";
    return NULL;
  }
  if(deviceBufferSize<=0){
    cerr<<"No devices available";
    return NULL;
  }
  //Allocate memory for device buffer
  devices= new cl_device_id[deviceBufferSize/sizeof(cl_device_id)];
  errNum= clGetContextInfo(context,CL_CONTEXT_DEVICES,deviceBufferSize,devices, NULL);
  if(errNum!=CL_SUCCESS){
    cerr<<"Failed to get device ID";
    return NULL;
  }
  commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
  if(commandQueue==NULL){
    cerr<<"Failed to create command queue";
    return NULL;
  }
  *device = devices[0];
  delete [] devices;
  return commandQueue;
}


cl_program CTrainMLP_CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
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