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
*Trains the neural network 
*
*@param[in] ev Structure with data of the events for training. Includes the class, the weight and the values of the events
*@param[in] learnRate The learning rate for the neural network.
*@param[in] nVars Number of Inputvariables + Bias Node
*@param[in] nSynapses Number of Synapses in the neurl network
*@param[in]	nEpochs Number of Epochs to train.
*@param[in] nEvents Number of Events to train.
*@param[in,out] Synweights 3-Dimensional array for Synapseweights. 
*													 Index 1 is the layer starting at the weights between the inputlayer and the first hiddenlayer.
*													 Index 2 is the neuron of the lower layer.
*													 Index 3 is the neuron of the upper layer.
*@param[in] Neurons Matrix for the outputvalue of each neuron. 
*						Index 1 is the layer. 
*						Index 2 is the neuron on the layer.
*@param[in] NeuronsPerLayer Array that contains the number of neurons per layer.
*@param[in] NumberOfLayers Number of Layers in the neural network.
*/
double*** CTrainMLP(CEvents* ev, double learnRate, int nVars, int nEpochs, 
		int nEvents, double*** Synweights, double** Neurons,int* NeuronsPerLayer, int NumberOfLayers, double decayRate,double max, double min){
	
	int* bias=(int*)malloc(NumberOfLayers*sizeof(int));
	double* desired;																														// holds the desired output value for each output neuron.
	double** deltas;																														//Matrix for calculated deltas used in back propagation. index1: Layer index2: Neuron
	double sumdeltas=0.0;																														//Sum of delta_k*weight_jk used in back propagation
	int lateEpochs= (int)(nEpochs*0.95) - 1;
	desired = (double*)malloc(NeuronsPerLayer[NumberOfLayers-1]*sizeof(double));//Create Vector of desired Values for back propagation
	int l,i,j,k;
	int a;																																//indices in for loops																															//Number of neurons on the biggest Layer
	deltas=(double**)malloc(NumberOfLayers*sizeof(double*));
	
	for(l=0;l<NumberOfLayers;l++){																							//Get Space for deltas, each Neuron has a delta
		deltas[l]=(double*)malloc(NeuronsPerLayer[l]*sizeof(double));
	}

	bias[NumberOfLayers-1]=0;

	for(i=0;i<NumberOfLayers-1;i++){ 																						//for each Layer except the Output Layer
		Neurons[i][NeuronsPerLayer[i]-1]=1.0;
		bias[i]=1; 																			//Set Bias Node to 1
	}

	for(int nEp=0;nEp<nEpochs;nEp++){																						//for each epoch
		for(int nEv=0;nEv<nEvents;nEv++){																					//for each event

			//Initialization for each event
			

			for(i=0;i<NeuronsPerLayer[NumberOfLayers-1];i++){ 											//Set desired output to 1 if eventclass is the same as the index of the output neuron
				if(ev->eventClass[nEv]==i){
					desired[i]=1.0;
				}
				else{
					desired[i]=0.0;
				}
			}
			
			// aus eventValue bias-Knoten wieder raus und for-loop nur bis < NeuronsPerLayer[0]-1
			for(i=0;i<NeuronsPerLayer[0]-1;i++){ 																			//Use Eventvalues as output of the input layer to make the next step easier.
				Neurons[0][i]=ev->eventValues[nEv][i];
			}
			
			//forward propagation	
			
			for(l=1;l<NumberOfLayers;l++){ 																				//For each layer except Input and Output Layer. 
				for(j=0;j<NeuronsPerLayer[l]-bias[l];j++){ 														//For each neuron on the next layer except bias node
					Neurons[l][j]=0.0;																									//clear old output
					for(i=0;i<NeuronsPerLayer[l-1];i++){ 																//For each input coming from the lower layer
						Neurons[l][j]+=Neurons[l-1][i]*Synweights[l-1][i][j];							//Calculate the neuron input
					}
					if(l==NumberOfLayers-1)																							//decide if current layer is Output
						continue;										
					else																															
						Neurons[l][j]=FUNCTION(Neurons[l][j]);																//Calculate the output as f_act(Input)
				}
			}
		

			//backward
			l=NumberOfLayers-1;
			for (j=0;j<NeuronsPerLayer[l];j++){																	//On the output layer
				deltas[l][j]=Neurons[l][j]-desired[j];														//Calculate delta. Since the Output is linear there is no need to calculate the derivate of the activation function
				deltas[l][j]*=ev->eventWeights[nEv];
			}

			for (l=NumberOfLayers-2;l>=0;l--){																				//Beginning from last layer where the next layer is hiden layer
				for(j=0;j<NeuronsPerLayer[l];j++){																		//for every Neuron on the current Layer
					sumdeltas=0.0;
					for(k=0;k<NeuronsPerLayer[l+1]-bias[l+1];k++){																//for every Neuron on the next higher Layer
						sumdeltas+=deltas[l+1][k]*Synweights[l][j][k];										//Calculate delta_k*w_jk to calculate the new deltas
					}
					deltas[l][j]=DERIVATE(Neurons[l][j])*sumdeltas;												//Calculate delta for current layer
				}
			}	
																					

			for(l=0;l<NumberOfLayers-1;l++){																				//For all Layers 
				for(j=0;j<NeuronsPerLayer[l+1]-bias[l+1];j++){
					for(i=0;i<NeuronsPerLayer[l];i++){
						Synweights[l][i][j]+=-learnRate*Neurons[l][i]*deltas[l+1][j];     //Upate Synapse weight
					}
				}
			}
		
		}
		if(nEp>=lateEpochs){																											//check if we are in late epochs
			learnRate *= (1.0 - sqrt(decayRate));																			// In order to lower the learning rate even more, we need to apply sqrt instead of square.
		}
		else{
			learnRate *= (1.0 - decayRate);																						// //decay learnrate to get better minimum
		}
	}

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

double** CTrainMLP_testing(CEvents* ev, int nEpochs, 
		int nEvents, double*** Synweights, double** Neurons,int* NeuronsPerLayer, int NumberOfLayers,double** testout){
	
	int* bias=(int*)malloc(NumberOfLayers*sizeof(int));
	int l,i,j,k;																															//indices in for loops
	bias[NumberOfLayers-1]=0;
	for(i=0;i<NumberOfLayers-1;i++){ 																						//for each Layer except the Output Layer
		Neurons[i][NeuronsPerLayer[i]-1]=1.0;
		bias[i]=1; 																			//Set Bias Node to 1
	}
	for(int nEp=0;nEp<nEpochs;nEp++){																						//for each epoch
		for(int nEv=0;nEv<nEvents;nEv++){																					//for each event
			
			// aus eventValue bias-Knoten wieder raus und for-loop nur bis < NeuronsPerLayer[0]-1
			for(i=0;i<NeuronsPerLayer[0]-1;i++){ 																			//Use Eventvalues as output of the input layer to make the next step easier.
				Neurons[0][i]=ev->eventValues[nEv][i];
			}
			
			//forward propagation	
			
			for(l=1;l<NumberOfLayers;l++){ 																				//For each layer except Input and Output Layer. 
				for(j=0;j<NeuronsPerLayer[l]-bias[l];j++){ 														//For each neuron on the next layer except bias node
					Neurons[l][j]=0.0;																									//clear old output
					for(i=0;i<NeuronsPerLayer[l-1];i++){ 																//For each input coming from the lower layer
						Neurons[l][j]+=Neurons[l-1][i]*Synweights[l-1][i][j];							//Calculate the neuron input
					}
					if(l==NumberOfLayers-1)																							//decide if current layer is Output
						continue;										
					else																															
						Neurons[l][j]=FUNCTION(Neurons[l][j]);																//Calculate the output as f_act(Input)
				}
			}
		}
	}
}