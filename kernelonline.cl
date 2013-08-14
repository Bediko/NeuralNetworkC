
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
__kernel void CTrainMLP_forward_tanh(__global double *Neurons0, __global double *Synapses0, __global double *Synapses1, __global double *Synapses2,
                                    __global double *desired, __global double *weights,
                                     int nEv_begin, int nEv_end,double rate)
{

    // int bEv = get_group_id(0);
    // int bj = get_group_id(1);
    // int tEv=get_local_id(0);
    // int tj= get_local_id(1);

    int nEv = get_global_id(0);
    int block_id = get_local_id(0);
    int j = get_global_id(1);
    // NeuronsPerLayer[0]=get_num_groups(0);
    // NeuronsPerLayer[1]=get_num_groups(1);
    // return;
    // int nBegin=nEvents*BLOCK_SIZE*bj;
    // int nEnd=nBegin+nEvents-1;
    // int nStep= BLOCK_SIZE;
    // int jBegin=BLOCK_SIZE*bj;
    // int jStep=BLOCK_SIZE*NEURONS0;

    double tmp;
    //8=blocksize
    __local double n0[BLOCK_SIZE][NEURONS0];//Neurons
    __local double n1[BLOCK_SIZE][NEURONS1];
    __local double n2[BLOCK_SIZE][NEURONS2];
    __local double n3[BLOCK_SIZE][NEURONS3];
    __local double d1[BLOCK_SIZE][NEURONS1];//deltas
    __local double d2[BLOCK_SIZE][NEURONS1];
    __local double d3[BLOCK_SIZE][NEURONS1];

    __local double syn0[NEURONS0][NEURONSB1];//synapses
    __local double syn1[NEURONS1][NEURONSB2];
    __local double syn2[NEURONS2][NEURONSB3];
    // for(int n=nBegin, j=jBegin; n<=nEnd; n+=nStep, j+=jStep){
    //     __local double Ns[BLOCK_SIZE][NEURONS0];
    //     __local double Ss[NEURONS0][NEURONSB0];
    //     Ns[tEv][tj] = Neurons0[n+nEvents*tj+tEv];
    //     for (int i = 0; i < NeuronsPerLayer[0]; i++)

    // }
    //forward
    n1[block_id][NEURONS1-1]=1; //BIAS
    n2[block_id][NEURONS2-1]=1; //BIAS

    int batchsize = nEv_end - nEv_begin;
    int block_size = get_local_size(0);
    int n = as_int(batchsize / block_size); //nEv_end-nEv_begin=batchsize
    if(batchsize%block_size!=0)
        n++;
    if (j < NEURONSB1) {
        for (int i = 0; i < NEURONS0; i++) {
            syn0[i][j] = 0;
        }
    }
    if (j < NEURONSB2) {
        for (int i = 0; i < NEURONS1; i++) {
            syn1[i][j] = 0;
        }
    }
    if (j < NEURONSB3) {
        for (int i = 0; i < NEURONS2; i++) {
            syn2[i][j] = 0;
        }
    }
    //if (batchsize % BLOCK_SIZE != 0) n++;

    for (int l = 0; l < n; l++) {
        if(j<NEURONS0) //Read Eventvalues into local memory
            n0[block_id][j]=Neurons0[(nEv + nEv_begin) * NEURONS0 + j];

        barrier(CLK_LOCAL_MEM_FENCE);
        if (nEv < batchsize && j < NEURONSB1) {
            tmp = 0.0;
            for (int i = 0; i < NEURONS0; i++) {
                tmp += n0[block_id][i] * Synapses0[i * NEURONSB1 + j];
            }
            n1[block_id][j] = tanh(tmp); //BIAS!!!!
        }


        barrier(CLK_LOCAL_MEM_FENCE);
        if (nEv < batchsize && j < NEURONSB2) {
            tmp = 0.0;
            for (int i = 0; i < NEURONS1; i++) {
                tmp += Synapses1[i * NEURONSB2 + j] * n1[block_id][i] ;
            }
            n2[block_id][j] = tanh(tmp); //BIAS!!!!
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (nEv < batchsize && j < NEURONSB3) {
            tmp = 0.0;
            for (int i = 0; i < NEURONS2; i++) {
                tmp += Synapses2[i * NEURONSB3 + j] * n2[block_id][i] ;
            }
            n3[block_id][j] = tmp; //KEIN BIAS!!!!
        }

        //deltas
        barrier(CLK_LOCAL_MEM_FENCE);
        if (nEv < batchsize && j < NEURONS3) {
            d3[block_id][j] = n3[block_id][j] * weights[nEv + nEv_begin] - desired[(nEv + nEv_begin) * NEURONS3 + j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (nEv < batchsize && j < NEURONS2) {
            tmp = 0.0;
            for (int k = 0; k < NEURONSB3; k++) {
                tmp += d3[block_id][k] * Synapses2[j * NEURONSB3 + k];
            }
            d2[block_id][j] = tanhd(n2[block_id][j]) * tmp;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (nEv < batchsize && j < NEURONS1) {
            tmp = 0.0;
            for (int k = 0; k < NEURONSB2; k++) {
                tmp += d2[block_id][k] * Synapses1[j * NEURONSB2 + k];
            }
            d1[block_id][j] = tanhd(n1[block_id][j]) * tmp;
        }
        //synapses
        barrier(CLK_LOCAL_MEM_FENCE);

        int iEv;

        // initialize local variables


        // now a simple serial update in O(nEv), should be modified to a O(log(nEv))

        for (iEv = 0; iEv <  batchsize; iEv++) {
            if (nEv == iEv) {

                if (j < NEURONSB1) {
                    for (int i = 0; i < NEURONS0; i++) {
                        syn0[i][j] += n0[block_id][i]
                                      * d1[block_id][j]*rate;
                    }
                }
                if (j < NEURONSB2) {
                    for (int i = 0; i < NEURONS1; i++) {
                        syn1[i][j] += n1[block_id][i]
                                      * d2[block_id][j]*rate;
                    }
                }
                if (j < NEURONSB3) {
                    for (int i = 0; i < NEURONS2; i++) {
                        syn2[i][j] += n2[block_id][i]
                                      * d3[block_id][j]*rate;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // finally normalize new values
        nEv += block_size;
    }
    
    if (get_global_id(0) == 0) {

        if (j < NEURONSB1)  {
            for (int i = 0; i < NEURONS0; i++) {
                Synapses0[i * NEURONSB1 + j] += syn0[i][j] ;
            }
        }
        if (j < NEURONSB2) {
            for (int i = 0; i < NEURONS1; i++) {
                Synapses1[i * NEURONSB2 + j] += syn1[i][j] ;
            }
        }
        if (j < NEURONSB3) {
            for (int i = 0; i < NEURONS2; i++) {
                Synapses2[i * NEURONSB3 + j] += syn2[i][j] ;
            }
        }
    }
    // nEv_begin += BLOCK_SIZE;
    // nEv_end  += BLOCK_SIZE;
    //         if (nEv_end > nEvents) nEv_end = nEvents;


    // __local double syn0[NEURONS0][NEURONSB1];
    // if (nEv < batchsize && j < NEURONSB1) {
    //     for (int i = 0; i < NEURONS0; i++) {
    //         syn0[i][j]= 0;
    //     }
    //     for (int i = 0; i < NEURONS0; i++) {
    //         syn0[i][j]+= Neurons0[(nEv + nEv_begin) * NEURONS0 + i] * deltas1[(nEv + nEv_begin) * NEURONS1 + j];
    //     }
    //     barrier(CLK_LOCAL_MEM_FENCE);
    //     if(get_global_id(1)==0 && get_global_id(0)==0){
    //         for(int j=0;j<NEURONSB1;j++)
    //             for(int i=0;i<NEURONS0;i++)
    //                 Synapses0[i*NEURONSB1+j]+=syn0[i][j]*rate;
    //     }
    // }
    // barrier(CLK_GLOBAL_MEM_FENCE);

    return;
}



