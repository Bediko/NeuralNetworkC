
#define FUNCTION  tanh     //function
#define DERIVATE tanhd     //derivate of function
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define BLOCK_SIZE 16


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
                                     __global double *deltas1, __global double *deltas2, __global double *deltas3, __global double *desired, __global double *weights,
                                     __global int *NeuronsPerLayer, __global int *bias,
                                     int nEv_begin, int nEv_end, int batchsize, double rate)
{

    // int bEv = get_group_id(0);
    // int bj = get_group_id(1);
    // int tEv=get_local_id(0);
    // int tj= get_local_id(1);

    int nEv = get_global_id(0);
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
    __local double syn0[NEURONS0][NEURONSB1];
    __local double syn1[NEURONS1][NEURONSB2];
    __local double syn2[NEURONS2][NEURONSB3];
    // for(int n=nBegin, j=jBegin; n<=nEnd; n+=nStep, j+=jStep){
    //     __local double Ns[BLOCK_SIZE][NEURONS0];
    //     __local double Ss[NEURONS0][NEURONSB0];
    //     Ns[tEv][tj] = Neurons0[n+nEvents*tj+tEv];
    //     for (int i = 0; i < NeuronsPerLayer[0]; i++)

    // }
    //forward
     
     __private int n = as_int(batchsize / BLOCK_SIZE);
     //if (batchsize % BLOCK_SIZE != 0) n++;

    for(int l=0;l<n;l++){
        if (nEv< nEv_end - nEv_begin && j < NEURONSB1) {
            tmp = 0.0;
            for (int i = 0; i < NEURONS0; i++) {
                tmp += Neurons0[(nEv + nEv_begin) * NEURONS0 + i] * Synapses0[i * NEURONSB1 + j];
            }
            Neurons1[(nEv + nEv_begin) * NEURONS1 + j] = tanh(tmp); //BIAS!!!!
        }
    
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (nEv < nEv_end - nEv_begin && j < NEURONSB2) {
            tmp = 0.0;
            for (int i = 0; i < NEURONS1; i++) {
                tmp += Synapses1[i * NEURONSB2 + j] * Neurons1[(nEv + nEv_begin) * NEURONS1 + i] ;
            }
            Neurons2[(nEv + nEv_begin) * NEURONS2 + j] = tanh(tmp); //BIAS!!!!
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
        if (nEv < nEv_end - nEv_begin && j < NEURONSB3) {
            tmp = 0.0;
            for (int i = 0; i < NEURONS2; i++) {
                tmp += Synapses2[i * NEURONSB3 + j] * Neurons2[(nEv + nEv_begin) * NEURONS2 + i] ;
            }
            Neurons3[(nEv + nEv_begin) * NEURONS3 + j] = tmp; //KEIN BIAS!!!!
        }
    }
        //deltas
        barrier(CLK_GLOBAL_MEM_FENCE);
        if (nEv < nEv_end - nEv_begin && j < NEURONS3) {
            deltas3[(nEv + nEv_begin)*NEURONS3 + j] = Neurons3[(nEv + nEv_begin) * NEURONS3 + j] * weights[nEv + nEv_begin] - desired[(nEv + nEv_begin) * NEURONS3 + j];
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
        if (nEv < nEv_end - nEv_begin && j < NEURONS2) {
            tmp = 0.0;
            for (int k = 0; k < NEURONSB3; k++) {
                tmp += deltas3[(nEv + nEv_begin) * NEURONS3 + k] * Synapses2[j * NEURONSB3 + k];
            }
            deltas2[(nEv + nEv_begin)*NEURONS2 + j] = tanhd(Neurons2[(nEv + nEv_begin) * NEURONS2 + j]) * tmp;
        }
    
        barrier(CLK_GLOBAL_MEM_FENCE);
        if (nEv < nEv_end - nEv_begin && j < NEURONS1) {
            tmp = 0.0;
            for (int k = 0; k < NEURONSB2; k++) {
                tmp += deltas2[(nEv + nEv_begin) * NEURONS2 + k] * Synapses1[j * NEURONSB2 + k];
            }
            deltas1[(nEv + nEv_begin)*NEURONS1 + j] = tanhd(Neurons1[(nEv + nEv_begin) * NEURONS1 + j]) * tmp;
        }
        //synapses
        barrier(CLK_GLOBAL_MEM_FENCE);

        int iEv;

        // initialize local variables

        if (nEv < nEv_end - nEv_begin) {
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
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        // now a simple serial update in O(nEv), should be modified to a O(log(nEv))

        for (iEv = 0; iEv <  nEv_end - nEv_begin; iEv++) {
            if (nEv == iEv) {

                if (j < NEURONSB1) {
                    for (int i = 0; i < NEURONS0; i++) {
                        syn0[i][j] += Neurons0[(nEv + nEv_begin) * NEURONS0 + i]
                                      * deltas1[(nEv + nEv_begin) * NEURONS1 + j];
                    }
                }
                if (j < NEURONSB2) {
                    for (int i = 0; i < NEURONS1; i++) {
                        syn1[i][j] += Neurons1[(nEv + nEv_begin) * NEURONS1 + i]
                                      * deltas2[(nEv + nEv_begin) * NEURONS2 + j];
                    }
                }
                if (j < NEURONSB3) {
                    for (int i = 0; i < NEURONS2; i++) {
                        syn2[i][j] += Neurons2[(nEv + nEv_begin) * NEURONS2 + i]
                                      * deltas3[(nEv + nEv_begin) * NEURONS3 + j];
                    }
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        // finally normalize new values

        if (nEv == 0) {

            if (j < NEURONSB1)  {
                for (int i = 0; i < NEURONS0; i++) {
                    Synapses0[i * NEURONSB1 + j] += syn0[i][j] * rate;
                }
            }
            if (j < NEURONSB2) {
                for (int i = 0; i < NEURONS1; i++) {
                    Synapses1[i * NEURONSB2 + j] += syn1[i][j] * rate;
                }
            }
            if (j < NEURONSB3) {
                for (int i = 0; i < NEURONS2; i++) {
                    Synapses2[i * NEURONSB3 + j] += syn2[i][j] * rate;
                }
            }
        }
    
             // nEv_begin += BLOCK_SIZE;
             // nEv_end  += BLOCK_SIZE;
    //         if (nEv_end > nEvents) nEv_end = nEvents;
     



    barrier(CLK_GLOBAL_MEM_FENCE);
    // __local double syn0[NEURONS0][NEURONSB1];
    // if (nEv < nEv_end - nEv_begin && j < NEURONSB1) {
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



