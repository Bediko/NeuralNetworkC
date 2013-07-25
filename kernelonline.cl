
#define FUNCTION  tanh     //function
#define DERIVATE tanhd     //derivate of function
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define BLOCK_SIZE 5


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
                                     int nEv_begin, int nEv_end, int nEvents)
{

    // int bEv = get_group_id(0);
    // int bj = get_group_id(1);
    // int tEv=get_local_id(0);
    // int tj= get_local_id(1);

    int nEv = get_global_id(0);
    int j = get_global_id(1);

    // int nBegin=nEvents*BLOCK_SIZE*bj;
    // int nEnd=nBegin+nEvents-1;
    // int nStep= BLOCK_SIZE;
    // int jBegin=BLOCK_SIZE*bj;
    // int jStep=BLOCK_SIZE*NEURONS0;

    double tmp;
    // for(int n=nBegin, j=jBegin; n<=nEnd; n+=nStep, j+=jStep){
    //     __local double Ns[BLOCK_SIZE][NEURONS0];
    //     __local double Ss[NEURONS0][NEURONSB0];
    //     Ns[tEv][tj] = Neurons0[n+nEvents*tj+tEv];
    //     for (int i = 0; i < NeuronsPerLayer[0]; i++)

    // }
    //forward
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

    //deltas
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (nEv < nEv_end - nEv_begin && j < NeuronsPerLayer[3]) {
        deltas3[(nEv + nEv_begin)*NeuronsPerLayer[3] + j] = Neurons3[(nEv + nEv_begin) * NeuronsPerLayer[3] + j] * weights[nEv + nEv_begin] - desired[(nEv + nEv_begin) * NeuronsPerLayer[3] + j];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (nEv < nEv_end - nEv_begin && j < NeuronsPerLayer[2]) {
        tmp = 0.0;
        for (int k = 0; k < NeuronsPerLayer[3] - bias[3]; k++) {
            tmp += deltas3[(nEv + nEv_begin)*NeuronsPerLayer[3] + k] * Synapses2[j*(NeuronsPerLayer[3] - bias[3])+k];
        }
        deltas2[(nEv + nEv_begin)*NeuronsPerLayer[2] +j] = tanhd(Neurons2[(nEv + nEv_begin) * NeuronsPerLayer[2] + j]) * tmp;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (nEv < nEv_end - nEv_begin && j < NeuronsPerLayer[1]) {
        tmp = 0.0;
        for (int k = 0; k < NeuronsPerLayer[2] - bias[2]; k++) {
            tmp += deltas2[(nEv + nEv_begin)*NeuronsPerLayer[2] + k] * Synapses1[j*(NeuronsPerLayer[2] - bias[2])+k];
        }
        deltas1[(nEv + nEv_begin)*NeuronsPerLayer[1] +j] = tanhd(Neurons1[(nEv + nEv_begin) * NeuronsPerLayer[1] + j]) * tmp;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
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


