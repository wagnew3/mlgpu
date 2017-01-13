extern "C"

__global__
void sigmoid(float *activation, unsigned int length)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < length; 
         i += blockDim.x * gridDim.x) 
      {
          
          activation[i]=1.0f/(1.0f+__expf(-activation[i]));
          
          //activation[i]=1.0f/(1.0f+expf(-activation[i]));
          
          //activation[i]=activation[i]/(0.5f+0.5f*fabsf(activation[i]))+0.5f;
      }
}

