extern "C"

__global__
void crossEntropyCost(float *desiredOutput, unsigned int length, float *networkOutput, float* result)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < length; 
         i += blockDim.x * gridDim.x) 
      {
          result[i]=desiredOutput[i]*logf(0.00001f+networkOutput[i])+(1.0f-desiredOutput[i])*logf(1.00001f-networkOutput[i]);
      }
}

