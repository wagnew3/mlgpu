extern "C"

__global__
void sigmoidDerivative(float *activation, unsigned int length)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < length; 
         i += blockDim.x * gridDim.x) 
      {
          activation[i]=1.0/(1.0+__expf(-activation[i]));
          activation[i]=activation[i]*(1.0-activation[i]);
      }
}