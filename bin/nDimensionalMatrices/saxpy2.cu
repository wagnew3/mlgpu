extern "C"

__global__
void saxpy2(float a, float *x, float *y, float* r, unsigned int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
          r[i] = a * x[i] + y[i];
      }
}