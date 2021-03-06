/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include "reference.cpp"
#include <stdio.h>
const int n = 16;

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals,
               int numBins)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
  
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = threadIdx.x;
  int dim = blockDim.x;
  int x_corner = blockIdx.x * blockDim.x * n;
  
  if(index > numVals) return;
  
  extern __shared__ unsigned int s_histo[];
  
    s_histo[idx] = 0;
   __syncthreads();
  
  
  for(int i = 0; i < n; i++)
  {
    atomicAdd(&s_histo[vals[x_corner + idx + dim * i]], 1);
  }
   __syncthreads();
  
  atomicAdd(&histo[idx], s_histo[idx]);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //printf("numBin = %d\n",numBins);
  //printf("numElems = %d\n",numElems);
  int blockLength = numBins;
  dim3 threads(blockLength, 1, 1);
  dim3 blocks(ceil(numElems/(n*blockLength)), 1, 1);
  //TODO Launch the yourHisto kernel
  yourHisto<<<blocks, threads, (numBins) * sizeof(unsigned int)>>>(d_vals, d_histo, numElems, numBins);
  //if you want to use/launch more than one kernel,
  //feel free
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //delete[] h_vals;
  //delete[] h_histo;
  //delete[] your_histo;*/
}
