//Udacity HW 4
//Radix Sorting

#include "reference_calc.h"
#include "utils.h"

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__ void histogram(unsigned int* const d_inputVals,
                          unsigned int* d_bins, 
                          int bit,
                          const size_t numElems)
{
 int index = blockDim.x * blockIdx.x + threadIdx.x;
 if(index < int(numElems))
 {
  int bin = (d_inputVals[index] >> bit) & 1;
  atomicAdd(&(d_bins[bin]),1);
 }
}

__global__ void exclusiveSum(unsigned int* const d_cdf, 
                             unsigned int* d_bins)
{
 int acc = 0;
 int size = sizeof(d_bins)/sizeof(unsigned int);
 
 for(int i = 0; i < 2; i++)
 {
  d_cdf[i] = acc;
  acc = acc + d_bins[i];
 }
}

__global__ void offset(unsigned int* const d_inputVals,
                       int bit,
                       int* d_offset,
                       const size_t numElems)
{
 int counter0 = 0;
 int counter1 = 0;
 for(int i = 0; i < int(numElems); i++)
 {
  int bin = (d_inputVals[i]>>bit)&1;
  if(bin == 0)
  {
   d_offset[i] = counter0;
   counter0++;
  }
  if(bin == 1)
  {
    d_offset[i] = counter1;
    counter1++;
  }
 }
}


__global__ void radixSort(unsigned int* const d_inputVals,
                          unsigned int* const d_inputPos,
                          unsigned int* const d_outputVals,
                          unsigned int* const d_outputPos,
                          unsigned int* const d_cdf,
                          int* d_offset,
                          int bit,
                          const size_t numElems)
{
 int index = blockDim.x * blockIdx.x + threadIdx.x;
 if(index < int(numElems))
 {
  int bin = (d_inputVals[index] >> bit) & 1;
  //atomicSub(&(d_cdf[bin]),1);
  //int idx = d_cdf[bin];
  int idx = d_offset[index] + d_cdf[bin];
  
  d_outputVals[idx] = d_inputVals[index];
  d_outputPos[idx] = d_inputPos[index];
 }
}

__global__ void cpyArray(unsigned int* const d_inputVals,
                         unsigned int* const d_inputPos,
                         unsigned int* const d_outputVals,
                         unsigned int* const d_outputPos,
                         const size_t numElems)
{
 int index = blockDim.x * blockIdx.x + threadIdx.x;
 if(index<int(numElems))
 {
  d_inputVals[index] = d_outputVals[index];
  d_inputPos[index] = d_outputPos[index];
 }
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
 unsigned int h_bins[2];
 unsigned int h_cdf[2];
 int h_offset[numElems];
 
 for(int i = 0; i < 2; i++)
 {
  h_bins[i] = 0;
  h_cdf[i] = 0;
 }
 
 for(int i = 0; i < int(numElems); i++)
 {
 h_offset[i] = 0;
 }
 

 unsigned int* d_bins;
 unsigned int* d_cdf;
 //unsigned int* d_bit;
 int* d_offset;

 checkCudaErrors(cudaMalloc((void**) &d_bins, 2 * sizeof(unsigned int)));
 checkCudaErrors(cudaMemcpy(d_bins, h_bins, 2 * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
 checkCudaErrors(cudaMalloc((void**) &d_cdf, 2 * sizeof(unsigned int)));
 checkCudaErrors(cudaMemcpy(d_cdf, h_cdf, 2 * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
 checkCudaErrors(cudaMalloc((void**) &d_offset, numElems * sizeof(int)));
 checkCudaErrors(cudaMemcpy(d_offset, h_offset, numElems * sizeof(int), cudaMemcpyHostToDevice));
 
 //checkCudaErrors(cudaMalloc((void**) &d_bit, sizeof(unsigned int)));

 int blockLength = 32;
 const dim3 blockSize(blockLength, 1, 1);
 const dim3 gridSize(ceil(numElems/blockLength), 1, 1);
 
 for (unsigned int bit = 0; bit < 32; bit++)
 {
 
  //checkCudaErrors(cudaMemset(d_cdf, 0,  2 * sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_bins, 0,  2 * sizeof(unsigned int)));
  
  //checkCudaErrors(cudaMemcpy(d_bit, &bit, sizeof(unsigned int), cudaMemcpyHostToDevice));
 
  histogram<<<gridSize, blockSize>>>(d_inputVals, d_bins, bit, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
 
  exclusiveSum<<<1, 1>>>(d_cdf, d_bins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  offset<<<1,1>>>(d_inputVals, bit, d_offset, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
 
  radixSort<<<gridSize, blockSize>>>(d_inputVals, d_inputPos,
                                     d_outputVals, d_outputPos,
                                     d_cdf, d_offset, bit, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  cpyArray<<<gridSize, blockSize>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

 }
 cudaFree(d_bins);
 cudaFree(d_cdf);
 cudaFree(d_offset);
}
