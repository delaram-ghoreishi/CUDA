/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "stdio.h"
#include "reference_calc.cpp"
#include "utils.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__global__ void shmem_min_kernel(const float* const d_logLuminance, float* d_min_logLum,
                                          const size_t numRows, const size_t numCols)
{
   extern __shared__ float sdata[];
   int x_index = blockDim.x * blockIdx.x + threadIdx.x;
   int y_index = blockDim.y * blockIdx.y + threadIdx.y;
   int offset = numCols * y_index + x_index;
   int myId = blockDim.x * threadIdx.y + threadIdx.x;
   
   sdata[myId] = d_logLuminance[offset];
   __syncthreads();
       
   for (unsigned int r = blockDim.y/2; r > 0; r >>= 1){
       if(threadIdx.y < r)
       {
          sdata[myId] = MIN(sdata[myId], sdata[blockDim.x * (threadIdx.y + r) + threadIdx.x]);
       }
    }
       
   for (unsigned int c = blockDim.x/2; c > 0; c >>= 1){
      if(threadIdx.x < c)
      {
         sdata[threadIdx.x] = MIN(sdata[threadIdx.x], sdata[threadIdx.x + c]);
      }
   }
       
   if(threadIdx.x == 0 && threadIdx.y ==0)
   {
      *d_min_logLum  = MIN(*d_min_logLum,sdata[0]);
   }
}

__global__ void shmem_max_kernel(const float* const d_logLuminance, float *d_max_logLum,
                                          const size_t numRows, const size_t numCols)
{
   extern __shared__ float sdata[];
   int x_index = blockDim.x * blockIdx.x + threadIdx.x;
   int y_index = blockDim.y * blockIdx.y + threadIdx.y;
   int offset = numCols * y_index + x_index;
   int myId = blockDim.x * threadIdx.y + threadIdx.x;
   
   sdata[myId] = d_logLuminance[offset];
   __syncthreads();
       
   for (unsigned int r = blockDim.y/2; r > 0; r >>= 1){
       if(threadIdx.y < r)
       {
          sdata[myId] = MAX(sdata[myId], sdata[blockDim.x * (threadIdx.y + r) + threadIdx.x]);
       }
    }
       
   for (unsigned int c = blockDim.x/2; c > 0; c >>= 1){
      if(threadIdx.x < c)
      {
         sdata[threadIdx.x] = MAX(sdata[threadIdx.x], sdata[threadIdx.x + c]);
      }
   }
       
   if(threadIdx.x == 0 && threadIdx.y ==0)
   {
      *d_max_logLum  = MAX(*d_max_logLum,sdata[0]);
   }
}

__global__ void histogram(const float* const d_logLuminance, float *d_min_logLum, float *d_max_logLum, int *d_bins,
                          float *d_lumRange, const size_t numRows, const size_t numCols, const size_t numBins)
{
   int x_index = blockDim.x * blockIdx.x + threadIdx.x;
   int y_index = blockDim.y * blockIdx.y + threadIdx.y;
   int offset = numCols * y_index + x_index;
   int bin = int((d_logLuminance[offset] - *d_min_logLum) / *d_lumRange * (numBins-1));
   atomicAdd(&(d_bins[bin]),1);
}

__global__ void exclusiveScan(unsigned int* const d_cdf, int *d_bins, const size_t numBins)
{
   int acc = 0;
   //int index = blockIdx.x* blockDim.x + threadIdx.x;
   for (int i = 0; i < numBins; i++)
   {
      d_cdf[i] = acc;
      acc = acc + d_bins[i];
   }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    
    int blockLength = 10;
    const dim3 blockSize(blockLength, blockLength, 1);
    const dim3 gridSize(ceil(numCols/blockLength), ceil(numRows/blockLength), 1);
    
    float * d_min_logLum;
    float * d_max_logLum;
    
    checkCudaErrors(cudaMalloc((void **) &d_min_logLum, sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_min_logLum, &min_logLum, sizeof(float), cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMalloc((void **) &d_max_logLum, sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_max_logLum, &max_logLum, sizeof(float), cudaMemcpyHostToDevice));
    
    shmem_min_kernel<<<gridSize, blockSize, blockLength * blockLength * sizeof(int)>>>(d_logLuminance, d_min_logLum, numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    shmem_max_kernel<<<gridSize, blockSize, blockLength * blockLength * sizeof(int)>>>(d_logLuminance, d_max_logLum, numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(&min_logLum, d_min_logLum, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&max_logLum, d_max_logLum, sizeof(float), cudaMemcpyDeviceToHost));

    float lumRange = max_logLum - min_logLum;
    
    float * d_lumRange;
    checkCudaErrors(cudaMalloc((void **) &d_lumRange, sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_lumRange, &lumRange, sizeof(float), cudaMemcpyHostToDevice));
    
    int h_bins[numBins];
    for (int i = 0; i < numBins; i++){
       h_bins[i] = 0;
    }
    
    int * d_bins;
    
    checkCudaErrors(cudaMalloc((void **) &d_bins, numBins * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_bins, h_bins, numBins * sizeof(int), cudaMemcpyHostToDevice));
    
    histogram<<<gridSize, blockSize>>>(d_logLuminance, d_min_logLum, d_max_logLum, d_bins, d_lumRange, numRows, numCols, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    checkCudaErrors(cudaMemcpy(h_bins, d_bins, numBins * sizeof(int), cudaMemcpyDeviceToHost));
    
    exclusiveScan<<<1, 1>>>(d_cdf, d_bins, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    cudaFree(d_bins);
    cudaFree(d_min_logLum);
    cudaFree(d_max_logLum);
    cudaFree(d_lumRange);
    
}
