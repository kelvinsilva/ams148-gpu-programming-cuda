#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
typedef unsigned long long timestamp_t;
static timestamp_t get_timestamp();

extern "C"{
	#include "bmp.h"
}
__global__ void render(char * out, const int width, const int height, const int max_iter) {
    int x_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int y_dim = blockIdx.y * blockDim.y + threadIdx.y;
    int index = 3 * width * y_dim + x_dim * 3;
    float x_origin = ((float) x_dim / width) * 3.25 - 2;
    float y_origin = ((float) y_dim / width) * 2.5 - 1.25;
    float x = 0.0;
    float y = 0.0;
    int iteration = 0;
    while (x * x + y * y <= 4 && iteration < max_iter) {
      float xtemp = x * x - y * y + x_origin;
      y = 2 * x * y + y_origin;
      x = xtemp;
      iteration++;
    }
    if (iteration == max_iter) {
      out[index] = 0;
      out[index + 1] = 0;
      out[index + 2] = 0;
    } else {
      out[index] = iteration;
      out[index + 1] = iteration;
      out[index + 2] = iteration;
    }
}
    
int main(){
  //  Multiply  by 3 here , since  we need red , green  and  blue  for  each  pixel
  int N = 8196;
  int width = N;
  int height = N;
  int max_iter = 2048;
  size_t buffer_size = sizeof(char) * width * height * 3;

  char * d_image; //  device  image
  cudaMalloc((void * * ) & d_image, buffer_size); // allocation

  char * host_image = (char * ) malloc(buffer_size); // host  image
  dim3 blockDim(16, 16, 1);
  dim3 gridDim(width / blockDim.x, height / blockDim.y, 1);

  timestamp_t t0 = get_timestamp();
  render << < gridDim, blockDim, 0 >>> (d_image, width, height, max_iter);
  cudaMemcpy(host_image, d_image, buffer_size, cudaMemcpyDeviceToHost);

  timestamp_t t1 = get_timestamp();
  // Now  write  the  file
  write_bmp("output.bmp", width, height, host_image);
  cudaFree(d_image);
  free(host_image);

  double diff = ((double)t1 - (double)t0);
  printf("Time: %lf", diff);

}
static timestamp_t get_timestamp(){
         struct timeval now;
         gettimeofday(&now, NULL);
         return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

