#include <stdio.h> 
#include <cuda.h>
#include <sys/time.h>

__global__ void saxpy(int n, float a, float * x, float * y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
typedef unsigned long long timestamp_t;
static timestamp_t get_timestamp();

int main(void) {
    int N = 65536;
    float *x, *y, *d_x, *d_y;
    x = (float * ) malloc(N * sizeof(float));
    y = (float * ) malloc(N * sizeof(float));
    cudaMalloc( & d_x, N * sizeof(float));
    cudaMalloc( & d_y, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    timestamp_t t0 = get_timestamp();
    //  Perform  SAXPY  on 1M elements
    saxpy <<< (N + 255) / 256, 256 >>> (N, 2.0f, d_x, d_y);
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    timestamp_t t1 = get_timestamp();

    double diff = ((double)t1 - (double)t0);
    printf("Completed in: %lf microseconds\n", diff);

    float maxError = 0.0f;

    for (int i = 0; i < N; i++)
        maxError = max(maxError, abs(y[i] - 4.0f));

    printf("Max  error: %f\n", maxError);
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}

static timestamp_t get_timestamp(){
	struct timeval now;
	gettimeofday(&now, NULL);
	return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}


