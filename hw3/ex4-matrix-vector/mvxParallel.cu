//Kelvin silva
//matrix vector parallel naive

#include <stdio.h> 
#include <cuda.h>
#include <sys/time.h>

//matrix vector -> y = A*x

__global__ void simpleMxv(int width, int height, float *matrix, float *vector, float * result_vector) {

    int current_index = blockIdx.x * blockDim.x + threadIdx.x;
	float accumulate = 0.0;

	int vector_index = 0;
	for (int i = current_index; i < current_index+width; i++){
		accumulate += (matrix[i] * vector[vector_index]);
		vector_index++;	
	}	
	result_vector[blockIdx.x * blockDim.x] = accumulate;
}
typedef unsigned long long timestamp_t;
static timestamp_t get_timestamp();

int main(void) {
	int width = 32768;
	int height = 32768;
    float *matrix, *vector, *result_vector;
	float *d_matrix, *d_vector, *d_result_vector;

	// allocate host
    matrix = (float * ) malloc(width*height * sizeof(float));
    vector = (float * ) malloc(width * sizeof(float));
	result_vector = (float *) calloc(width, sizeof(float));
	
	// allocate device mem
    cudaMalloc( &d_matrix, width*height * sizeof(float));
    cudaMalloc( &d_vector, height * sizeof(float));
	cudaMalloc( &d_result_vector, width * sizeof(float)); 

	// initialize host mem
    for (int i = 0; i < width*height; i++) {
		matrix[i] = 2.5;
    }
	for (int i = 0; i < width; i++){
		vector[i] = 2.3;
	}

	// transfer from host to dev
    cudaMemcpy(d_matrix, matrix, width*height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, width * sizeof(float), cudaMemcpyHostToDevice);

    timestamp_t t0 = get_timestamp();
    //  Perform  SAXPY  on 1M elements
	// device configuration, <<< number of blocks ,num threads in block, size of shared memory >>>
    simpleMxv <<<width, 1 >>> (width, height, d_matrix, d_vector, d_result_vector);
    cudaMemcpy(result_vector, d_result_vector, width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    timestamp_t t1 = get_timestamp();

    double diff = ((double)t1 - (double)t0);
    printf("Completed in: %lf microseconds\n", diff);
	printf("Result:\n");
	/*for (int i = 0; i < width; i++){
		printf("%f\n", result_vector[i]);
	}
*/
    float maxError = 0.0f;

    printf("Max  error: %f\n", maxError);
    cudaFree(d_matrix);
    cudaFree(d_vector);
	cudaFree(d_result_vector);
    free(matrix);
    free(vector);
	free(result_vector);
}

static timestamp_t get_timestamp(){
	struct timeval now;
	gettimeofday(&now, NULL);
	return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}


