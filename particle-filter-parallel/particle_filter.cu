#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>


typedef unsigned long long timestamp_t;
static timestamp_t get_timestamp();

int main(){
/*	
	cudaMalloc( &d_matrix, row * col * sizeof(float));
	cudaMalloc( &d_result, row * col * sizeof(float));
	timestamp_t t0 = get_timestamp();

	cudaMemcpy(d_matrix, matrix, col * row * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, result, col * row * sizeof(float), cudaMemcpyHostToDevice);

	matTran <<<row, col>>> (row, col, d_result, row, col, d_matrix);
		
	cudaMemcpy(result, d_result, col * row * sizeof(float), cudaMemcpyDeviceToHost);

	timestamp_t t1 = get_timestamp();

	double diff = (double)t1 - (double)t0;
	printf("RUNNING TIME: %f microsecond\n", diff);	
*/
	return 0;
}

static timestamp_t get_timestamp(){
	struct timeval now;
	gettimeofday(&now, NULL);
	return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}
