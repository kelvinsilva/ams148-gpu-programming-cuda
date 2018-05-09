#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>



__global__ void matTran(int result_row_size, int result_col_size, float* result, int input_row_size, int input_col_size, float* matrix){
// each row is a block
// size of row (vert length) is block dim	

	int current_row = blockIdx.x; 
	int current_col = threadIdx.x;
	int current_vector_format = current_row * input_row_size + current_col ;

	int destination_row = current_col;
	int destination_col = current_row;
	int destination_vector_format = destination_row * result_row_size + destination_col; 
	
	result[destination_vector_format] = matrix[current_vector_format];
}

typedef unsigned long long timestamp_t;
static timestamp_t get_timestamp();

int main(){
	
	int row = 16384;
	int col = 16384;

	float *matrix, *result;
	float *d_matrix, *d_result;

	matrix = (float*) malloc(row * col * sizeof(float));
	result = (float*)  malloc(row * col * sizeof(float));

	cudaMalloc( &d_matrix, row * col * sizeof(float));
	cudaMalloc( &d_result, row * col * sizeof(float));
	
	for(int i = 0; i < row*col; i++){
		
		matrix[i] = 0;
		result[i] = 0;
	}

	for(int i = 0, j = row*2; i < row; j++, i++){
		
		matrix[i] = 2.5;
		matrix[j] = 2.6;
		result[i] = 0;
	}
	timestamp_t t0 = get_timestamp();

	cudaMemcpy(d_matrix, matrix, col * row * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, result, col * row * sizeof(float), cudaMemcpyHostToDevice);

	matTran <<<row, col>>> (row, col, d_result, row, col, d_matrix);
		
	cudaMemcpy(result, d_result, col * row * sizeof(float), cudaMemcpyDeviceToHost);

	timestamp_t t1 = get_timestamp();

	double diff = (double)t1 - (double)t0;
	printf("RUNNING TIME: %f microsecond\n", diff);	
	/*printf("Original: \n");
	for (int i = 0; i < row; i++){
		printf("\n");
		for(int j = 0; j < col; j++){
			printf(" %.2f ", matrix[i * row + j]);
		}
	}*/

	/*printf("Result: \n");
	for (int i = 0; i < row; i++){
		printf("\n");
		for(int j = 0; j < col; j++){
			printf(" %.2f ", result[i * row + j]);
		}
	}*/


	return 0;
}

static timestamp_t get_timestamp(){
	struct timeval now;
	gettimeofday(&now, NULL);
	return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}
