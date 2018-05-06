#include "maxtranParallelShare.h"

// using shared memory
__global__ void matTran(int result_row_size, int result_col_size, float* result, int input_row_size, int input_col_size, float* matrix){
// each row is a block
// size of row (vert length) is block dim	
	extern __shared__ int shared[];
	shared[threadIdx.x] = matrix[blockIdx.x * blockDim.x + threadIdx.x];


	int current_row = blockIdx.x; 
	int current_col = threadIdx.x;
	int current_vector_format = current_row * input_row_size + current_col ;

	int destination_row = current_col;
	int destination_col = current_row;
	int destination_vector_format = destination_row * result_row_size + destination_col; 
	
	//make sure all threads loaded in shared and computed their proper indicies.
	__syncthreads();
	//now transfer shared mem into proper location on global mem array
	result[destination_vector_format] = shared[current_col];
}

void matTranSharedParallel(float* matrix, float *result, int row, int col){

	float *d_matrix, *d_result;

	cudaMalloc( &d_matrix, row * col * sizeof(float));
	cudaMalloc( &d_result, row * col * sizeof(float));

	cudaMemcpy(d_matrix, matrix, col * row * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, result, col * row * sizeof(float), cudaMemcpyHostToDevice);

	matTran <<<row, col, col>>> (row, col, d_result, row, col, d_matrix);
		
	cudaMemcpy(result, d_result, col * row * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_matrix);
	cudaFree(d_result);

}

