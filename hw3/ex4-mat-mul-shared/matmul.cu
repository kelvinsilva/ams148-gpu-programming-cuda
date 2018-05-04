//Kelvin silva
//matrix multiply 
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <ctime>
#include <omp.h>
#include <stdio.h> 
#include <cuda.h>
#include <sys/time.h>

#define BLOCK_SIZE 32
typedef struct{
	int width;
	int height;
	int stride;
	float* elements;
} Matrix;

__device__  float  GetElement(const  Matrix A, int row , int  col)
{
	return A.elements[row*A.stride + col];
}

__device__  void  SetElement(Matrix A, int row , int col , float  value)
{
	A.elements[row * A.stride + col] = value;
}

__device__  Matrix GetSubMatrix(Matrix A, int row , int  col)
{
	Matrix  Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return  Asub;
}

__global__  void  MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	//  Block  row  and  column;
	int  blockRow = blockIdx.y;
	int  blockCol = blockIdx.x;
	// Thread  block  computes  one  sub  matrix  Csub of C
	Matrix  Csub = GetSubMatrix(C, blockRow , blockCol);
	// Each  thread  computes  one  element  of Csub
	// By  accumulating  results  into  Cvalue
	float  Cvalue = 0.0f;
	// Thread  row and  column  index  within  the  submatrix
	int  row = threadIdx.y;
	int  col = threadIdx.x;
	// Loop  over  submatrices  of A and B that  are  required  for  Csub
	// Multiply  each  pair of sub -matrices  together
	//and  summ  the  results
	for (int m = 0; m < (A.width/BLOCK_SIZE); m++){
		//Get A submatrix
		Matrix  Asub = GetSubMatrix(A, blockRow , m);
		//Get B submatrix
		Matrix  Bsub = GetSubMatrix(B, m ,blockCol);
		// Static  shared  memory  for  Asub  and  Bsub
		__shared__  float  As[BLOCK_SIZE ][ BLOCK_SIZE ];
		__shared__  float  Bs[BLOCK_SIZE ][ BLOCK_SIZE ]; //Great  name  for an  array
		//Load  Asub  and  Bsub  from  global  memory  into  shared;
		As[row][col] = GetElement(Asub ,row ,col);
		Bs[row][col] = GetElement(Bsub ,row ,col);
		// Always  sync  threads  when  loading  shared  memory  before  doing  computation
		__syncthreads ();
		// Multiply  the  submatrices
		for (int e = 0; e < BLOCK_SIZE; e++)
			Cvalue  += As[row][e]*Bs[e][col];
		// synchronize  to make  sure  all  threads  are  done  computing
		__syncthreads();
	}
	//write  Csub  back  into  global  memory
	//each  thread  writes  one  element
	SetElement(Csub , row , col , Cvalue);
}

// Matrix multiplication host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
//Load A and B to device memory 
	Matrix d_A, d_B, d_C;
	d_A.width = d_A.stride = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float); 
	cudaMalloc(&d_A.elements,size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice); 

	d_B.width = d_B.stride = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float); 
	cudaMalloc(&d_B.elements,size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice); 

	//Allocate C in device memory
	d_C.width = d_C.stride = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float); 
	cudaMalloc(&d_C.elements, size); 

	// Invoke Kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height/ dimBlock.y); 
	clock_t begin = clock();
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C); 
	cudaDeviceSynchronize();
	clock_t end = clock();

	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout<<"Run Time! "<< elapsed_secs << std::endl;

	// Read C from Device memory 
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost); 

	//Free device memory 
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

typedef unsigned long long timestamp_t;
static timestamp_t get_timestamp();

int main(void) {

	Matrix A, B, C; 
	int N = 1024;
	int M = 1024;

	A.width = N;
	B.width = N; 
	C.width = N; 
	
	A.height = M;
	B.height = M;
	C.height = M;

	A.stride = A.width;
	B.stride = B.width;
	C.stride = C.width;

	size_t asize = A.width * A.height * sizeof(float);
	size_t bsize = B.width * B.height * sizeof(float);
	size_t csize = C.width * C.height * sizeof(float);

	A.elements = (float*)malloc(asize);
	B.elements = (float*)malloc(bsize);
	C.elements = (float*)malloc(csize);

	//set values for A and B 
	for( int i = 0; i < A.height; i++){
		for( int j = 0; j < A.width; j++)
		{
			A.elements[i*A.stride + j] = 1.0f;
			B.elements[i*B.stride + j] = 1.0f;
		}
	}
    timestamp_t t1 = get_timestamp();
	//SharedMemCUDA
	MatMul(A,B,C);
    timestamp_t t2 = get_timestamp();
}

static timestamp_t get_timestamp(){
	struct timeval now;
	gettimeofday(&now, NULL);
	return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}


