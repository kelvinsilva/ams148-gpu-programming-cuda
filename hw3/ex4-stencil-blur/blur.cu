#include <cuda_runtime.h>
#include "CImg.h"

using namespace std;
using namespace cimg_library;

__device__ uchar3 mutate_values(unsigned char * data, int2 pos, float multiplier, int width, int height ){

	uchar3 d;
	d.x = ((float)data[pos.y * width + pos.x]) * multiplier; //red
	d.y = ((float)data[(height + pos.y ) * width + pos.x]) * multiplier; //green
	d.z = ((float)data[(height * 2 + pos.y) * width + pos.x]) * multiplier; //blue
	return d;
}
__device__ void writeRGBtoData(unsigned char * data_out, uchar3 rgb, int2 pos, int width, int height){
	data_out[pos.y * width + pos.x] = rgb.x; //red
	data_out[(height + pos.y ) * width + pos.x] = rgb.y; //green
	data_out[(height * 2 + pos.y) * width + pos.x] = rgb.z; //blue
}
__device__ void stencil_blur(unsigned char * data_out, int2 pos, unsigned char * data, int width, int height){

	int2 pos0 = {pos.x-1, pos.y-1};
	uchar3 rgb0 = mutate_values(data, pos0, 0.0625, width, height);

	int2 pos1 = {pos.x, pos.y-1};
	uchar3 rgb1 = mutate_values(data, pos1, 0.125, width, height);

	int2 pos2 = {pos.x+1, pos.y-1};
	uchar3 rgb2 = mutate_values(data, pos2, 0.0625, width, height);

	int2 pos3 = {pos.x-1, pos.y};
	uchar3 rgb3 = mutate_values(data, pos3, 0.125, width, height);

	int2 pos4 = {pos.x, pos.y};
	uchar3 rgb4 = mutate_values(data, pos4, 0.25, width, height);

	int2 pos5 = {pos.x+1, pos.y};
	uchar3 rgb5 = mutate_values(data, pos5, 0.125, width, height);

	int2 pos6 = {pos.x-1, pos.y+1};
	uchar3 rgb6 = mutate_values(data, pos6, 0.0625, width, height);

	int2 pos7 = {pos.x, pos.y+1};
	uchar3 rgb7 = mutate_values(data, pos7, 0.125, width, height);

	int2 pos8 = {pos.x+1, pos.y+1};
	uchar3 rgb8 = mutate_values(data, pos8, 0.0625, width, height);

	uchar3 rgb_result;
		rgb_result.x = rgb0.x + rgb1.x + rgb2.x + rgb3.x + rgb4.x + rgb5.x + rgb6.x + rgb7.x + rgb8.x;
		rgb_result.y = rgb0.y + rgb1.y + rgb2.y + rgb3.y + rgb4.y + rgb5.y + rgb6.y + rgb7.y + rgb8.y;
		rgb_result.z = rgb0.z + rgb1.z + rgb2.z + rgb3.z + rgb4.z + rgb5.z + rgb6.z + rgb7.z + rgb8.z;

	writeRGBtoData(data_out, rgb_result, pos4, width, height); // pos4 is center pixel
}
__global__ void rgb2gray(unsigned char * d_src, unsigned char * d_dst, int width, int height)
{
	int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
	int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (pos_x >= (width) || pos_y >= (height))
		return;

	int2 pos = {pos_x, pos_y};

	stencil_blur(d_dst, pos, d_src, width, height);	
	uchar3 current_pixel = mutate_values(d_dst, pos, 1.0, width,height);
	writeRGBtoData(d_src, current_pixel, pos, width, height); // overwrite source with blurred data. allows for successive calls of kernel.
}


int main()
{
	//load image
	CImg<unsigned char> src("picture.bmp");
	int width = src.width();
	int height = src.height();
	unsigned long size = src.size();

	//create pointer to image
	unsigned char *h_src = src.data();

	CImg<unsigned char> gs(width, height, 1, 3);
	unsigned char *h_gs = gs.data();
	unsigned char *d_src;
	unsigned char *d_gs;

	cudaMalloc((void**)&d_src, size);
	cudaMalloc((void**)&d_gs, width*height*3*sizeof(unsigned char));

	cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice);

	//launch the kernel
	dim3 blkDim (16, 16, 1);
	dim3 grdDim ((width + 15)/16, (height + 15)/16, 3);

	rgb2gray<<<grdDim, blkDim>>>(d_src, d_gs, width, height);
	rgb2gray<<<grdDim, blkDim>>>(d_src, d_gs, width, height);
	rgb2gray<<<grdDim, blkDim>>>(d_src, d_gs, width, height);
	rgb2gray<<<grdDim, blkDim>>>(d_src, d_gs, width, height);
	//successive calls of blur will blur picture a lot.
	//calling blur only once will barely change the picture
	//wait until kernel finishes
	cudaDeviceSynchronize();

	//copy back the result to CPU
	cudaMemcpy(h_gs, d_gs, width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(d_src);
	cudaFree(d_gs);

	CImg<unsigned char> out(h_gs,width,height, 1, 3);
	out.save("blur_picture.bmp");
	return 0;
}
