#include <stdio.h>
#include "robot.h"
#include <cuda.h>
#include <sys/time.h>
#include <algorithm>
#include <iostream>

//number of particles to run in particle filter
#define N_PARTICLES 524288 //256 min value 
//number of iterations of the particle filter
#define N_ITERATIONS 1 
//how many particles to assign to each thread block
#define DIVISIONS 1024 

//Robot parameters
#define MOVE_UPDATE_X 5.0 
#define MOVE_UPDATE_THETA 0.1 
#define NUM_LANDMARKS 4

#define PARTICLE_NOISE_F 0.05
#define PARTICLE_NOISE_T 0.05
#define PARTICLE_NOISE_S 5.00

#define METROPOLIS_B 16 
const int NUM_SENSOR_READINGS = NUM_LANDMARKS; //landmarks = sensor readings enforce


typedef unsigned long long timestamp_t;
static timestamp_t get_timestamp();

// generate_particles <<<NUM_BLOCKS, NUM_THREADS>>> (d_particle_list, NUM_THREADS, N_PARTICLES);
__global__ void generate_particles( struct Robot* d_particle_list, int n_threads, int n_particles){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > n_particles - 1){
		return;
	}
	struct Robot* r = &d_particle_list[i];
	gpu_init_robot(r);
	gpu_set_noise(r, PARTICLE_NOISE_F, PARTICLE_NOISE_T, PARTICLE_NOISE_S);
}
	
/*
Particle Filter One Iteration Function
-----------------------------------------------------------------------------------------------------
@float turn 								- Current angle of real robot
@float xdir 								- Current forward velocity of real robot
@struct Robot* d_particle_list 				- Pointer to device memory for particle list
@struct SensorData* d_my_robot_sensor_data  - Pointer to device memory for real robots sensor data
@struct LandmarkData* d_map_landmarks 		- Pointer to device memory representing landmarks on map
@double* d_weight_prob 						- Pointer to device memory for weight distribution list
@int B										- Parameter for Metropolis Resampling specifying degree of unbiasedness in resampling
@int n_particles 							- Number of particles for particle filter 


*/
__global__ void particle_filter_one_iteration(float turn, float xdir, struct Robot* d_particle_list, struct SensorData* d_my_robot_sensor_data, struct LandmarkData* d_map_landmarks, double* d_weight_prob, int B, int n_particles){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > n_particles-1){
		return;
	}
// /****************************Motion Update********************************************/
	{
		struct Robot* r = &d_particle_list[i];
		gpu_move_and_get_particle(r, turn, xdir, r);  
	}
	__syncthreads();

// /*****************************Measurement Update****************************************/

	{

		struct Robot* r = &d_particle_list[i];
		d_weight_prob[i] = gpu_calculate_measurement_probability(r, d_my_robot_sensor_data, d_map_landmarks);
	}
	__syncthreads();
// /*********************************Metropolis Resampling********************************/
	
	int k = i;

	curandState state;
	curand_init((unsigned long long)clock() + k, 0, 0, &state);

	for (int i = 0; i < B; i++){
		double u = curand_uniform_double(&state);
		float rand_num = curand_uniform(&state);
		rand_num *= ( (n_particles-1) - 0 + 0.999999 );
		rand_num += 0;
		int j = __float2int_rz(rand_num);
		
		if ((double)u <= ((double)d_weight_prob[j] / (double)d_weight_prob[k] )){
			k = j;
		}
	}
	struct Robot r_temp = d_particle_list[k];
	__syncthreads();
	d_particle_list[i] = r_temp; 
	__syncthreads();
}	

//https://stackoverflow.com/questions/24537112/uniformly-distributed-pseudorandom-integers-inside-cuda-kernel/24537113#24537113
//https://arxiv.org/pdf/1301.4019.pdf



int main(){
	
	std::cout << "Started" << std::endl;
	std::cerr << "Started" << std::endl;
	//initialize random number gen
	srand(time(NULL));
	//initialize landmarks
	struct LandmarkData map_landmarks;
	std::cout << "S0" << std::endl;	
	map_landmarks.num_landmarks = NUM_LANDMARKS;
   	map_landmarks.landmarks = new struct Landmark[map_landmarks.num_landmarks];

	std::cout << "S1 " << std::endl;
	map_landmarks.landmarks[0].x = 20.0;
	map_landmarks.landmarks[0].y = 20.0;

	map_landmarks.landmarks[1].x = 80.0;
	map_landmarks.landmarks[1].y = 80.0;

	map_landmarks.landmarks[2].x = 20.0;
	map_landmarks.landmarks[2].y = 80.0;

	map_landmarks.landmarks[3].x = 80.0;
	map_landmarks.landmarks[3].y = 20.0;

	std::cout << "map landmark" << std::endl;
	//initialize gpu device landmarks
	struct Landmark * d_lm_list;
	cudaMalloc((void**) &d_lm_list, map_landmarks.num_landmarks * sizeof(struct Landmark));
    cudaMemcpy(d_lm_list, &map_landmarks.landmarks[0], sizeof(struct Landmark) * map_landmarks.num_landmarks, cudaMemcpyHostToDevice);

	std::cout << "Map landmark finish" << std::endl;
	struct LandmarkData temp;
	temp.num_landmarks = map_landmarks.num_landmarks;
	temp.landmarks = d_lm_list;
	struct LandmarkData* d_map_landmarks;
	cudaMalloc((void**) &d_map_landmarks, 1 * sizeof(struct LandmarkData));
	cudaMemcpy(d_map_landmarks, &temp, 1 * sizeof(struct LandmarkData), cudaMemcpyHostToDevice);
	//a copy of LandmarkData is on device with address of: d_map_landmarks	

	std::cout << "BEGIN Landmark" << std::endl;
	//ground truth robot
	struct Robot my_robot;
	struct SensorData my_robot_sensor_data; //my_robot is real robot
	my_robot_sensor_data.num_sensor_readings = NUM_SENSOR_READINGS;
	my_robot_sensor_data.sensor_readings = new struct SensorRead[map_landmarks.num_landmarks];
	init_robot(&my_robot);
	std::cout << "Ground truth" << std::endl;
	//move robot and reassign to my_robot
	move_and_get_particle(&my_robot, MOVE_UPDATE_THETA, MOVE_UPDATE_X, &my_robot);
	std::cout << "MOVE AND GET PASS" << std::endl;
	sense(&my_robot, &map_landmarks, &my_robot_sensor_data); 
	std::cout << "SENSE PASS" << std::endl;	
	//allocate GPU device information for my_robot
	struct SensorData* d_my_robot_sensor_data;
	struct SensorRead* d_my_robot_sensor_read;	

	struct SensorData temp1;
	temp1.num_sensor_readings = my_robot_sensor_data.num_sensor_readings;
	cudaMalloc((void**) &d_my_robot_sensor_data, 1 * sizeof(struct SensorData));
	cudaMalloc((void**) &d_my_robot_sensor_read, sizeof(struct SensorRead) * my_robot_sensor_data.num_sensor_readings);
	//assign device pointer in struct to sensor readings array
	std::cout << "PASS SENSOR ALLOC" << std::endl;
	temp1.sensor_readings = d_my_robot_sensor_read;
	cudaMemcpy(d_my_robot_sensor_read, &(my_robot_sensor_data.sensor_readings[0]), sizeof(struct SensorRead) * my_robot_sensor_data.num_sensor_readings, cudaMemcpyHostToDevice);
	cudaMemcpy(d_my_robot_sensor_data, &temp1, sizeof(struct SensorData), cudaMemcpyHostToDevice);



	//sanity check
	std::cout << "AFTER HERE" << std::endl;
	if (temp1.sensor_readings != d_my_robot_sensor_read){
		std::cout << "ERROR SANITY CHECK FAIL" << std::endl;
		return -1;
	}else std::cout << "SANITY CHECK PASS" << std::endl;

	//setup and generate particles parallelized (prediction)
	int NUM_BLOCKS = ceil((float)N_PARTICLES / (float)DIVISIONS);
	int NUM_THREADS = DIVISIONS; 
	struct Robot * d_particle_list;
	struct Robot * h_particle_list = new struct Robot[N_PARTICLES]; //using serial resampling

	cudaMalloc((void**) &d_particle_list, N_PARTICLES * sizeof(struct Robot));
	//setup weight array host and device
	double* d_weight_prob;
	cudaMalloc((void**) &d_weight_prob, N_PARTICLES * sizeof(double));
	double* h_weight_prob = new double[N_PARTICLES];

	//setup resampling result array
	struct Robot * h_particle_list_p3 = new struct Robot[N_PARTICLES];
	struct Robot * d_particle_list_p3;
	cudaMalloc((void**) &d_particle_list_p3, sizeof(struct Robot) * N_PARTICLES);


	timestamp_t t0 = get_timestamp();
	generate_particles <<<NUM_BLOCKS, NUM_THREADS>>> (d_particle_list, NUM_THREADS, N_PARTICLES);


	for(int i = 0; i < N_ITERATIONS; i++){
	// for particle filter iterations
		//move ground truth robot
		move_and_get_particle(&my_robot, MOVE_UPDATE_THETA, MOVE_UPDATE_X, &my_robot);
		sense(&my_robot, &map_landmarks, &my_robot_sensor_data);
		// after on host robotic movement, copy sensor readings on host to device	
		//transfer host sensor readings to device
		cudaMemcpy(d_my_robot_sensor_read, &my_robot_sensor_data.sensor_readings[0], sizeof(struct SensorRead) * my_robot_sensor_data.num_sensor_readings, cudaMemcpyHostToDevice);

		particle_filter_one_iteration<<<NUM_BLOCKS, NUM_THREADS>>>(MOVE_UPDATE_THETA, MOVE_UPDATE_X, d_particle_list, d_my_robot_sensor_data, d_map_landmarks, d_weight_prob, METROPOLIS_B, N_PARTICLES); 

	}
	cudaDeviceSynchronize();
	timestamp_t t1 = get_timestamp();

	double diff = (double)t1 - (double)t0;
	std::cout << "Running TIME: " << diff << "microseconds" << std::endl;


	cudaMemcpy(h_weight_prob, d_weight_prob, sizeof(double) * N_PARTICLES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_particle_list, d_particle_list, sizeof(struct Robot) * N_PARTICLES, cudaMemcpyDeviceToHost);
	std::cout << "-----------------------" << std::endl;
	for (int j = 0; j < 10; j++){
		std::cout <<(float) h_weight_prob[j] << std::endl;
	}

	std::cout << "Error: " << eval_error(&my_robot, h_particle_list, N_PARTICLES) << std::endl;

	cudaFree(d_my_robot_sensor_data);
	cudaFree(d_my_robot_sensor_read);
	cudaFree(d_particle_list);
	cudaFree(d_weight_prob);
	cudaFree(d_lm_list);
	cudaFree(d_map_landmarks);
	

	delete[] my_robot_sensor_data.sensor_readings; 
	delete[] h_particle_list;
	delete[] h_weight_prob;
	delete[] h_particle_list_p3;
	delete[] map_landmarks.landmarks;
	return 0;
}

static timestamp_t get_timestamp(){
	struct timeval now;
	gettimeofday(&now, NULL);
	return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}
