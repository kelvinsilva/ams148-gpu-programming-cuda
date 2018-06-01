#include <stdio.h>
#include "robot.h"
#include <cuda.h>
#include <sys/time.h>
#include <algorithm>
#include <iostream>

//number of particles to run in particle filter
#define N_PARTICLES 4092 //256 min value 
//number of iterations of the particle filter
#define N_ITERATIONS 1000
//how many particles to assign to each thread block
#define DIVISIONS 1024 

//Robot parameters
#define MOVE_UPDATE_X 5.0 
#define MOVE_UPDATE_THETA 0.1 
#define NUM_LANDMARKS 4

#define PARTICLE_NOISE_F 0.05
#define PARTICLE_NOISE_T 0.05
#define PARTICLE_NOISE_S 5.00
const int NUM_SENSOR_READINGS = NUM_LANDMARKS; //landmarks = sensor readings enforce

typedef unsigned long long timestamp_t;
static timestamp_t get_timestamp();

// generate_particles <<<NUM_BLOCKS, NUM_THREADS>>> (d_particle_list, NUM_THREADS, N_PARTICLES);
__global__ void generate_particles( struct Robot* d_particle_list, int n_threads, int n_particles){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	struct Robot* r = &d_particle_list[i];
	gpu_init_robot(r);
	gpu_set_noise(r, PARTICLE_NOISE_F, PARTICLE_NOISE_T, PARTICLE_NOISE_S);
}

//motion update
//motion_update_parallel <<<NUM_BLOCKS, NUM_THREADS>>>(d_particle_list, N_PARTICLES); 

__global__ void motion_update_parallel (struct Robot* d_particle_list, float turn, float xdir, int particles){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	struct Robot* r = &d_particle_list[i];

	gpu_move_and_get_particle(r, turn, xdir, r);  
	__syncthreads();
}

//measurement_update_parallel <<<NUM_BLOCKS, NUM_THREADS>>>(d_particle_list, d_my_robot_sensor_data, d_map_landmarks, d_weight_prob, N_PARTICLES);

__global__ void measurement_update_parallel(struct Robot* d_particle_list, struct SensorData* d_my_robot_sensor_data, struct LandmarkData* d_map_landmarks, double* d_weight_prob, int n_particles){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	struct Robot* r = &d_particle_list[i];
    d_weight_prob[i] = gpu_calculate_measurement_probability(r, d_my_robot_sensor_data, d_map_landmarks);
	__syncthreads();
}	



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
	generate_particles <<<NUM_BLOCKS, NUM_THREADS>>> (d_particle_list, NUM_THREADS, N_PARTICLES);

	//setup weight array host and device
	double* d_weight_prob;
	cudaMalloc((void**) &d_weight_prob, N_PARTICLES * sizeof(double));
	double* h_weight_prob = new double[N_PARTICLES];

	//setup resampling result array
	struct Robot * h_particle_list_p3 = new struct Robot[N_PARTICLES];

	for(int i = 0; i < N_ITERATIONS; i++){
	// for particle filter iterations
		//move ground truth robot
		move_and_get_particle(&my_robot, MOVE_UPDATE_THETA, MOVE_UPDATE_X, &my_robot);
		sense(&my_robot, &map_landmarks, &my_robot_sensor_data);
		// after on host robotic movement, copy sensor readings on host to device	
		// cuda is syncronized on memcpy
		/*
		std::cout << "Sensor Data: " << std::endl;
		for (int i = 0; i < my_robot_sensor_data.num_sensor_readings; i++){
			std::cout << my_robot_sensor_data.sensor_readings[i].x << std::endl;
		}
		std::cout << "Robot Move: " << my_robot.x << " " << my_robot.y << std::endl;
		*/
		cudaMemcpy(d_my_robot_sensor_read, &my_robot_sensor_data.sensor_readings[0], sizeof(struct SensorRead) * my_robot_sensor_data.num_sensor_readings, cudaMemcpyHostToDevice);



		// do motion update parallelized	
		//cuda malloc and copy my robot measurement dat to device

		motion_update_parallel <<<NUM_BLOCKS, NUM_THREADS>>>(d_particle_list, MOVE_UPDATE_THETA, MOVE_UPDATE_X, N_PARTICLES); 
		//SYNC CUDA 
		cudaDeviceSynchronize();



		//do measurement update on particles (prediction), return weights for particles
		//float* measurements = new float[N_PARTICLES];

		measurement_update_parallel <<<NUM_BLOCKS, NUM_THREADS>>>(d_particle_list, d_my_robot_sensor_data, d_map_landmarks, d_weight_prob, N_PARTICLES);

		cudaDeviceSynchronize();

		// now need to do particle filter resampling
		// for simple case do a serial particle filter resampling. after code written, then do metropolis		

		// bring weights back to the host
		cudaMemcpy(h_weight_prob, d_weight_prob, sizeof(double) * N_PARTICLES, cudaMemcpyDeviceToHost);
		
		std::cout << "-----------------------" << std::endl;
		for (int j = 0; j < 10; j++){
			std::cout <<(float) h_weight_prob[j] << std::endl;
		}
		
		// bring the new particles back to the host from motion_update_parallel
		cudaMemcpy(h_particle_list, d_particle_list, sizeof(struct Robot) * N_PARTICLES, cudaMemcpyDeviceToHost);

		int index = rand() % (N_PARTICLES); // fix
		float beta = 0.0;
		float max_weight = *std::max_element(h_weight_prob, h_weight_prob + N_PARTICLES); 

		for (int i = 0; i < N_PARTICLES; i++){
			beta += (float)((double) rand() / (RAND_MAX)) * 2.0 * max_weight;
			while(beta > h_weight_prob[index]){
				
				beta -= h_weight_prob[index];
				index = (index + 1 ) % N_PARTICLES;
			}
			h_particle_list_p3[i] = h_particle_list[index];  //ith weight at ith particle
		}
		std::cout << "Particle List: x: " << h_particle_list_p3[0].x << " y: " << h_particle_list_p3[0].y << std::endl;
		//update particles on gpu after resampling
		std::cout << "Error: " << i << " th iteration: " << eval_error(&my_robot, h_particle_list_p3, N_PARTICLES) << std::endl;
		cudaMemcpy(d_particle_list, h_particle_list_p3, sizeof(struct Robot) * N_PARTICLES, cudaMemcpyHostToDevice);
		//
		

	}
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
