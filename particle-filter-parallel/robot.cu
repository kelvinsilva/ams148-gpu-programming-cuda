#include "robot.h"

//initializes pose for robot given world size
__device__ void gpu_init_robot(struct Robot* robot){

	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	curandState state;
	curand_init((unsigned long long)clock() + tId, 0, 0, &state);

	float r = curand_uniform(&state);
	robot->x = (float)WORLDSIZE * curand_uniform(&state);
	robot->y = (float)WORLDSIZE * curand_uniform(&state);
	robot->orientation = curand_uniform(&state) * PI * 2.0;

	robot->forward_noise = 0.0;
	robot->turn_noise = 0.0;
	robot->sense_noise = 0.0;
}

__device__ int gpu_set(struct Robot* robot, float new_x, float new_y, float new_orientation){

	if( (new_x < 0) || (new_x >= WORLDSIZE) ){
		return -1; // error x coord oob
	}
	if ( (new_y < 0) || (new_y >= WORLDSIZE) ){
		return -1;
	}
	if ( (new_orientation < 0) || (new_orientation >= 2 * PI) ){
		return -1;
	}
	robot->x = new_x;
	robot->y = new_y;
	robot->orientation = new_orientation;
	return 1;
}

//sets noise parameters for robot
__device__ void gpu_set_noise(struct Robot* robot, float new_f_noise, float new_t_noise, float new_s_noise){

	robot->forward_noise = new_f_noise;
	robot->turn_noise = new_t_noise;
	robot->sense_noise = new_s_noise;

}

//routine for robotic "sensing" to be ran on the GPU
//takes in robot, landmark data, and returns a list of measurements to landmarks in the out pointer.
__device__ void gpu_sense(struct Robot* robot, struct LandmarkData* ld, struct SensorData * outld){
	int sz = ld->num_landmarks; 
	struct SensorRead* out = (struct SensorRead *)malloc(sz * sizeof(float));

	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	curandState state;
	curand_init((unsigned long long)clock() + tId, 0, 0, &state);


	for(int i = 0; i < sz; i++){
		float first = robot->x - ld->landmarks[i].x; first = first * first;
		float second = robot->y - ld->landmarks[i].y; second = second * second;
		out[i].x = sqrt( first + second);
		out[i].x += curand_uniform(&state);

	}
	outld->sensor_readings = out;
	outld->num_sensor_readings = sz;
};

//move robot and return a particle
__device__ void gpu_move_and_get_particle(struct Robot * robot, float turn, float forward, struct Robot * particle_out_alloc){

	
	if (forward < 0){
		particle_out_alloc = NULL;
	}
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	curandState state;
	curand_init((unsigned long long)clock() + tId, 0, 0, &state);


	float orientation = robot->orientation + turn + curand_log_normal(&state, 0.0, robot->turn_noise);
	orientation =  fmodf(orientation, (float)2.0 * PI);
	float dist = forward + curand_log_normal(&state, 0.0, robot->forward_noise);

	float x = robot->x + cos(orientation * dist);
	float y = robot->y + sin(orientation * dist);
	x = fmodf(x, (float)WORLDSIZE);
	y = fmodf(y, (float)WORLDSIZE); //cyclic map

	//particle_out_alloc = (struct Robot * ) malloc (sizeof(struct Robot));
	particle_out_alloc->x = x;
	particle_out_alloc->y = y;
	particle_out_alloc->orientation = orientation;

	gpu_set_noise(particle_out_alloc, robot->forward_noise, robot->turn_noise, robot->sense_noise);
	return;
}
 
//calculate the probability of x for 1 dim gaussian
__device__ float gpu_gaussian(float mu, float sigma, float x){

	float first = (mu - x); first = first * first;
	sigma = sigma * sigma;
	return exp(-1 * (first / sigma / 2.0) / sqrt(2.0 * PI * sigma));	

}

//calculate how likely a measurement will be for particle filtering of a particular particle
__device__ float gpu_calculate_measurement_probability(struct Robot* particle, struct SensorData * sd, struct LandmarkData * ld ){

	if (sd->num_sensor_readings != ld->num_landmarks){ //sensor read data must be same as landmark data

		return -1.0;
	}	
	float prob = 1.0;
	int sz = ld->num_landmarks;
	for(int i = 0; i < sz; i++){

		float first = particle->x - ld->landmarks[i].x; first = first * first;
		float second = particle->y - ld->landmarks[i].y; second = second * second;
		float dist = sqrt(first + second); 
	 	prob *= gpu_gaussian(dist, particle->sense_noise, sd->sensor_readings[i].x); 
	}
	return prob;
}

