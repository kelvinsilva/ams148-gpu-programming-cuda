#ifndef ROBOT_H_
#define ROBOT_H_

#include <cstdlib>
#include <vector>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


#define PI 3.14159
#define WORLDSIZE 100

struct Robot{

	float x, y; // location
	float orientation;
	float forward_noise;
	float turn_noise;
	float sense_noise;
};

//sensor measurements or "landmarks"
struct Landmark{
	float x, y;
};

struct LandmarkData{

	struct Landmark* landmarks;
	int num_landmarks;
};

struct SensorRead{
	int x;
};

struct SensorData{
	struct SensorRead* sensor_readings;
	int num_sensor_readings;
};
/* GPU VERSION */
__device__ void gpu_init_robot(struct Robot* robot);
__device__ int gpu_set(struct Robot* robot, float new_x, float new_y, float new_orientation);
__device__ void gpu_set_noise(struct Robot* robot, float new_f_noise, float new_t_noise, float new_s_noise);

__device__ void gpu_sense(struct Robot* robot, struct LandmarkData* ld, struct SensorData * outld);

__device__ void gpu_move_and_get_particle(struct Robot * robot, float turn, float forward, struct Robot * particle_out_alloc);

__device__ float gpu_gaussian(float mu, float sigma, float x);

__device__ float gpu_calculate_measurement_probability(struct Robot* particle, struct SensorData * sd, struct LandmarkData * ld );

/* CPU VERSION */

void init_robot(struct Robot* robot);
int set(struct Robot* robot, float new_x, float new_y, float new_orientation);
void set_noise(struct Robot* robot, float new_f_noise, float new_t_noise, float new_s_noise);

void sense(struct Robot* robot, struct LandmarkData* ld, struct SensorData * outld);

void move_and_get_particle(struct Robot * robot, float turn, float forward, struct Robot * particle_out_alloc);

float gaussian(float mu, float sigma, float x);

float calculate_measurement_probability(struct Robot* particle, struct SensorData * sd, struct LandmarkData * ld );





#endif


