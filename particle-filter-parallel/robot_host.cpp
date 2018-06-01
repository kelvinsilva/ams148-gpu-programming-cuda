#include "robot.h"
#include <math.h>
#include <time.h>

#define RANDOM_NUMBER_0_1 ((double) rand() / (RAND_MAX))

void init_robot(struct Robot* robot){
	srand(time(NULL));

	robot->x = (float) ((double)WORLDSIZE * RANDOM_NUMBER_0_1);
	robot->y = (float) ((double)WORLDSIZE * RANDOM_NUMBER_0_1);
	robot->orientation =  RANDOM_NUMBER_0_1 * PI * 2.0;

	robot->forward_noise = 0.0;
	robot->turn_noise = 0.0;
	robot->sense_noise = 0.0;

}
int set(struct Robot* robot, float new_x, float new_y, float new_orientation){
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
void set_noise(struct Robot* robot, float new_f_noise, float new_t_noise, float new_s_noise){
	robot->forward_noise = new_f_noise;
	robot->turn_noise = new_t_noise;
	robot->sense_noise = new_s_noise;
}

void sense(struct Robot* robot, struct LandmarkData* ld, struct SensorData * outld){
	int sz = ld->num_landmarks; 
	//struct SensorRead* out = (struct SensorRead *)malloc(sz * sizeof(float));

	for(int i = 0; i < sz; i++){
		float first = robot->x - ld->landmarks[i].x; first = first * first;
		float second = robot->y - ld->landmarks[i].y; second = second * second;
		(outld->sensor_readings[i]).x = sqrt( first + second);
		(outld->sensor_readings[i]).x += RANDOM_NUMBER_0_1;
	}
	outld->num_sensor_readings = ld->num_landmarks;
}

void move_and_get_particle(struct Robot * robot, float turn, float forward, struct Robot * particle_out_alloc){


	float orientation = robot->orientation + turn + RANDOM_NUMBER_0_1; // simple noise for now 
	orientation =  fmod(orientation, (float)2.0 * PI);
	float dist = forward + RANDOM_NUMBER_0_1 ; // leave out noise

	float x = robot->x + cos(orientation * dist);
	float y = robot->y + sin(orientation * dist);
	x = fmodf(x, (float)WORLDSIZE);
	y = fmodf(y, (float)WORLDSIZE); //cyclic map

	//particle_out_alloc = (struct Robot * ) malloc (sizeof(struct Robot));
	particle_out_alloc->x = x;
	particle_out_alloc->y = y;
	particle_out_alloc->orientation = orientation;

	set_noise(particle_out_alloc, robot->forward_noise, robot->turn_noise, robot->sense_noise);
	return;

}

float gaussian(float mu, float sigma, float x){

	float first = (mu - x); first = first * first;
	sigma = sigma * sigma;
	return exp(-1 * (first / sigma / 2.0) / sqrt(2.0 * PI * sigma));	
}

float calculate_measurement_probability(struct Robot* particle, struct SensorData * sd, struct LandmarkData * ld ){

	if (sd->num_sensor_readings != ld->num_landmarks){ //sensor read data must be same as landmark data

		return -1.0;
	}	
	float prob = 1.0;
	int sz = ld->num_landmarks;
	for(int i = 0; i < sz; i++){

		float first = particle->x - ld->landmarks[i].x; first = first * first;
		float second = particle->y - ld->landmarks[i].y; second = second * second;
		float dist = sqrt(first + second); 
	 	prob *= gaussian(dist, particle->sense_noise, sd->sensor_readings[i].x); 
	}
	return prob;
}

//single thread eval
double eval_error(struct Robot* r, struct Robot* particle_list, int n_particles){

	double sum = 0.0;
	for (int i = 0; i < n_particles; i++){
		
		double dx = (particle_list[i].x - r->x + fmod((WORLDSIZE / 2.0),  WORLDSIZE)) - (WORLDSIZE/2.0) ;
		double dy = (particle_list[i].y - r->y + fmod((WORLDSIZE / 2.0),  WORLDSIZE)) - (WORLDSIZE/2.0) ;
		double err = sqrt( dx * dx + dy * dy);
		sum += err;
	}	
	return sum / (float)n_particles;
}
