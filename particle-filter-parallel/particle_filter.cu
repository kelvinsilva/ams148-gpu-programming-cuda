#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

//number of particles to run in particle filter
#define N_PARTICLES 1000
//number of iterations of the particle filter
#define N_ITERATIONS 10
//how many particles to assign to each thread block
#define DIVISIONS 256 

typedef unsigned long long timestamp_t;
static timestamp_t get_timestamp();

int main(){

	//initialize landmarks
	struct LandmarkData map_landmarks;
	map_landmarks.size = 4;
   	map_landmarks.landmarks = new struct Landmark[map_landmarks.size];

	map_landmarks.landmarks[0].x = 20.0;
	map_landmarks.landmarks[0].y = 20.0;

	map_landmarks.landmarks[1].x = 80.0;
	map_landmarks.landmarks[1].y = 80.0;

	map_landmarks.landmarks[2].x = 20.0;
	map_landmarks.landmarks[2].y = 80.0;

	map_landmarks.landmarks[3].x = 80.0;
	map_landmarks.landmarks[3].y = 20.0;

	//initialize gpu device landmarks
	struct Landmark * d_lm_list;
	cudaMalloc(d_lm_list, map_landmarks.size, sizeof(struct Landmark));
    cudaMemcpy(d_lm_list, map_landmarks.landmarks, sizeof(struct Landmark) * map_landmarks.size, cudaMemcpyHostToDevice);

	struct LandmarkData temp;
	temp.size = map_landmarks.size;
	temp.landmarks = d_lm_list;
	struct LandmarkData* d_map_landmarks;
	cudaMalloc(d_map_landmarks, 1, sizeof(struct Landmark));
	cudaMemcpy(d_map_landmarks, &temp, sizeof(struct LandmarkData), cudaMemcpyHostToDevice);
	//a copy of LandmarkData is on device with address of: d_map_landmarks	


	//ground truth robot
	struct Robot my_robot;
	struct SensorData my_robot_sensor_data; //my_robot is real robot
	my_robot_sensor_data.size = 4;
	my_robot_sensor_data.sensor_readings = new struct SensorRead[map_landmarks.size];
	init_robot(&myRobot);

	//move robot and reassign to my_robot
	move_and_get_particle(&my_robot, 0.1, 5.0, &my_robot);
	sense(&my_robot, &map_landmarks, &my_robot_sensor_data); 
	
	//allocate GPU device information for my_robot
	struct SensorData* d_my_robot_sensor_data;
	struct SensorRead* d_my_robot_sensor_read;	

	cudaMalloc(d_my_robot_sensor_data, 1, sizeof(struct SensorData));
	cudaMalloc(d_my_robot_sensor_read, my_robot_sensor_data.num_sensor_readings, sizeof(int) * my_robot_sensor_data.num_sensor_readings);
	//assign device pointer in struct to sensor readings array
	cudaMemcpy(&(d_my_robot_sensor_data->sensor_readings), &(d_my_robot_sensor_read), sizeof(struct SensorRead *), cudaMemcpyHostToDevice); 	
	cudaMemcpy(d_my_robot_sensor_read, my_robot_sensor_data.sensor_readings, sizeof(struct SensorRead) * my_robot_sensor_data.num_sensor_readings, cudaMemcpyHostToDevice);


	//setup and generate particles parallelized (prediction)
	int NUM_BLOCKS = N_PARTICLES / DIVISIONS;
	int NUM_THREADS = DIVISIONS; 
	struct Robot * d_particle_list;
	cudaMalloc(d_particle_list, N_PARTICLES, sizeof(struct Robot));
	//cudamalloc d_particle list
	generate_particles <<<NUM_BLOCKS, NUM_THREADS>>> (d_particle_list, N_THREADS, N_PARTICLES);
	
	for(int i = 0; i < N_ITERATIONS; i++){
	// for particle filter iterations
		//move ground truth robot
		move_and_get_particle(&my_robot, 0.1, 5.0, &my_robot);
		sense(&my_robot, &map_landmarks, &my_robot_sensor_data);
		// after on host robotic movement, copy sensor readings on host to device	
		cudaMemcpy(d_my_robot_sensor_read, my_robot_sensor_data.sensor_readings, sizeof(struct SensorRead) * my_robot_sensor_data.num_sensor_readings, cudaMemcpyHostToDevice);
		// do motion update parallelized	
		//cuda malloc and copy my robot measurement dat to device
		motion_update_parallel <<<NUM_BLOCKS, NUM_THREADS>>>(d_particle_list); 

		//SYNC CUDA 
		cudaSync()
		//do measurement update on particles (prediction)
		float* measurements = new float[N_PARTICLES];
		float* d_measurements;
		cudaMalloc(d_measurements, N_PARTICLES, sizeof(float));
		measurement_update_parallel <<<NUM_BLOCKS, NUM_THREADS>>>(d_particle_list, d_my_robot_sensor_data, d_map_landmarks, d_measurements);

		//
	}

	
		

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
	delete map_landmarks->landmarks [];
	return 0;
}

static timestamp_t get_timestamp(){
	struct timeval now;
	gettimeofday(&now, NULL);
	return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}
