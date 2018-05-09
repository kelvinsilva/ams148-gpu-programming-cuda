#include <iostream>
#include <stdio.h> 
#include <sys/time.h>

//compile with -std=c++11
#define WIDTH 32768 
#define HEIGHT 32768 
typedef unsigned long long timestamp_t;
static timestamp_t get_timestamp();


int main(){
	using namespace std;

	auto *vector = new float[WIDTH];
	auto *matrix = new float[WIDTH][HEIGHT];
	auto *vector_result = new float[HEIGHT];
	for (int i = 0; i < HEIGHT; i++){
		
		for (int j = 0; j < WIDTH; j++){
			matrix[i][j] = 2.3;
			vector[j] = 2.1;
		}
	}
    timestamp_t t0 = get_timestamp();

	for (int i = 0; i < HEIGHT; i++){
		int accum = 0;
		for (int j = 0; j < WIDTH; j++){
			accum += vector[j] * matrix[i][j];	
		}
		vector_result[i] = accum;
		cout << vector_result[i] << endl;	
	}
	timestamp_t t1 = get_timestamp();
	double diff = ((double)t1 - (double)t0);

    printf("Completed in: %lf microseconds\n", diff);

}

static timestamp_t get_timestamp(){
	struct timeval now;
	gettimeofday(&now, NULL);
	return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}
