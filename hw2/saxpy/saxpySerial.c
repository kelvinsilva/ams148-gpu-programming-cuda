#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef unsigned long long timestamp_t;
static timestamp_t get_timestamp();


int main(){

	timestamp_t t0 = get_timestamp();
	long N = 65535;
	int *x = (int *) malloc(N * sizeof(int));
	int *y = (int *) malloc(N * sizeof(int));
	float a = 2.0;
	int i = 0;
	for (i = 0; i < N; i++){
		x[i] = 2.0;
		y[i] = 2.0;
	}
	for (i = 0; i < N; i++){
		y[i] += a * x[i]; // y = cx
	}
	
	free(x);
	free(y);
	timestamp_t t1 = get_timestamp();
	double diff = ((double)t1 - (double)t0);
	
	printf("Completed in : %lf microseconds\n", diff);	
	return 0;
}

static timestamp_t get_timestamp(){
	struct timeval now;
	gettimeofday(&now, NULL);
	return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}
