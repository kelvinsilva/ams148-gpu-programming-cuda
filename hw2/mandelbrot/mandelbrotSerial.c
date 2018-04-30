 #include <stdio.h>
 #include <stdlib.h>
 #include <sys/time.h>

 typedef unsigned long long timestamp_t;
 static timestamp_t get_timestamp();


 void render(char * out, const int width, const int height, const int max_iter) {
 
      float x_origin, y_origin, xtemp, x, y;
      int iteration, index;
 
      for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++){
 
            index = 3 * width * j + i * 3;
            iteration = 0;
            x = 0.0f;
            y = 0.0f;
            x_origin = ((float) i / width) * 3.25f - 2.0f;
            y_origin = ((float) j / width) * 2.5f - 1.25f;
 
            while (x * x + y * y <= 4 && iteration < max_iter) {
              xtemp = x * x - y * y + x_origin;
              y = 2 * x * y + y_origin;
              x = xtemp;
              iteration++;
            }
 
            if (iteration == max_iter) {
              out[index] = 0;
              out[index + 1] = 0;
              out[index + 2] = 0;
            } else {
              out[index] = iteration;
              out[index + 1] = iteration;
              out[index + 2] = iteration;
            }
          }
      }
  }

 char** char2d(int N, int M){
         char **a = malloc(sizeof *a * N);
         if (a)
         {
           for (int i = 0; i < N; i++)
           {
             a[i] = malloc(sizeof *a[i] * M);
           }
         }
         return a;
 }
 int main(){

    timestamp_t t0 = get_timestamp();
    int N = 8192;
    char * outArr = malloc(sizeof(char) * N * N * 3);

    render(outArr, N, N, 512);
    timestamp_t t1 =  get_timestamp();
    double diff = ((double)t1 - (double)t0);
    printf("Time: %lf", diff);
    return 0;
 }

 static timestamp_t get_timestamp(){
         struct timeval now;
         gettimeofday(&now, NULL);
         return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
 }

