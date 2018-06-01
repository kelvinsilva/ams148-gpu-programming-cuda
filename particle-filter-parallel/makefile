particle: particle_filter.cu robot.h robot.cu robot_host.cpp
	nvcc --std=c++11 particle_filter.cu robot.cu robot_host.cpp -dc
	nvcc -o particle_filter.exe particle_filter.o robot_host.o robot.o  -lcurand -arch=sm_60

