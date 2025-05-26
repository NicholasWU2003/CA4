#
# Skeleton code for use with Computer Architecture 2023 assignment 4,
# LIACS, Leiden University.
#

CC = gcc
CXX = g++
CFLAGS = -O3 -g -Wall -std=gnu99
CXXFLAGS = -O3 -g -Wall -std=c++17 -fopenmp
LDFLAGS = -lm
PNGFLAGS = `pkg-config --cflags --libs libpng`

NVCC = nvcc
NVCCFLAGS = -O3 -std=c++14 --use_fast_math

# Add additional targets here:
TARGETS = \
		task1	\
		task2


all:		$(TARGETS)

task1:		task1.cu image.o
		$(NVCC) $(NVCCFLAGS) -o $@ $^ $(PNGFLAGS) $(LDFLAGS)

task2:		task2.cu image.o
		$(NVCC) $(NVCCFLAGS) -o $@ $^ $(PNGFLAGS) $(LDFLAGS)

# Could add additional rules here
#

%.o:		%.c
		$(CC) $(CFLAGS) -o $@ -c $^ $(PNGFLAGS) $(LDFLAGS)

%:		%.cpp image.o experiment.hpp
		$(CXX) $(CXXFLAGS) -o $@ $^ $(PNGFLAGS) $(LDFLAGS)

clean:
		rm -f $(TARGETS) image.o
