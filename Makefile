INCLUDES = -I./include
SRCS = src/main.cpp src/data_loader.cpp src/calcs.cpp
ERRORS = -Wall -Wextra -Werror -Wpedantic

all: main

main: $(SRCS)
	g++ -std=c++17 -mavx2 -fopenmp $(ERRORS) -o $@.out $(SRCS) $(INCLUDES)