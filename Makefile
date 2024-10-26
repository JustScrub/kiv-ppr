INCLUDES = -I./include

all: main

main: src/main.cpp src/data_loader.cpp
	g++ -std=c++17 -o main.out src/main.cpp src/data_loader.cpp $(INCLUDES)