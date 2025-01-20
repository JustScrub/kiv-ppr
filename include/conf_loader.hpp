#ifndef CONF_LOADER_HPP
#define CONF_LOADER_HPP

#include <string>
#include <fstream>
#include <iostream>
#include <omp.h>

namespace Conf {

    extern size_t CHUNK_SIZE;
    extern size_t SIZE_INCR_STEPS;
    extern unsigned NUM_REPS;
    extern size_t CHUNK_CAPACITY;
    extern unsigned MODES;
    extern int NUM_THREADS;
    extern std::string plots_dir;
    extern std::string kernels_file;

    extern const size_t VEC_REG_CAP;
    extern std::string mode_names[];

    int load_conf(const std::string& conf_path);
}


#endif // CONF_LOADER_HPP