#ifndef CONF_LOADER_HPP
#define CONF_LOADER_HPP

#include <string>
#include <fstream>
#include <iostream>
#include <omp.h>

namespace Conf{

    size_t CHUNK_SIZE = 1024;
    size_t SIZE_INCR_STEPS = 10;
    unsigned NUM_REPS = 10;
    size_t CHUNK_CAPACITY = 1024/sizeof(float);
    unsigned MODES = 31;
    int NUM_THREADS = -1;

    constexpr size_t VEC_REG_CAP = 8;

    int load_conf(const std::string& conf_path){
        std::ifstream conf_file(conf_path);
        if(!conf_file.is_open()){
            std::cerr << "Failed to open config file: " << conf_path << std::endl;
            return 0;
        }
        std::string line;
        while(std::getline(conf_file, line)){
            if(line[0] == '#') continue;
            size_t pos = line.find('=');
            if(pos == std::string::npos) continue;
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos+1);
            if(key == "CHUNK_SIZE"){
                CHUNK_SIZE = std::stoul(value);
                CHUNK_CAPACITY = CHUNK_SIZE / sizeof(float);
            } else if(key == "DATA_LEN_STEPS"){
                SIZE_INCR_STEPS = std::stoul(value);
            } else if(key == "AVG_REPS"){
                NUM_REPS = std::stoul(value);
            } else if(key == "MODES"){
                MODES = std::stoul(value);
            } else if(key == "NUM_THREADS"){
                NUM_THREADS = std::stoi(value);
            }
        }
        if(NUM_THREADS == -1) {
            NUM_THREADS = omp_get_max_threads();
        }
        return 0;
    }

}


#endif // CONF_LOADER_HPP