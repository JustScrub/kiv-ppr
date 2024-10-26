#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <string>
#include <fstream>
#include <vector>
#include "comp_conf.hpp"

struct patient_data {
    std::vector<float> X, Y, Z;
    size_t id;
};

class data_loader {
public:
    data_loader(const std::string& filename);
    ~data_loader();
    int load_data(patient_data& data);
    int load_data(patient_data& data, size_t chunk_size);
private:
    std::string filename;
    std::ifstream file;
};

#endif // DATA_LOADER_HPP