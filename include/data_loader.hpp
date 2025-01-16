#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <string>
#include <fstream>
#include <vector>

struct patient_data {
    std::vector<float> X, Y, Z;
    //size_t id;

    std::vector<float> &operator[](size_t i) {
        if(i == 0) return X;
        if(i == 1) return Y;
        if(i == 2) return Z;
        throw std::out_of_range("Index out of range");
    }
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