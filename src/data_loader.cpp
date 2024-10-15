#include "data_loader.hpp"
#include <sstream>

data_loader::data_loader(const std::string& filename) : filename(filename) {
    file.open(filename);
}

data_loader::~data_loader() {
    file.close();
}

//loaded csv includes date as string, x, y, z -> only x, y, z are loaded (float)

int data_loader::load_data(patient_data& data) {
    return load_data(data, -1);
}

int data_loader::load_data(patient_data& data, size_t chunk_size) {
    if (!file.is_open()) {
        return -1;
    }
    std::string line;
    if(!std::getline(file, line)) {
        //skip header or fail if eof
        return -2;
    }
    std::vector<float> values;
    while(std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        while(std::getline(ss, token, ',')) {
            values.push_back(stof(token));
        }
        data.X.push_back(values[1]);
        data.Y.push_back(values[2]);
        data.Z.push_back(values[3]);
        values.clear();
        if (!(--chunk_size)) {
            return 0;
        }
    }
    return 0;
}
