#include "data_loader.hpp"
#include <sstream>
#include <iostream>

data_loader::data_loader(const std::string& filename) : filename(filename) {
    //std::cout << "Opening file: " << filename << std::endl;
    file.open(filename);
}

data_loader::~data_loader() {
    //std::cout << "Closing file: " << filename << std::endl;
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
    const char *cline;
    if(!std::getline(file, line)) {
        //skip header or fail if eof
        return -2;
    }
    while(std::getline(file, line)) {
        cline = line.c_str();
        while(*cline != ',') cline++;
        cline++;
        data.X.push_back(atof(cline));

        while(*cline != ',') cline++;
        cline++;
        data.Y.push_back(atof(cline));

        while(*cline != ',') cline++;
        cline++;
        data.Z.push_back(atof(cline));

        if(chunk_size > 0 && data.X.size() >= chunk_size) {
            break;
        }

    }
    return 0;
}
