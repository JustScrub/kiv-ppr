#include "comp_conf.hpp"
#include "data_loader.hpp"
#include <iostream>

int main() {
    data_loader dl("data/ACC_001.csv");
    patient_data data;
    dl.load_data(data);
    return 0;
}