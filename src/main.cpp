#include "comp_conf.hpp"
#include "data_loader.hpp"
#include "calcs.hpp"
#include "json.hpp"
#include <chrono>
#include <iostream>
#include <omp.h>

using json = nlohmann::json;

int main(int argc, char *argv[]) {

    if(argc <3 ){
        std::cerr << "Usage: " << argv[0] << " <mode> <filename> ..." << std::endl;
        std::cerr << "mode: 0 - sequential, 1 - vectorized" << std::endl;
        return 1;
    }

    calcs::Calc *calc;

    if(atoi(argv[1]) == 0){
        calc = new calcs::SeqCalc();
    } else {
        calc = new calcs::VecCalc();
    }

    json j;
    j["mode"] = atoi(argv[1]) == 0 ? "sequential" : "vectorized";
    j["n_threads"] = omp_get_max_threads();
    j["results"] = json::array();

    patient_data data;
    float cv, mad;
    double t;
    for (int i = 2; i < argc; i++) {
        data_loader dl = data_loader(argv[i]);

        auto start = std::chrono::high_resolution_clock::now();
        if(dl.load_data(data) != 0){
            std::cerr << "Failed to load data from file: " << argv[i] << std::endl << std::endl;
            continue;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end-start;

        j["results"].push_back({
            {"file", argv[i]},
            {"size", data.X.size()},
            {"load_time", diff.count()},
            {"x", json::array()},       // includes objects: { data_length, avg_time, iters = [ { time, cv, mad } ] }
            {"y", json::array()},
            {"z", json::array()}
        });

        std::string dim_names[] = {"x", "y", "z"};

        std::cout << argv[i] << std::endl;

        for(int dim = 0; dim <3; dim++){
            size_t n_step = data[dim].size()/SIZE_INCR_STEPS + 1;

            for(size_t step = 1; step <= SIZE_INCR_STEPS; step++){
                size_t n = std::min(data[dim].size(), n_step*step);
                json res;
                res["data_length"] = n;
                res["iters"] = json::array();
                t = calc->calc_time(data[dim].data(), n, NUM_REPS, &cv, &mad, res);

                res["avg_time"] = t;
                j["results"].back()[dim_names[dim]].push_back(res);
                std::cout << dim_names[dim] << " - " << n << std::endl;
            }
        }

        data.X.clear();
        data.Y.clear();
        data.Z.clear();
    }

    std::string ofname = (j["n_threads"].get<int>() > 1 ? "par_" : "ser_") + j["mode"].get<std::string>().substr(0,3) + ".json";
    std::ofstream o( "out/" + ofname);
    o << j.dump(2) << std::endl;
    o.close();
}