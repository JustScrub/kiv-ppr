#include "comp_conf.hpp"
#include "data_loader.hpp"
#include "calcs.hpp"
#include <chrono>
#include <iostream>
#include <errno.h>
#include <omp.h>
#include <tuple>


int main(int argc, char *argv[]) {

    calcs::Calc *tcalc = new calcs::GpuCalc();
    tcalc->calc(nullptr, 0, nullptr, nullptr);

    exit(0);

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

    std::string calc_mode = atoi(argv[1]) == 0 ? "sequential" : "vectorized";
    int n_threads = omp_get_max_threads();
    calcs::CalcData calc_data;

    patient_data data;
    float cv, mad;
    double t;
    for (int i = 2; i < argc; i++) {
        data_loader dl = data_loader(argv[i]);
        std::cout << "processing: " << argv[i];

        auto start = std::chrono::high_resolution_clock::now();
        if(dl.load_data(data) != 0){
            std::cerr << "Failed to load data from file: " << argv[i] << std::endl << std::endl;
            continue;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end-start;

		std::cout << " (loaded in " << diff.count() << " s):" << std::endl;

        calc_data[argv[i]] = {calcs::CalcDataArr(), calcs::CalcDataArr(), calcs::CalcDataArr()};

        std::string dim_names[] = {"x", "y", "z"};

        for(size_t dim = 0; dim <3; dim++){
            size_t n_step = data[dim].size()/SIZE_INCR_STEPS + 1U;

            for(size_t step = 1; step <= SIZE_INCR_STEPS; step++){
                size_t n = std::min(data[dim].size(), n_step*step);

                t = calc->calc_time(data[dim].data(), n, NUM_REPS, &cv, &mad);

                (calc_data[argv[i]])[dim].push_back(std::make_tuple(n, t, cv, mad));
                std::cout << "  " << dim_names[dim] << ": length " << n << "; avg time " << t << " s" << std::endl;
            }
        }

        data.X.clear();
        data.Y.clear();
        data.Z.clear();
    }

    std::cout << calcs::calc_data_json_dump(calc_data) << std::endl;
}