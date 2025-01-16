#include "conf_loader.hpp"
#include "data_loader.hpp"
#include "calcs.hpp"
#include <chrono>
#include <iostream>
#include <errno.h>
#include <omp.h>
#include <tuple>


int main(int argc, char *argv[]) {

    /*
    calcs::Calc *tcalc = new calcs::GpuCalc();
    tcalc->calc(nullptr, 0, nullptr, nullptr);
    */
   /*
    std::vector<size_t> x_values = {1, 200, 3000, 40000, 500000000};
    std::vector<float> line1 = {-3.234, -2.345, -1.456, -0.567, 0.321};
    std::vector<float> line2 = {0.123, 1.234, 2.345, 3.456, 4.567};
    std::vector<float> line3 = {4.567, 3.456, 2.345, 1.234, 6.3};
    std::vector<std::pair<std::string, std::vector<float>& >> lines_data = {
        {"GPU", line1},
        {"sequential parallel", line2},
        {"vectorized serial", line3}
    };

    std::string tsvg = calcs::plot_line_data_svg("title", x_values, lines_data);
    //write to file
    std::ofstream tfile("plot.svg");
    tfile << tsvg;
    tfile.close();
    exit(0);
    */


    if(argc <3 ){
        std::cerr << "Usage: " << argv[0] << " <config_path> <filename> ..." << std::endl;
        return 1;
    }

    if(Conf::load_conf(argv[1]) != 0){
        std::cerr << "Failed to load config file: " << argv[1] << std::endl;
        return 1;
    }

    calcs::Calc *calc;

    patient_data data;
    float cv, mad;
    float t;
    for (int i = 2; i < argc; i++) {
        data_loader dl = data_loader(argv[i]);
        std::cout << "processing: " << argv[i];

        auto start = std::chrono::high_resolution_clock::now();
        if(dl.load_data(data) != 0){
            std::cerr << "Failed to load data from file: " << argv[i] << std::endl << std::endl;
            continue;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> diff = end-start;
        std::cout << " (loaded in " << diff.count() << " s):" << std::endl;

        calcs::CalcData calc_data; // new for each file

        for(int m = 0; m < 5; m++) { //modes
            if((calc = calcs::calc_builder( Conf::MODES & (1<<m) )) == nullptr) continue;
            std::string mode_name = Conf::mode_names[m];

            calc_data[mode_name] = {
                std::make_tuple(std::vector<size_t>(), std::vector<float>(), std::vector<float>(), std::vector<float>()),
                std::make_tuple(std::vector<size_t>(), std::vector<float>(), std::vector<float>(), std::vector<float>()),
                std::make_tuple(std::vector<size_t>(), std::vector<float>(), std::vector<float>(), std::vector<float>())
            };

            std::string dim_names[] = {"x", "y", "z"};

            std::cout << "  " << mode_name << ":" << std::endl;
            for(size_t dim = 0; dim <3; dim++){
                size_t n_step = data[dim].size()/Conf::SIZE_INCR_STEPS + 1U;

                for(size_t step = 1; step <= Conf::SIZE_INCR_STEPS; step++){
                    size_t n = std::min(data[dim].size(), n_step*step);

                    t = calc->calc_time(data[dim].data(), n, Conf::NUM_REPS, &cv, &mad);

                    std::get<0>((calc_data[mode_name])[dim]).push_back(n);
                    std::get<1>((calc_data[mode_name])[dim]).push_back(t);
                    std::get<2>((calc_data[mode_name])[dim]).push_back(cv);
                    std::get<3>((calc_data[mode_name])[dim]).push_back(mad);
                    std::cout << "    " << dim_names[dim] << ": length " << n << "; avg time " << t << " s" << std::endl;
                } // data length steps
            } // dimensions
            delete calc;
        } // modes

        calcs::plot_line_data_svg("plots/" + std::to_string(i-1), calc_data);
        std::cout << calcs::calc_data_json_dump(calc_data, argv[i]) << std::endl;

        data.X.clear();
        data.Y.clear();
        data.Z.clear();
    }

}