#ifndef CALCS_HPP
#define CALCS_HPP

#include "conf_loader.hpp"
#include <stddef.h>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <array>
#include <sstream>
#include <CL/cl.hpp>
#undef min
#undef max

namespace calcs {

    struct Chunk {
        size_t lo, hi;  // size: so that the whole chunk fits into cache
    };

    // set thread num via OMP_NUM_THREADS environment variable
    class Calc {
    public:
        virtual void sum_chunk(const float* const data, Chunk& chunk, float* const out) = 0;
        virtual void varmed_chunk(float* const data, Chunk& chunk, float med, float mean, float* const out) = 0;
        virtual void abs_chunk(float* const data, Chunk& chunk) = 0; 
        virtual void sort_chunk(float* const data, Chunk& chunk);
        /*
         * Merges sorted chunks of data into a single sorted array.
         */
        void reduce_merge_chunks(float* const data, Chunk* chunks, size_t n_chunks);

        float reduce_sum(float* data, size_t n);

        virtual void calc(float* const data, size_t n, float* cv, float* mad);

        float calc_time(float* const data, size_t n, unsigned reps, float* cv, float* mad);

    };

    // sequential instructions
    class SeqCalc : public Calc {
    public:
        void sum_chunk(const float* const data, Chunk& chunk, float* const out) override;
        void varmed_chunk(float* const data, Chunk& chunk, float med, float mean, float* const out) override;
        void abs_chunk(float* const data, Chunk& chunk) override;
    };

    // vectorized instructions
    class VecCalc : public Calc {
    public:
        void sum_chunk(const float* const data, Chunk& chunk, float* const out) override;
        void varmed_chunk(float* const data, Chunk& chunk, float med, float mean, float* const out) override;
        void abs_chunk(float* const data, Chunk& chunk) override;
    };

    // GPU using OpenCL
    class GpuCalc : public Calc {
    public:
        GpuCalc();

        void calc(float* const data, size_t n, float* cv, float* mad) override;

        void sum_chunk(const float* const data, Chunk& chunk, float* const out) override {
            // not implemented
        }
        void varmed_chunk(float* const data, Chunk& chunk, float med, float mean, float* const out) override {
            // not implemented
        }
        void abs_chunk(float* const data, Chunk& chunk) override {
            // not implemented
        }
        void test_calc(float* const data, size_t n, float* cv, float* mad);

    private:
		cl::Platform platform;
        cl::Device device;
        cl::Context context;
        cl::Program program;
        cl::Kernel sum_reduce;
        cl::Kernel sort_ker;
        cl::Kernel varmed;
        cl::Kernel med_mean;
        cl::Kernel final_calc;
    };

    /* 
     * Factory function for creating Calc objects.
     * mode: 
     * - 1 = sequential serial,
     * - 2 = sequential parallel,
     * - 4 = vectorized serial,
     * - 8 = vectorized parallel,
     * - 16 = GPU 
     * returns: pointer to Calc object or nullptr if mode is invalid
     */
    Calc *calc_builder(int mode);

    // structure for storing calculated data

    /*
    structure:
        [ mode: { 
                x: [ [ data lengths...], [ avg_times...], [ cv...], [ mad...]],
                y: ...,
                z: ... 
            }, 
            ... 
        ]

    One CalcData object per file.
    */

    using CalcDataArr = std::tuple<
        std::vector<size_t>,    // data lengths
        std::vector<float>,     // avg times
        std::vector<float>,     // cv
        std::vector<float>      // mad
    >;
    using CalcData = std::unordered_map<
        std::string,        // mode name
        std::array<
            CalcDataArr,
            3               // x, y, z  
        >
    >;

    std::string calc_data_json_dump(const CalcData& calc_data, const std::string file_name);
 
    using SvgLine = std::pair<std::string, const std::vector<float>& >; // line name, line data
    std::string plot_line_data_svg(
        const std::string title, 
        const std::vector<size_t>& x_values, 
        const std::vector<SvgLine>& lines
        );

    void plot_line_data_svg(
        const std::string file_prefix,
        const CalcData& calc_data
        );
}

#endif // CALCS_HPP