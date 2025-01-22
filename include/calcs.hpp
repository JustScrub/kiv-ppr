#ifndef CALCS_HPP
#define CALCS_HPP

#include "conf_loader.hpp"
#include <stddef.h>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <array>
#include <sstream>
#include <algorithm>
#include <CL/cl.hpp>
#undef min
#undef max

namespace calcs {

    struct Chunk {
        size_t lo, hi;  // size: so that the whole chunk fits into cache
    };

    class Calc {
    public:
        virtual void calc(std::vector<float> &data_vector, size_t n, float *cv, float *mad) = 0;
        virtual float calc_time(std::vector<float> &data_vector, size_t n, unsigned reps, float* cv, float* mad);

		virtual ~Calc() = default;
    };

    class OMPCalc : public Calc {
    public:
        virtual void sum_chunk(const float* const data, Chunk& chunk, float* const out) = 0;
        virtual void varmed_chunk(float* const data, Chunk& chunk, float med, float mean, float* const out) = 0;
        virtual void abs_chunk(float* const data, Chunk& chunk) = 0;

        void reduce_merge_chunks(float* const data, Chunk* chunks, size_t n_chunks);
        float reduce_sum(float* data, size_t n);
        void sort_chunk(float* const data, Chunk& chunk);

        virtual void calc(std::vector<float> &data_vector, size_t n, float* cv, float* mad) override;

    };

    // sequential instructions
    class SeqCalc : public OMPCalc {
    public:
        void sum_chunk(const float* const data, Chunk& chunk, float* const out) override;
        void varmed_chunk(float* const data, Chunk& chunk, float med, float mean, float* const out) override;
        void abs_chunk(float* const data, Chunk& chunk) override;
    };

    // vectorized instructions
    class VecCalc : public OMPCalc {
    public:
        void sum_chunk(const float* const data, Chunk& chunk, float* const out) override;
        void varmed_chunk(float* const data, Chunk& chunk, float med, float mean, float* const out) override;
        void abs_chunk(float* const data, Chunk& chunk) override;
    };

    // OpenCL implementation
    class OCLCalc : public Calc {
    public:
        OCLCalc();

		static void prepare_data(std::vector<float>& data);

        void calc(std::vector<float> &data_vec, size_t n, float* cv, float* mad) override;
		float calc_time(std::vector<float>& data_vec, size_t n, unsigned reps, float* cv, float* mad);
        //void test_calc(std::vector<float> &data_vec, size_t n, float* cv, float* mad);

		void sort(cl::CommandQueue &q, size_t n, size_t work_group_size);
        /**
		* Reduces the partial sums in the buffer to a number of partial sums less than or equal to work_group_size.
		* The output is stored in the partial_sums buffer and must summed again to get the final result.
		* Returns the number of partial sums after the reduction.
        */
		size_t reduce_sum(cl::CommandQueue &q, cl::Buffer &b, size_t n, size_t work_group_size);

    private:
		cl::Platform platform;
        cl::Device device;
        cl::Context context;
        cl::Program program;
        cl::Kernel sum_reduce;
        cl::Kernel sort_ker;
        cl::Kernel varmed;
        cl::Kernel med_mean;
        cl::Kernel cv_calc;
		cl::Kernel reverse;
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