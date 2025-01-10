#ifndef CALCS_HPP
#define CALCS_HPP

#include <stddef.h>
#include <vector>
#include "json.hpp"
using json = nlohmann::json;


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

        double calc_time(float* const data, size_t n, unsigned reps, float* cv, float* mad, json &j);

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
    };

 
}

#endif // CALCS_HPP