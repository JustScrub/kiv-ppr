#include "calcs.hpp"
#include "comp_conf.hpp"
#include <algorithm>
#include <omp.h>
#include <immintrin.h>
#include <math.h>

namespace calcs {

    void Calc::calc(float* const data, size_t n, float* cv, float* mad){

        // n is the number of elements in the data array
        // CHUNK_SIZE is the size of the cache in bytes
        size_t n_chunks = (n / CHUNK_CAPACITY) + (size_t)(n % CHUNK_CAPACITY != 0);
        Chunk *chunks = new Chunk[n_chunks];
        float *sums = new float[n_chunks];

        // divide the data into chunks
        #pragma omp parallel for
        for(int i = 0; i < n_chunks; i++){
            chunks[i].lo = i*CHUNK_CAPACITY;
            chunks[i].hi = std::min(n, (i + 1)*CHUNK_CAPACITY);
        }

        // sum and sort each chunk
        #pragma omp parallel for
        for(int i = 0; i < n_chunks; i++){
            sums[i] = 0;
            sum_chunk(data, chunks[i], &sums[i]);
            sort_chunk(data, chunks[i]);
        }

        // merge chunks
        reduce_merge_chunks(data, chunks, n_chunks);

        float mean = reduce_sum(sums, n_chunks) / n;
        float med = n % 2 == 0 ? (data[n/2] + data[n/2 - 1]) / 2 : data[n/2];

        // calculate partial variances, offset chunks by median (varmed) and map to absolute values
        #pragma omp parallel for
        for(int i = 0; i < n_chunks; i++){
            sums[i] = 0;
            varmed_chunk(data, chunks[i], med, mean, &sums[i]);
            abs_chunk(data, chunks[i]);
        }

        // reverse bottom half of the array and merge it
        std::reverse(data, data + n/2);
        std::inplace_merge(data, data + n/2, data + n);

        // calculate coefficient of variation and median absolute deviation
        *cv = reduce_sum(sums, n_chunks) / n;
        *cv = sqrt(*cv)/mean;
        *mad = n % 2 == 0 ? (data[n/2] + data[n/2 - 1]) / 2 : data[n/2];

        delete[] chunks;
        delete[] sums;
    }

    void Calc::reduce_merge_chunks(float* const data, Chunk* chunks, size_t n_chunks){
        Chunk *_chunks = new Chunk[n_chunks];
        std::copy(chunks, chunks + n_chunks, _chunks);

        int start = 0, incr = 1;
        int to_merge = n_chunks;
        while(to_merge > 1){
            
            // if there are odd number of blocks to merge,
            // merge the first one to the second one and "throw away" the first block
            if(to_merge%2){
                std::inplace_merge(
                        data + _chunks[start].lo,
                        data + _chunks[start].hi,
                        data + _chunks[start+incr].hi
                    );
                _chunks[start+incr].lo = _chunks[start].lo;
                start += incr;
                to_merge--;
            } //to_merge is now even
            
            // take pairs of blocks and merge right one to the left one
            // this creates holes between still used chunk structs
            // "throw away" the right blocks (and update the left blocks)
            #pragma omp parallel for
            for(int i=start; i < start+to_merge*incr; i += 2*incr){
                std::inplace_merge(
                        data + _chunks[i].lo,
                        data + _chunks[i].hi,
                        data + _chunks[i+incr].hi
                    );
                _chunks[i].hi = _chunks[i+incr].hi;
            }
            // merging blocks creates holes in array of blocks
            // each step reduces the number of blocks two times, 
            // therefore we need to double the "steps" in our array
            incr *= 2;
            to_merge /= 2;
        }
        
        delete[] _chunks;

    }

    /* After use, data is invalidated (it is used as internal buffer) */
    float Calc::reduce_sum(float* data, size_t n){
        float sum = 0;
        
        size_t n_chunks = (n / CHUNK_CAPACITY) + (size_t)(n % CHUNK_CAPACITY != 0);
        float *_sums = new float[n_chunks];
        float *sums = _sums;
        float *tmp;

        while(n>1){
            #pragma omp parallel for
            for(int i = 0; i < n_chunks; i++){
                sums[i] = 0;
                Chunk chunk = {i*CHUNK_CAPACITY, std::min(n, (i + 1)*CHUNK_CAPACITY)};
                sum_chunk(data, chunk, &sums[i]);
            }
            n = n_chunks;
            n_chunks = (n / CHUNK_CAPACITY) + (size_t)(n % CHUNK_CAPACITY != 0);
            tmp = data;
            data = sums;
            sums = tmp;
        }

        sum = data[0];
        delete[] _sums;
        return sum;

    }

    void Calc::sort_chunk(float* const data, Chunk& chunk){
        std::sort(data + chunk.lo, data + chunk.hi);
    }



    /* ----------------- SEQUENTIAL INSTRUCTION IMPLEMENTATION -------------------- */

    void SeqCalc::sum_chunk(const float* const data, Chunk& chunk, float* const out){
        for(size_t i = chunk.lo; i < chunk.hi; i++){
            *out += data[i];
        }
    }

    void SeqCalc::varmed_chunk(float* const data, Chunk& chunk, float med, float mean, float* const out){
        for(size_t i = chunk.lo; i < chunk.hi; i++){
            *out += (data[i] - mean) * (data[i] - mean);
            data[i] -= med;
        }
    }

    void SeqCalc::abs_chunk(float* const data, Chunk& chunk){
        for(size_t i = chunk.lo; i < chunk.hi; i++){
            data[i] = fabs(data[i]);
        }
    }

    /* ----------------- VECTOR INSTRUCTION IMPLEMENTATION -------------------- */
    // by hand, not using OpenMP SIMD directives

    void VecCalc::sum_chunk(const float* const data, Chunk& chunk, float* const out){
        __m256 sum = _mm256_setzero_ps();
        for(size_t i = chunk.lo; i < chunk.hi; i += VEC_REG_CAP){
            __m256 vec = _mm256_loadu_ps(data + i);
            sum = _mm256_add_ps(sum, vec);
        }
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        *out = ((float*)&sum)[0] + ((float*)&sum)[4];
    }

    void VecCalc::varmed_chunk(float* const data, Chunk& chunk, float med, float mean, float* const out){
        __m256 _med = _mm256_set1_ps(med);
        __m256 _mean = _mm256_set1_ps(mean);
        __m256 sum = _mm256_setzero_ps();
        for(size_t i = chunk.lo; i < chunk.hi; i += VEC_REG_CAP){
            __m256 vec = _mm256_loadu_ps(data + i);
            __m256 diff = _mm256_sub_ps(vec, _mean);
            __m256 sq = _mm256_mul_ps(diff, diff);
            sum = _mm256_add_ps(sum, sq);

            diff = _mm256_sub_ps(vec, _med);
            _mm256_storeu_ps(data + i, diff);
        }
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        *out = ((float*)&sum)[0] + ((float*)&sum)[4];
    }

    void VecCalc::abs_chunk(float* const data, Chunk& chunk){
        __m256 mask = _mm256_set1_ps(-0.0f);
        for(size_t i = chunk.lo; i < chunk.hi; i += VEC_REG_CAP){
            __m256 vec = _mm256_loadu_ps(data + i);
            vec = _mm256_andnot_ps(mask, vec);
            _mm256_storeu_ps(data + i, vec);
        }
    }


}