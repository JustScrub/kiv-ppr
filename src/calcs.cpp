#include "calcs.hpp"
#include "conf_loader.hpp"
#include <chrono>
#include <omp.h>
#include <immintrin.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <iomanip>

namespace calcs {

    void OMPCalc::calc(std::vector<float> &data_vector, size_t n, float* cv, float* mad) {
		float* data = data_vector.data();
        // n is the number of elements in the data array
        // CHUNK_SIZE is the size of the cache in bytes
        size_t n_chunks = (n / Conf::CHUNK_CAPACITY) + (size_t)(n % Conf::CHUNK_CAPACITY != 0);
        Chunk* chunks = new Chunk[n_chunks];
        float* sums = new float[n_chunks];

        // divide the data into chunks
#pragma omp parallel for
        for (size_t i = 0; i < n_chunks; i++) {
            chunks[i].lo = i * Conf::CHUNK_CAPACITY;
            chunks[i].hi = std::min(n, (i + 1) * Conf::CHUNK_CAPACITY);
        }

        // sum and sort each chunk
#pragma omp parallel for
        for (size_t i = 0; i < n_chunks; i++) {
            sums[i] = 0;
            sum_chunk(data, chunks[i], &sums[i]);
            sort_chunk(data, chunks[i]);
        }

        // merge chunks
        reduce_merge_chunks(data, chunks, n_chunks);

        float mean = reduce_sum(sums, n_chunks) / n;
        float med = n % 2 == 0 ? (data[n / 2] + data[n / 2 - 1]) / 2 : data[n / 2];

        // calculate partial variances, offset chunks by median (varmed) and map to absolute values
#pragma omp parallel for
        for (size_t i = 0; i < n_chunks; i++) {
            sums[i] = 0;
            varmed_chunk(data, chunks[i], med, mean, &sums[i]);
            abs_chunk(data, chunks[i]);
        }

        // reverse bottom half of the array and merge it
        std::reverse(data, data + n / 2);
        std::inplace_merge(data, data + n / 2, data + n);

        // calculate coefficient of variation and median absolute deviation
        *cv = sqrt(reduce_sum(sums, n_chunks) / n) / mean;
        *mad = n % 2 == 0 ? (data[n / 2] + data[n / 2 - 1]) / 2 : data[n / 2];

        delete[] chunks;
        delete[] sums;
    }

    void OMPCalc::reduce_merge_chunks(float* const data, Chunk* chunks, size_t n_chunks) {
        Chunk* _chunks = new Chunk[n_chunks];
        std::copy(chunks, chunks + n_chunks, _chunks);

        int start = 0, incr = 1;
        size_t to_merge = n_chunks;
        while (to_merge > 1) {

            // if there are odd number of blocks to merge,
            // merge the first one to the second one and "throw away" the first block
            if (to_merge % 2) {
                std::inplace_merge(
                    data + _chunks[start].lo,
                    data + _chunks[start].hi,
                    data + _chunks[start + incr].hi
                );
                _chunks[start + incr].lo = _chunks[start].lo;
                start += incr;
                to_merge--;
            } //to_merge is now even

            // take pairs of blocks and merge right one to the left one
            // this creates holes between still used chunk structs
            // "throw away" the right blocks (and update the left blocks)
#pragma omp parallel for
            for (int i = start; i < start + to_merge * incr; i += 2 * incr) {
                std::inplace_merge(
                    data + _chunks[i].lo,
                    data + _chunks[i].hi,
                    data + _chunks[i + incr].hi
                );
                _chunks[i].hi = _chunks[i + incr].hi;
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
    float OMPCalc::reduce_sum(float* data, size_t n) {
        float sum = 0;

        size_t n_chunks = (n / Conf::CHUNK_CAPACITY) + (size_t)(n % Conf::CHUNK_CAPACITY != 0);
        float* _sums = new float[n_chunks];
        float* sums = _sums;
        float* tmp;

        while (n > 1) {
#pragma omp parallel for
            for (size_t i = 0; i < n_chunks; i++) {
                sums[i] = 0;
                Chunk chunk = { i * Conf::CHUNK_CAPACITY, std::min(n, (i + 1) * Conf::CHUNK_CAPACITY) };
                sum_chunk(data, chunk, &sums[i]);
            }
            n = n_chunks;
            n_chunks = (n / Conf::CHUNK_CAPACITY) + (size_t)(n % Conf::CHUNK_CAPACITY != 0);
            tmp = data;
            data = sums;
            sums = tmp;
        }

        sum = data[0];
        delete[] _sums;
        return sum;

    }

    void OMPCalc::sort_chunk(float* const data, Chunk& chunk) {
        std::sort(data + chunk.lo, data + chunk.hi);
    }

    // median time of reps runs
    float Calc::calc_time(std::vector<float> &data_vector, size_t n, unsigned reps, float* cv, float* mad) {
		std::vector<float> times;
        for (unsigned i = 0; i < reps; i++) {
			std::vector<float> _data = data_vector;
            auto start = std::chrono::high_resolution_clock::now();
            calc(_data, n, cv, mad);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> diff = end - start;
			times.push_back(diff.count());
        }
		std::sort(times.begin(), times.end());
		return times.size() % 2 == 0 ? (times[times.size() / 2] + times[times.size() / 2 - 1]) / 2 : times[times.size() / 2];
    }



    /* ----------------- SEQUENTIAL INSTRUCTION IMPLEMENTATION -------------------- */

    void SeqCalc::sum_chunk(const float* const data, Chunk& chunk, float* const out) {
        float sum = 0;
        for (size_t i = chunk.lo; i < chunk.hi; i++) {
            sum += data[i];
        }
        *out = sum;
    }

    void SeqCalc::varmed_chunk(float* const data, Chunk& chunk, float med, float mean, float* const out) {
        float sum = 0;
        for (size_t i = chunk.lo; i < chunk.hi; i++) {
            sum += (data[i] - mean) * (data[i] - mean);
            data[i] -= med;
        }
        *out = sum;
    }

    void SeqCalc::abs_chunk(float* const data, Chunk& chunk) {
        for (size_t i = chunk.lo; i < chunk.hi; i++) {
            data[i] = fabs(data[i]);
        }
    }

    /* ----------------- VECTOR INSTRUCTION IMPLEMENTATION -------------------- */
    // by hand, not using OpenMP SIMD directives

    void VecCalc::sum_chunk(const float* const data, Chunk& chunk, float* const out) {
        __m256 sum = _mm256_setzero_ps();
        float ret = 0;
        size_t i;
        for (i = chunk.lo + Conf::VEC_REG_CAP; i <= chunk.hi; i += Conf::VEC_REG_CAP) {
            __m256 vec = _mm256_loadu_ps(data + i - Conf::VEC_REG_CAP);
            sum = _mm256_add_ps(sum, vec);
        }
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        ret = ((float*)&sum)[0] + ((float*)&sum)[4];

        // sum the rest -> max VEC_REG_CAP - 1 elements
        for (i = i - Conf::VEC_REG_CAP; i < chunk.hi; i++) {
            ret += data[i];
        }
        *out = ret;
    }

    void VecCalc::varmed_chunk(float* const data, Chunk& chunk, float med, float mean, float* const out) {
        // all __m256 variables altogether occupy 6*32 = 192 bytes (256 bits = 32 bytes)
        __m256 _med = _mm256_set1_ps(med);
        __m256 _mean = _mm256_set1_ps(mean);
        __m256 sum = _mm256_setzero_ps();
        size_t i;
        float ret = 0;
        for (i = chunk.lo + Conf::VEC_REG_CAP; i <= chunk.hi; i += Conf::VEC_REG_CAP) {
            __m256 vec = _mm256_loadu_ps(data + i - Conf::VEC_REG_CAP);
            __m256 diff = _mm256_sub_ps(vec, _mean);
            __m256 sq = _mm256_mul_ps(diff, diff);
            sum = _mm256_add_ps(sum, sq);

            diff = _mm256_sub_ps(vec, _med);
            _mm256_storeu_ps(data + i - Conf::VEC_REG_CAP, diff);
        }
        sum = _mm256_hadd_ps(sum, sum);
        sum = _mm256_hadd_ps(sum, sum);
        ret = ((float*)&sum)[0] + ((float*)&sum)[4];

        // do the rest -> max Conf::VEC_REG_CAP - 1 elements
        for (i = i - Conf::VEC_REG_CAP; i < chunk.hi; i++) {
            ret += (data[i] - mean) * (data[i] - mean);
            data[i] -= med;
        }
        *out = ret;
    }

    void VecCalc::abs_chunk(float* const data, Chunk& chunk) {
        __m256 mask = _mm256_set1_ps(-0.0f);
        size_t i;
        for (i = chunk.lo + Conf::VEC_REG_CAP; i <= chunk.hi; i += Conf::VEC_REG_CAP) {
            __m256 vec = _mm256_loadu_ps(data + i - Conf::VEC_REG_CAP);
            vec = _mm256_andnot_ps(mask, vec);
            _mm256_storeu_ps(data + i - Conf::VEC_REG_CAP, vec);
        }
        // do the rest -> max Conf::VEC_REG_CAP - 1 elements
        for (i = i - Conf::VEC_REG_CAP; i < chunk.hi; i++) {
            data[i] = fabs(data[i]);
        }

    }

    /* ----------------- FACTORY FUNCTION -------------------- */

    Calc* calc_builder(int mode) {
        switch (mode) {
        case 1:
            omp_set_num_threads(1);
            return new SeqCalc();
        case 2:
            omp_set_num_threads(Conf::NUM_THREADS);
            return new SeqCalc();
        case 4:
            omp_set_num_threads(1);
            return new VecCalc();
        case 8:
            omp_set_num_threads(Conf::NUM_THREADS);
            return new VecCalc();
        case 16:
            return new OCLCalc();
		case 32:
			return new OCLCalcCpuSort();
        default:
            return nullptr;
        }
    }

    /* ----------------- RESULTS OUTPUTS -------------------- */

    std::string calc_data_json_dump(const CalcData& calc_data, std::string file_name) {
        std::string json = "{\n";
        json += "\t\"file\": \"" + file_name + "\",\n";
        for (auto& [mode, data] : calc_data) {
            json += "\t\"" + mode + "\": {\n";
            for (size_t dim = 0; dim < 3; dim++) {
                json += "\t\t\"" + std::string(1, "xyz"[dim]) + "\": {\n";
                json += "\t\t\t\"lengths\": [";
                for (size_t i = 0; i < std::get<0>(data[dim]).size(); i++) {
                    json += std::to_string(std::get<0>(data[dim])[i]) + ", ";
                }
                json.pop_back(); json.pop_back();
                json += "],\n";
                json += "\t\t\t\"avg_times\": [";
                for (size_t i = 0; i < std::get<1>(data[dim]).size(); i++) {
                    json += std::to_string(std::get<1>(data[dim])[i]) + ", ";
                }
                json.pop_back(); json.pop_back();
                json += "],\n";
                json += "\t\t\t\"cv\": [";
                for (size_t i = 0; i < std::get<2>(data[dim]).size(); i++) {
                    json += std::to_string(std::get<2>(data[dim])[i]) + ", ";
                }
                json.pop_back(); json.pop_back();
                json += "],\n";
                json += "\t\t\t\"mad\": [";
                for (size_t i = 0; i < std::get<3>(data[dim]).size(); i++) {
                    json += std::to_string(std::get<3>(data[dim])[i]) + ", ";
                }
                json.pop_back(); json.pop_back();
                json += "]";
                json += "\n\t\t},\n";
            }
            json.pop_back();
            json.pop_back();
            json += "\n\t},\n";
        }
        json.pop_back();
        json.pop_back();
        json += "\n}\n";
        return json;
    }

    std::string plot_line_data_svg(
        const std::string title, 
        const std::vector<size_t>& x_values, 
        const std::vector<SvgLine>& lines
        ) {
        std::ostringstream svg;
        float margin = 70;
        float left_margin = 70;
        float width = 1000;
        float height = 600;

        svg << "<svg width=\"" << width << "\" height=\"" << height << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
        svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";

        // Draw title
        svg << "<text x=\"" << width / 2 << "\" y=\"" << margin / 2 << "\" font-size=\"30\" text-anchor=\"middle\">" << title << "</text>\n";

        // Draw axes
        svg << "<line x1=\"" << left_margin << "\" y1=\"" << margin << "\" x2=\"" << left_margin << "\" y2=\"" << height - margin << "\" stroke=\"black\"/>\n";
        svg << "<line x1=\"" << left_margin << "\" y1=\"" << height - margin << "\" x2=\"" << width - margin << "\" y2=\"" << height - margin << "\" stroke=\"black\"/>\n";

        // Draw X-axis labels and grid
        // skew labels to avoid overlap
        for (size_t i = 0; i < x_values.size(); ++i) {
            float x = left_margin + (i + 1) * (width - margin - left_margin) / (x_values.size() + 1);
            float y = height - margin / 2;
            svg << "<text x=\"" << x << "\" y=\"" << y << "\" font-size=\"20\" text-anchor=\"middle\" transform=\"rotate(-45 " << x << " " << y << ")\">" << x_values[i] << "</text>\n";
            svg << "<line x1=\"" << x << "\" y1=\"" << margin << "\" x2=\"" << x << "\" y2=\"" << height - margin << "\" stroke=\"lightgray\"/>\n";
        }

        // Determine scaling factor
        float y_min = std::numeric_limits<float>::max();
        float y_max = std::numeric_limits<float>::min();

        for (const auto& line : lines) {
            y_min = std::min(y_min, *std::min_element(line.second.begin(), line.second.end()));
            y_max = std::max(y_max, *std::max_element(line.second.begin(), line.second.end()));
        }

        float y_scale = (height - 2 * margin) / (y_max - y_min);

        // Draw Y-axis labels and grid
        for (size_t i = 0; i < 5; ++i) {
            float y = height - margin - i * (height - 2 * margin) / 5;
            svg << "<text x=\"" << left_margin << "\" y=\"" << y << "\" font-size=\"20\" text-anchor=\"end\">" << std::fixed << std::setprecision(3) << y_min + i * (y_max - y_min) / 5 << "</text>\n";
            svg << "<line x1=\"" << left_margin << "\" y1=\"" << y << "\" x2=\"" << width - margin << "\" y2=\"" << y << "\" stroke=\"lightgray\"/>\n";
        }

        // Set collor pallette
        std::string colors[] = { "red", "green", "blue", "orange", "purple", "gold"};
        size_t i = 0;
        // Plot lines
        for (const auto& line : lines) {
            svg << "<polyline fill=\"none\" stroke=\"" << colors[i++] << "\" points=\"";
            for (size_t j = 0; j < line.second.size(); ++j) {
                float x = left_margin + (j + 1) * (width - margin - left_margin) / (x_values.size() + 1);
                float y = height - margin - (line.second[j] - y_min) * y_scale;
                svg << x << "," << y << " ";
            }
            svg << "\"/>\n";
        }

        // Draw legend -- text is same color as line
        // below title, in line
        i = 0;
        float legend_x = width - margin;
        float legend_y = margin;
        for (const auto& line : lines) {
            svg << "<text x=\"" << legend_x << "\" y=\"" << legend_y << "\" font-size=\"20\" text-anchor=\"end\" fill=\"" << colors[i++] << "\">" << line.first << "</text>\n";
            legend_y += 30;
        }

        svg << "</svg>\n";
        return svg.str();
    }

    void plot_line_data_svg(
        const std::string out_file_prefix,
        const CalcData& calc_data
        ) {
            std::vector<SvgLine> avgtime_lines;
            std::vector<SvgLine> cv_lines;
            std::vector<SvgLine> mad_lines;

            for(size_t dim = 0; dim < 3; ++dim){
                avgtime_lines.clear();
                cv_lines.clear();
                mad_lines.clear();
                std::string mode_name;
                // collect data for each mode and each metric
                for(int mode = 0; mode < 6; mode++){ 
                    if( (Conf::MODES & (1 << mode)) == 0 ) continue; // skip if mode not enabled
                    mode_name = Conf::mode_names[mode];
                    if(calc_data.find(mode_name) == calc_data.end()) continue; // skip if data not found

                    avgtime_lines.push_back(std::make_pair(mode_name, std::ref(std::get<1>(calc_data.at(mode_name)[dim]))));
                    cv_lines.push_back(std::make_pair(mode_name, std::ref(std::get<2>(calc_data.at(mode_name)[dim]))));
                    mad_lines.push_back(std::make_pair(mode_name, std::ref(std::get<3>(calc_data.at(mode_name)[dim]))));
                }
                if(avgtime_lines.empty()) continue; // skip if no data found

                // generate SVGs
                std::string fname = out_file_prefix + "_" + std::string(1, "XYZ"[dim]) + "_";
                const std::vector<size_t> &x_values = std::get<0>(calc_data.at(mode_name)[dim]);
                // average time
                (std::ofstream(fname + "medtime.svg") << plot_line_data_svg(
                        "Median time", 
                        x_values, 
                        avgtime_lines)
                ).close();
                // coefficient of variation
                (std::ofstream(fname + "cv.svg") << plot_line_data_svg(
                        "Coefficient of variation", 
                        x_values, 
                        cv_lines)
                ).close();
                // median absolute deviation
                (std::ofstream(fname + "mad.svg") << plot_line_data_svg(
                        "Median absolute deviation", 
                        x_values, 
                        mad_lines)
                ).close();

                }
        }
} // namespace calcs
