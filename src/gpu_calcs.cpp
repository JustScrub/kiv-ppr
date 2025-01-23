#include "calcs.hpp"
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

namespace calcs {

	OCLCalc::OCLCalc() {
		cl_int err;
		// prep platform, device, context, and kernels
		err = cl::Platform::get(&(this->platform));
		if (err != CL_SUCCESS) {
			std::cerr << "OpenCL: Failed to get platform" << std::endl;
			exit(1);
		}

		std::vector<cl::Device> devices;
		err = this->platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		if (err != CL_SUCCESS) {
			std::cerr << "OpenCL: Failed to get devices" << std::endl;
			exit(1);
		}
		this->device = devices.front();
		this->context = cl::Context(this->device);

		// prep kernels
		std::ifstream kfile(Conf::kernels_file);
		if (!kfile.is_open()) {
			std::cerr << "OpenCL: Failed to open kernels file" << std::endl;
			exit(1);
		}

		std::string kernel_src((std::istreambuf_iterator<char>(kfile)), std::istreambuf_iterator<char>());
		this->program = cl::Program(this->context, kernel_src, true, &err); // true builds the program
		kfile.close();
		if (err != CL_SUCCESS) {
			std::cerr << "OpenCL: Failed to create program" << std::endl;
			exit(1);
		}

		this-> sum_reduce = cl::Kernel(program, "sum_reduce");
		this->   sort_ker = cl::Kernel(program, "sort");
		this->   med_mean = cl::Kernel(program, "get_med_mean");
		this->     varmed = cl::Kernel(program, "varmed");
		this->    cv_calc = cl::Kernel(program, "calc_cv");
		this->    reverse = cl::Kernel(program, "reverse");
	}

#define get_n_chunks(n, work_group_size) ( ((n)/(work_group_size)) + (((n) % (work_group_size)) != 0) )
#define get_global_size(n, work_group_size) get_n_chunks(n, work_group_size) * (work_group_size)
#define is_pow2(n) ( ((n) & ((n) - 1)) == 0 )

	void OCLCalc::prepare_data(std::vector<float>& data) {
		// pad data to a power of 2
		size_t n = data.size();
		size_t n_pow_padded = 1;
		while (n_pow_padded < n) n_pow_padded *= 2;
		data.resize(n_pow_padded, std::numeric_limits<float>::max());
	}

	void OCLCalc::calc(std::vector<float> &data_vector, size_t n, float* cv, float* mad) {
		cl_int err;

		float* data = data_vector.data();

		// determine number of work groups -- for partial sums
		size_t work_group_size, tmp, work_group_n, global_size;
		sum_reduce.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &work_group_size);
		varmed.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &tmp);
		work_group_size = tmp < work_group_size ? tmp : work_group_size;
		work_group_n = get_n_chunks(n, work_group_size);
		global_size = get_global_size(n, work_group_size);

		// prep buffers
		/*
		Buffers:
		------------------
		1. The input array is stored in global memory (size n)
		2. The partial sums are stored in global memory (size same as number of work groups)
		3. The local sums are stored in local memory (size same as number of work items in a work group)
		4. The temporary array used in the final kernel is stored in local memory (size n/2)
		5. The med_mean_buffer is stored in global memory (size 2)
		 */
		cl::Buffer input_buff(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * data_vector.size(), data);
		cl::Buffer partial_sums(context, CL_MEM_READ_WRITE, sizeof(float) * work_group_n);
		cl::Buffer med_mean_buffer(context, CL_MEM_READ_WRITE, sizeof(float) * 2);

		// set unchanging kernel arguments
		sum_reduce.setArg(3, cl::Local(sizeof(float) * work_group_size));
		// the rest is set for initial sum and then in the reduction procedure

		sort_ker.setArg(0, input_buff);
		// the rest of the arguments are set in the sorting procedure function

		med_mean.setArg(0, input_buff);
		med_mean.setArg(1, partial_sums); // partial sums to sum
		med_mean.setArg(2, med_mean_buffer); // output
		// arg 3 to be set as the return value of sum reduction procedure
		med_mean.setArg(4, n); // length of input

		varmed.setArg(0, input_buff);
		varmed.setArg(1, n);
		varmed.setArg(2, partial_sums);
		varmed.setArg(3, cl::Local(sizeof(float) * work_group_size));
		varmed.setArg(4, med_mean_buffer);

		cv_calc.setArg(0, partial_sums);
		// arg 1 to be set as the return value of sum reduction procedure after varmed
		cv_calc.setArg(2, med_mean_buffer);
		cv_calc.setArg(3, n);

		reverse.setArg(0, input_buff);
		reverse.setArg(1, n / 2); // only reverse the first half

		float* output = new float[2];
		size_t partial_n;
		// execute kernels
		cl::CommandQueue queue(context, device);

		// sort + sum 
		sort(queue, data_vector.size(), work_group_size);
		sum_reduce.setArg(0, input_buff);
		sum_reduce.setArg(1, n);
		sum_reduce.setArg(2, partial_sums);
		err = queue.enqueueNDRangeKernel(sum_reduce, cl::NullRange, cl::NDRange(global_size), cl::NDRange(work_group_size));
		partial_n = reduce_sum(queue, partial_sums, work_group_n, work_group_size);

		// calculate median and mean
		med_mean.setArg(3, partial_n);
		err = queue.enqueueNDRangeKernel(med_mean, cl::NullRange, cl::NDRange(1), cl::NullRange); // do this work on device rather than on host to avoid moving partial sums

		// calculate varmed
		err = queue.enqueueNDRangeKernel(varmed, cl::NullRange, cl::NDRange(global_size), cl::NDRange(work_group_size));
		partial_n = reduce_sum(queue, partial_sums, work_group_n, work_group_size);

		// calculate cv
		cv_calc.setArg(1, partial_n);
		err = queue.enqueueNDRangeKernel(cv_calc, cl::NullRange, cl::NDRange(1), cl::NullRange); // same as before

		// reverse bottom half of the array and merge it, get cv and calculate mad
		err = queue.enqueueNDRangeKernel(reverse, cl::NullRange, cl::NDRange(n / 4), cl::NullRange);
		queue.enqueueReadBuffer(med_mean_buffer, CL_TRUE, 0, sizeof(float) * 2, output);
		queue.enqueueReadBuffer(input_buff, CL_TRUE, 0, sizeof(float) * n, data);
		
		//this cannot be done in parallel :(
		//and calculating in a single work-item is inefficient --> move data to host and calculate there
        std::inplace_merge(data, data + n / 2, data + n);

		*cv = output[0]; // calculated by cv_calc
        *mad = n % 2 == 0 ? (data[n / 2] + data[n / 2 - 1]) / 2 : data[n / 2];
	}

	void OCLCalc::sort(cl::CommandQueue &q, size_t n, size_t work_group_size) {

		if (!is_pow2(n)) {
			std::cerr << "GPU sort implementation: Data array length must be a power of 2" << std::endl;
			std::cerr << "(actual data must be padded using calcs::OCLCalc::prepare_data)" << std::endl;
			exit(1);
		}

		unsigned int stage, substage, n_stages;
		stage = substage = n_stages = 0;
		for(size_t tmp = n; tmp > 1; tmp /= 2)
			++n_stages;
		size_t global_size = get_global_size(n/2, work_group_size);
		// assuming input arg already set!

		for (stage = 0; stage < n_stages; ++stage) {
			sort_ker.setArg(1, stage);
			for (substage = 0; substage < stage + 1; ++substage) {
				sort_ker.setArg(2, substage);
				q.enqueueNDRangeKernel(sort_ker, cl::NullRange, cl::NDRange(global_size), cl::NDRange(work_group_size));
			}
		}
	}

	size_t OCLCalc::reduce_sum(cl::CommandQueue& q, cl::Buffer &partial_sums, size_t partial_n, size_t work_group_size) {
		// repeatedly sum partial sums until less than work_group_size remain
		// before call, initial partial summation has been done (either by sum_reduce or varmed)
		// so only need to sum partial sums
		size_t n_chunks = get_n_chunks(partial_n, work_group_size);
		cl::Buffer tmp_buffer(context, CL_MEM_READ_WRITE, sizeof(float) * n_chunks);

		unsigned step_parity = 0; // even = summing partial_sums to tmp_buffer, odd = summing tmp_buffer to partial_sums

		// as opposed to OMP implementation, this reduction does not reduce into a single value
		// instead, it reduces into a number of partial sums less than or equal to work_group_size
		// the essential difference to OMP here is the check: in OMP, essentially `partial_n` is tested to be > 1
		// here, `n_chunks` is tested to be > 1
		// meaning that there are `partial_n` residual partial sums in the last chunk (which may not be wholly populated)
		while (n_chunks > 1) {
			sum_reduce.setArg(1, partial_n); // number of partial sums
			if (step_parity == 1) {
				sum_reduce.setArg(0, tmp_buffer);
				sum_reduce.setArg(2, partial_sums);
			}
			else {
				sum_reduce.setArg(0, partial_sums);
				sum_reduce.setArg(2, tmp_buffer);
			}
			size_t global_size = get_global_size(partial_n, work_group_size);
			q.enqueueNDRangeKernel(sum_reduce, cl::NullRange, cl::NDRange(global_size), cl::NDRange(work_group_size));
			partial_n = n_chunks;
			n_chunks = get_n_chunks(partial_n, work_group_size);
			step_parity ^= 1;
		}

		// make sure the final result is in the partial_sums buffer
		if (step_parity == 1) {
			q.enqueueCopyBuffer(tmp_buffer, partial_sums, 0, 0, sizeof(float) * partial_n);
		}
		return partial_n;

	}
#undef get_n_chunks
#undef get_global_size
#undef is_pow2

	// return median time of reps runs
	float OCLCalc::calc_time(std::vector<float>& data_vector, size_t n, unsigned reps, float* cv, float* mad) {
		std::vector<float> times;
		for (unsigned i = 0; i < reps; i++) {
			std::vector<float> _data;
			std::copy_n(data_vector.begin(), n, std::back_inserter(_data));

			// gotta do some mangling first: pad the data to a power of 2 or sort in case of subclass
			auto start = std::chrono::high_resolution_clock::now();
			prepare_data(_data); // count that in
			calc(_data, n, cv, mad);
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> diff = end - start;
			times.push_back(diff.count());
		}
		std::sort(times.begin(), times.end());
		return times.size() % 2 == 0 ? (times[times.size() / 2] + times[times.size() / 2 - 1]) / 2 : times[times.size() / 2];
	}

	void OCLCalcCpuSort::prepare_data(std::vector<float>& data) {
		// sort the data
		std::sort(data.begin(), data.end());
	}

	void OCLCalcCpuSort::sort(cl::CommandQueue&, size_t, size_t) {
		// do nothing -- data is already sorted
	}

	} // namespace calcs