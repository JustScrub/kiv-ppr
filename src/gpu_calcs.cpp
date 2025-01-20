#include "calcs.hpp"
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <vector>

const char* test_kernel = R"(
__kernel void vector_add(__global const float* A, __global const float* B, __global float* C) {
    int id = get_global_id(0);
    C[id] = A[id] + B[id];
}

__kernel void vector_mul(__global const float* A, __global const float* B, __global float* C) {
    int id = get_global_id(0);
    C[id] = A[id] * B[id];
}

)";

void sorting_driver_code(cl::CommandQueue &q, cl::Kernel &k, size_t n, size_t work_group_size) {
	unsigned int stage, substage, n_stages;
    stage = substage = n_stages = 0;
    for(size_t tmp = n; tmp > 1; tmp /= 2)
        ++n_stages;
	size_t global_size = ((n / 2 + work_group_size - 1) / work_group_size) * work_group_size;
	// assuming input arg already set!

	for (stage = 0; stage < n_stages; ++stage) {
		k.setArg(1, stage);
		for (substage = 0; substage < stage + 1; ++substage) {
			k.setArg(2, substage);
			q.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(global_size), cl::NDRange(work_group_size));
		}
	}

}

namespace calcs {

	GpuCalc::GpuCalc() {
		cl_int err;
		// prep platform, device, context, and queue
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
		this->    reverse = cl::Kernel(program, "reverse");
		this->    cv_calc = cl::Kernel(program, "calc_cv");
	}

	void GpuCalc::calc(float* const data, size_t n, float* cv, float* mad) {
		cl_int err;
		// copy data -- need to be padded to (at least) a power of 2
		size_t n_pow_padded = 1;
		while (n_pow_padded < n) n_pow_padded *= 2;
		std::vector<float> padded_data(data, data+n);

		// determine number of work groups -- for partial sums
		size_t work_group_size, tmp, work_group_n, global_size;
		sum_reduce.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &work_group_size);
		varmed.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &tmp);
		work_group_size = tmp < work_group_size ? tmp : work_group_size;
		work_group_n = n / work_group_size + (n % work_group_size != 0);
		global_size = work_group_n * work_group_size;

		//actually pad it so work items over the size of n don't segfault AND there are at least power of 2 elements
		padded_data.resize(std::max(n_pow_padded,global_size), std::numeric_limits<float>::max());

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
		cl::Buffer input_buff(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * n_pow_padded, padded_data.data());
		cl::Buffer partial_sums(context, CL_MEM_READ_WRITE, sizeof(float) * work_group_n);
		cl::Buffer med_mean_buffer(context, CL_MEM_READ_WRITE, sizeof(float) * 2);

		// set unchanging kernel arguments
		sum_reduce.setArg(0, input_buff);
		sum_reduce.setArg(1, partial_sums);
		sum_reduce.setArg(2, cl::Local(sizeof(float) * work_group_size));

		sort_ker.setArg(0, input_buff);
		// the rest of the arguments are set in the sorting_driver_code function

		med_mean.setArg(0, input_buff);
		med_mean.setArg(1, partial_sums); // partial sums to sum
		med_mean.setArg(2, med_mean_buffer); // output
		med_mean.setArg(3, work_group_n); // length of partial sums
		med_mean.setArg(4, n); // length of input

		varmed.setArg(0, input_buff);
		varmed.setArg(1, partial_sums);
		varmed.setArg(2, cl::Local(sizeof(float) * work_group_size));
		varmed.setArg(3, med_mean_buffer);

		cv_calc.setArg(0, partial_sums);
		cv_calc.setArg(1, work_group_n);
		cv_calc.setArg(2, med_mean_buffer);
		cv_calc.setArg(3, n);

		reverse.setArg(0, input_buff);
		reverse.setArg(1, n / 2); // only reverse the first half

		float* output = new float[2];
		// execute kernels
		cl::CommandQueue queue(context, device);
		// sum + sort -> med + mean -> variation + x=abs(x-median) -> sum partial variations (second med_mean) -> sort transformed -> cv, mad
		err = queue.enqueueNDRangeKernel(sum_reduce, cl::NullRange, cl::NDRange(global_size), cl::NDRange(work_group_size));
		sorting_driver_code(queue, sort_ker, n_pow_padded, work_group_size);
		err = queue.enqueueNDRangeKernel(med_mean, cl::NullRange, cl::NDRange(1), cl::NullRange); // do this work on device rather than on host to avoid moving partial sums
		err = queue.enqueueNDRangeKernel(varmed, cl::NullRange, cl::NDRange(global_size), cl::NDRange(work_group_size));
		err = queue.enqueueNDRangeKernel(reverse, cl::NullRange, cl::NDRange(n / 4), cl::NullRange);
		err = queue.enqueueNDRangeKernel(cv_calc, cl::NullRange, cl::NDRange(1), cl::NullRange); // same as before
		queue.enqueueReadBuffer(med_mean_buffer, CL_TRUE, 0, sizeof(float) * 2, output);
		queue.enqueueReadBuffer(input_buff, CL_TRUE, 0, sizeof(float) * n, data);
		
		//this cannot be done in parallel :(
		//and calculating in a single work-item is inefficient --> move data to host and calculate there
        std::inplace_merge(data, data + n / 2, data + n);

		*cv = output[0]; // calculated by cv_calc
        *mad = n % 2 == 0 ? (data[n / 2] + data[n / 2 - 1]) / 2 : data[n / 2];

		/*
		queue.enqueueNDRangeKernel(final_calc, cl::NullRange, cl::NDRange(1), cl::NullRange); // same as before
		queue.finish();
		std::cout << "final_calc done" << std::endl;

		queue.enqueueReadBuffer(med_mean_buffer, CL_TRUE, 0, sizeof(float) * 2, output);
		*mad = output[0];
		*cv  = output[1];
		*/
	}



	void GpuCalc::test_calc(float* const data, size_t n, float* cv, float* mad) {
		// Initialize data
		const int elements = 1<<20;
		std::vector<float> A(elements, 1.0f);
		std::vector<float> B(elements, 2.0f);
		std::vector<float> C(elements);

		// Get platforms and devices
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		std::cout << "Number of platforms: " << platforms.size() << std::endl;
		cl::Platform platform = platforms.front();
		std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		cl::Device device = devices.front();

		// Create context and command queue
		cl::Context context(device);
		cl::CommandQueue queue(context, device);

		// Create buffers
		cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * elements, A.data());
		cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * elements, B.data());
		cl::Buffer bufferC(context, CL_MEM_READ_WRITE, sizeof(float) * elements);
		cl::Buffer bufferD(context, CL_MEM_WRITE_ONLY, sizeof(float) * elements);

		// Create program and kernel
		cl::Program program(context, test_kernel);
		program.build({device});
		cl::Kernel kernel_vadd(program, "vector_add");
		cl::Kernel kernel_vmul(program, "vector_mul");

		// Set kernel arguments
		kernel_vadd.setArg(0, bufferA);
		kernel_vadd.setArg(1, bufferB);
		kernel_vadd.setArg(2, bufferC);

		kernel_vmul.setArg(0, bufferB);
		kernel_vmul.setArg(1, bufferC);
		kernel_vmul.setArg(2, bufferD);

		// Execute kernels
		cl::NDRange global(elements);
		queue.enqueueNDRangeKernel(kernel_vadd, cl::NullRange, global, cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_vmul, cl::NullRange, global, cl::NullRange);
		queue.finish();

		// Read results
		queue.enqueueReadBuffer(bufferD, CL_TRUE, 0, sizeof(float) * elements, C.data()); // read bufferD into C

		// Print results
		for (int i = 0; i < 10; ++i) {
			std::cout << C[i] << " ";
		}
		std::cout << std::endl;
	}
} // namespace calcs