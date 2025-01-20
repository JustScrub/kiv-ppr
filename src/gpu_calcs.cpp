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

void sorting_driver_code(cl::CommandQueue &q, cl::Kernel &k, size_t n, size_t work_group_size, size_t work_group_n) {
	unsigned int stage, passOfStage, numStages, temp;
    stage = passOfStage = numStages = 0;
    for(size_t temp = n; temp > 1; temp /= 2)
        ++numStages;
	size_t global_size = n/2;
	// assuming input arg already set!

	for (stage = 0; stage < numStages; ++stage) {
		k.setArg(1, stage);
		for (passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
			k.setArg(2, passOfStage);
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
		this->     varmed = cl::Kernel(program, "varmed");
		this->   med_mean = cl::Kernel(program, "get_med_mean");
		this-> final_calc = cl::Kernel(program, "final");
	}

	void GpuCalc::calc(float* const data, size_t n, float* cv, float* mad) {

		// determine number of work groups -- for partial sums
		size_t work_group_size, tmp, work_group_n;
		sum_reduce.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &work_group_size);
		varmed.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &tmp);
		work_group_size = tmp < work_group_size ? tmp : work_group_size;
		work_group_n = n / work_group_size + (n % work_group_size != 0);

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
		cl::Buffer input_buff(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, data);
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

		final_calc.setArg(0, input_buff);
		final_calc.setArg(1, n);
		final_calc.setArg(2, cl::Local(sizeof(float) * n/2)); // tmp memory for merging
		final_calc.setArg(3, partial_sums);
		final_calc.setArg(4, work_group_n); // lenght of partial sums
		final_calc.setArg(5, med_mean_buffer); // input for mean, output for CV and MAD

		float* output = new float[2];
		// execute kernels
		cl::CommandQueue queue(context, device);
		// sum + sort -> med + mean -> variation + x=abs(x-median) -> sort + cv + mad
		queue.enqueueNDRangeKernel(sum_reduce, cl::NullRange, cl::NDRange(n), cl::NDRange(work_group_size));
		sorting_driver_code(queue, sort_ker, n, work_group_size, work_group_n);
		queue.enqueueNDRangeKernel(med_mean, cl::NullRange, cl::NDRange(1), cl::NullRange); // do this work on device rather than on host to avoid moving partial sums
		queue.enqueueNDRangeKernel(varmed, cl::NullRange, cl::NDRange(n), cl::NDRange(work_group_size));
		queue.enqueueNDRangeKernel(final_calc, cl::NullRange, cl::NDRange(1), cl::NullRange); // same as before

		queue.enqueueReadBuffer(med_mean_buffer, CL_TRUE, 0, sizeof(float) * 2, output);
		*mad = output[0];
		*cv  = output[1];
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