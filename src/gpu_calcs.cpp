#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include "calcs.hpp"

const char* kernelSource = R"(
__kernel void vector_add(__global const float* A, __global const float* B, __global float* C) {
    int id = get_global_id(0);
    C[id] = A[id] + B[id];
}
)";

namespace calcs {
	void GpuCalc::calc(float* const data, size_t n, float* cv, float* mad) {
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
		cl::Program program(context, kernelSource);
		program.build({device});
		cl::Kernel kernel(program, "vector_add");

		// Set kernel arguments
		kernel.setArg(0, bufferA);
		kernel.setArg(1, bufferB);
		kernel.setArg(2, bufferC);

		// Execute kernel
		cl::NDRange global(elements);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);

		// Set kernel arguments
		kernel.setArg(0, bufferA);
		kernel.setArg(1, bufferC);
		kernel.setArg(2, bufferD);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);

		queue.finish();

		// Read results
		queue.enqueueReadBuffer(bufferD, CL_TRUE, 0, sizeof(float) * elements, C.data());

		// Print results
		for (int i = 0; i < 10; ++i) {
			std::cout << C[i] << " ";
		}
		std::cout << std::endl;
	}
} // namespace calcs