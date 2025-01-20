/*
Memory Reqirements
------------------
1. The input array is stored in global memory (size n)
2. The partial sums are stored in global memory (size same as number of work groups)
3. The local sums are stored in local memory (size same as number of work items in a work group)
4. The temporary array used in the final kernel is stored in local memory (size n/2)
5. The med_mean_buffer is stored in global memory (size 2)
 */


__kernel void sum_reduce(__global float* input, __global float* partial_sums, __local float* local_sums) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    // Load data into local memory
    local_sums[local_id] = input[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    for (int stride = 1; stride < local_size; stride *= 2) {
        int index = 2 * stride * local_id;
        if (index < local_size) {
            local_sums[index] += local_sums[index + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result for this block to global memory
    if (local_id == 0) {
        partial_sums[group_id] = local_sums[0];
    }
}
     
//The bitonic sort kernel, needs driver code to call it multiple times with different stage and substage values
__kernel void sort(__global float* input, const uint stage, const uint substage )      
{    
    uint global_id = get_global_id(0);   
    uint elem_dist = 1 << (stage - substage);   
    uint tmp;

    uint left = (global_id & (elem_dist -1)) + (global_id >> (stage - substage) ) * 2 * elem_dist;
    uint right = left + elem_dist;
     
    float left_val, right_val;
    float max, min;
    left_val  = input[left];
    right_val = input[right];
     
    uint dir = (global_id >> stage) & 0x1;
     
    tmp   =       right ;        // only used when dir == 1
    right = dir ? left  : right;
    left  = dir ? tmp   : left;
     
    max = (left_val < right_val) ? right_val : left_val;
    min = (left_val < right_val) ? left_val  : right_val;
     
    input[left]  = min; 
    input[right] = max;
};

// single work item kernel to calculate the median and mean of an array
// in order to avoid copy of partial_sums to host, we can use a single work item kernel
__kernel void get_med_mean(__global float* input, __global float* partial_sums, __global float* med_mean_buffer, size_t partial_n, size_t n) {
    if(get_global_id(0) == 0) {
        float sum = 0;
        for(int i = 0; i < partial_n; i++) {
            sum += partial_sums[i];
        }
        float mean = sum / n;
        float median = n % 2 == 0 ? (input[n/2] + input[n/2 - 1]) / 2 : input[n/2];
        med_mean_buffer[0] = median;
        med_mean_buffer[1] = mean;
    }
}

inline float diff_sqr(float a, float b) {
    return (a - b) * (a - b);
}

__kernel void varmed(__global float* input, __global float* partial_sums, __local float* local_sums, __global float* med_mean_buffer) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    float mean = med_mean_buffer[1];
    float median = med_mean_buffer[0];

    // Load data into local memory
    local_sums[local_id] = diff_sqr(input[global_id], mean);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform reduction in local memory
    for (int stride = 1; stride < local_size; stride *= 2) {
        int index = 2 * stride * local_id;
        if (index < local_size) {
            local_sums[index] += diff_sqr(local_sums[index + stride], mean);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the result for this block to global memory
    if (local_id == 0) {
        partial_sums[group_id] = local_sums[0];
    }

    // subtract median from input data and absolute value
    input[global_id] = fabs(input[global_id] - median);
}

__kernel void reverse(__global float* input, size_t n) {
    int global_id = get_global_id(0);
    if(global_id >= n / 2) return;

    float tmp = input[global_id];
    input[global_id] = input[n - global_id - 1];
    input[n - global_id - 1] = tmp;
}

__kernel void calc_cv(__global float* partial_sums, size_t partial_n, __global float* med_mean_buffer, size_t n) {
    if(get_global_id(0) == 0) {
        float sum = 0;
        for(int i = 0; i < partial_n; i++) {
            sum += partial_sums[i];
        }
        float mean = med_mean_buffer[1];
        float cv = sqrt(sum / n) / mean;
        med_mean_buffer[0] = cv;
    }
}