__kernel void sum_reduce(__global float* input, __global float* partial_sums, __local float* local_sums, int n) {
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
__kernel void sort(__global float* input_ptr, const uint stage, const uint substage )      
{    
    uint gid = get_global_id(0);   
    uint elem_dist = 1 << (stage - substage);   
    uint tmp;

    uint left = (gid & (elem_dist -1)) + (gid >> (stage - substage) ) * 2 * elem_dist;
    uint right = left + elem_dist;
     
    float left_val, right_val;
    float max, min;
    left_val  = input_ptr[left];
    right_val = input_ptr[right];
     
    uint dir = (gid >> stage) & 0x1;
     
    tmp   = dir ? right : tmp;
    right = dir ? left  : right;
    left  = dir ? tmp   : left;
     
    max = (left_val < right_val) ? right_val : left_val;
    min = (left_val < right_val) ? left_val  : right_val;
     
    input_ptr[left]  = min; 
    input_ptr[right] = max;
};

// single work item kernel to calculate the median and mean of an array
// in order to avoid copy of partial_sums to host, we can use a single work item kernel
__kernel void get_med_mean(__global float* input, __global float* partial_sums, __global float* med_mean_buffer, int partial_n, int n) {
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

__kernel void varmed(__global float* input, __global float* partial_sums, __local float* local_sums, __global float* med_mean_buffer, int n) {
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

// final single-threaded work to be done
// is not done on the host to avoid copying the array back to the host
// steps: 
//    1. merge the halves of the input after varmed
//    2. calculate MAD (median of merged array)
//    3. calculate CV
__kernel void final(__global float* input, int n, int mid, __local float* tmp, 
                    __global float* partial_sums, int partial_n, __global float* med_mean_buffer) {
    if(get_global_id(0)  == 0){
        // load first half of the array into local memory to make space
        // tmp must be in reverse order -- because of varmed kernel
        for(int i = 0; i < mid; i++){
            tmp[mid - i - 1] = input[i];
        }

        // merge the two halves
        int i = 0, j = mid, k = 0;
        while(i < mid && j < n){
            if(tmp[i] < input[j]){
                input[k++] = tmp[i++];
            } else {
                input[k++] = input[j++];
            }
        }
        while(i < mid){
            input[k++] = tmp[i++];
        }
        while(j < n){
            input[k++] = input[j++];
        }

        // calculate median of the merged array (= MAD)
        float mad = n % 2 == 0 ? (input[n/2] + input[n/2 - 1]) / 2 : input[n/2];

        // calculate CV
        float sum = 0;
        for(int i = 0; i < partial_n; i++){
            sum += partial_sums[i];
        }
        float cv = sqrt(sum / n) / mad;

        // store output
        med_mean_buffer[0] = mad;
        med_mean_buffer[1] = cv;

    }
}