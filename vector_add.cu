#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA核函数：GPU并行向量加法
__global__ void vecAdd(int *a, int *b, int *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];
}

// CPU串行向量加法
void cpuVecAdd(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

int main() {
    // 3组不同数据规模，做效率对比
    int size_list[] = {1024, 10240, 102400};
    int test_count = sizeof(size_list) / sizeof(size_list[0]);

    for (int t = 0; t < test_count; t++) {
        int n = size_list[t];
        int size = n * sizeof(int);
        printf("===== 数据规模：%d 个元素 =====\n", n);

        // 初始化主机端数据
        int *h_a = (int*)malloc(size);
        int *h_b = (int*)malloc(size);
        int *h_c_cpu = (int*)malloc(size);
        int *h_c_gpu = (int*)malloc(size);

        for (int i = 0; i < n; i++) {
            h_a[i] = i;
            h_b[i] = i * 2;
        }

        // CPU计算&计时
        clock_t cpu_start = clock();
        cpuVecAdd(h_a, h_b, h_c_cpu, n);
        clock_t cpu_end = clock();
        printf("CPU 计算耗时：%.2f ms\n", (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000);

        // GPU端内存申请
        int *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_c, size);

        // 数据从CPU拷贝到GPU
        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

        // GPU计算&计时
        clock_t gpu_start = clock();
        vecAdd<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize(); // 等待GPU计算完成
        clock_t gpu_end = clock();
        printf("GPU 计算耗时：%.2f ms\n", (double)(gpu_end - gpu_start) / CLOCKS_PER_SEC * 1000);

        // 错误检查
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA 运行错误: %s\n", cudaGetErrorString(err));
        }

        // 结果拷贝回CPU
        cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
        printf("计算结果验证：前2个元素结果为 %d, %d\n\n", h_c_gpu[0], h_c_gpu[1]);

        // 释放内存
        free(h_a); free(h_b); free(h_c_cpu); free(h_c_gpu);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }

    return 0;
}
