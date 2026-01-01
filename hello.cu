#include <cuda_runtime_api.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#define N 500
__global__ void gpuIncrementVecByConst(float *vec, float c) {
        //printf("GPU,Block %d, thread: %d\n", blockIdx.x, threadIdx.x);
        if (threadIdx.x < N)
                vec[threadIdx.x] += c;
}
__global__ void gpuSqrtVec(float *vec){
        if (threadIdx.x < N)
                vec[threadIdx.x] = sqrt(vec[threadIdx.x]);
}
__host__ void initVec(float *vec){
        srand(time(NULL));
        for (int i=0; i<N; i++){
                vec[i] = rand() % 100;

        }
}
__host__ void printVec(float *vec){
        for (int i=0; i<N ; i++){
                printf("vec[%d]=%f\n" , i, vec[i]);
        }
}
int main() {
        float *vec;
        cudaMallocManaged(&vec, N*sizeof(float));
        initVec(vec);
        printf("Input Vector:\n");
        printVec(vec);
        gpuIncrementVecByConst<<<1, N>>>(vec, 6);
        gpuSqrtVec<<<1, N>>>(vec);
        printf("output vector: \n");
        cudaDeviceSynchronize();
        printVec(vec);
        cudaFree(vec); }

