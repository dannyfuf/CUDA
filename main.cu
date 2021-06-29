#include <time.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define BS 256
#define T 16
/*
 *  Generador de matriz
 */
void genMatrix(int** matrix, int N, int M) {   
    int* matrix1 = new int[M*N];
    srand(1);
	for(int i = 0; i < M*N; i++)
		matrix1[i] = rand() % 1000 +1;
    *matrix = matrix1;
}

 /*
  *  Kernel inciso A
  */
__global__ void kernelA(int *A, int *x, int *b, int N){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int xtid = tid/N; // 0 a N, +1 cada N
    int ytid = tid%N; // 0 a N cada N
	if (tid < N*N){
        atomicAdd(&b[xtid], A[tid]*x[ytid]);
	}
}
void comprobar(int *arr){
    for(int i = 0; i < 10000; i++){
        if(arr[i] != 10000){
            printf("fallo: el numero es %d \n",arr[i]);
        }
    }
}
 /*
  *  Kernel inciso B
  */
__global__ void kernelx(int *A, int *x, int *b, int N){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N){
        for(int i = 0; i < N; i++){
            atomicAdd(&b[i], A[tid+i*N]*x[tid]);
        }
	}
}

 /*
  *  Kernel inciso C
  */
__global__ void kernelb(int *A, int *x, int *b, int N){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N){
        int total = 0;
        for(int i = 0; i < N; i++){
            total += A[i+tid*N]*x[i];
        }
        b[tid] = total;
	}
}


/*
1 -> aij * xj

1 2 3
4 5 6
7 8 9

4
5
6

*/
 /*
  *  Kernel inciso D
  */
__global__ void kernelRed(int *A, int *x, int *b, int N){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int row, col;
    __shared__ int ax[BS];
	if (tid < N*N){
        row = tid / N;
        col = tid % N;
        ax[threadIdx.x] = A[tid] * x[col];
        __syncthreads();
        if(threadIdx.x == 0){
            int total = 0;
            for(int i = 0; i < BS; i++){
                total += ax[i];
            }
            __syncthreads();
            atomicAdd(&b[row], total);
            __syncthreads();
        }
	}
}

/*
__global__ void matvec_kernel(const T * __restrict__ dA, const T * __restrict__ dx, T * __restrict__ dy, const unsigned int nRows, const unsigned int nCols){
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ T x_shared[BLOCK_SIZE];

    T y_val = 0.0;

    #pragma unroll
    for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1)/ BLOCK_SIZE); ++m){
        if ((m * BLOCK_SIZE + threadIdx.x) <  nCols) x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
        else x_shared[threadIdx.x] = 0.f;
        __syncthreads();

        #pragma unroll
        for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
            // --- Column-major ordering - faster
            y_val += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
            // --- Row-major ordering - slower
            //y_val += dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
        }

        __syncthreads();
    }

    if (tid < nRows) dy[tid] = y_val;

}
*/

int * suma(int * A, int * X, int N){
    int s;
    int * b = new int[N];
    int k = 0;
    for(int i = 0; i < N*N; i=i+N){
        s = 0; 
        for(int j = i; j < i+N; j++){
            s += A[j] * X[j%N];
        }
        b[k] = s;
        k++;
    }
    return b;
}




int main(){
    clock_t t1, t2;
	double ms;
	cudaEvent_t ct1, ct2;
	float dt;
    int *A, *x, *b;
    int *Ahost, *xhost, *bhost;
	int N = 10000, M = 10000;
    int gs, bs = 256;
    gs = (int)ceil((float)N*N / bs);

    genMatrix(&Ahost, N, M);
	genMatrix(&xhost, 1, N);
    int * a = suma(Ahost, xhost, N);
/*
    for(int i = 0; i < N*M; i++){
        printf("%d ", Ahost[i]);
    }
    printf("\n"); 
    for(int i = 0; i < N; i++){
        printf("%d ", xhost[i]);
    }
    printf("\n"); 
    for(int i = 0; i < N; i++){
        printf("%d ", a[i]);    
    }
*/
    //inciso a

    cudaMalloc((void**)&A, N * M * sizeof(int));
    cudaMalloc((void**)&x, N * sizeof(int));
    cudaMemcpy(A, Ahost, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(x, xhost, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&b, M * sizeof(int));
	    
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    kernelA<<<gs, bs>>>(A, x, b, N);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
	//printf("a) Tiempo GPU: %f[ms]\n", dt);

	bhost = new int[M];

	cudaMemcpy(bhost, b, M * sizeof(int), cudaMemcpyDeviceToHost);
    //printf("%d\n", bhost[1]);
	delete[] bhost;
    
	cudaFree(b); cudaFree(A); cudaFree(x);


    //inciso b

    gs = (int)ceil((float)N / bs);
    cudaMalloc((void**)&A, N * M * sizeof(int));
    cudaMalloc((void**)&x, N * sizeof(int));
    cudaMemcpy(A, Ahost, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(x, xhost, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&b, M * sizeof(int));
	    
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    kernelx<<<gs, bs>>>(A, x, b, N);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
	//printf("b) Tiempo GPU: %f[ms]\n", dt);

	bhost = new int[M];
	cudaMemcpy(bhost, b, M * sizeof(int), cudaMemcpyDeviceToHost);
    //printf("%d\n", bhost[1]);
	delete[] bhost;
	cudaFree(b); cudaFree(A); cudaFree(x);

    //inciso c

    cudaMalloc((void**)&A, N * M * sizeof(int));
    cudaMalloc((void**)&x, N * sizeof(int));
    cudaMemcpy(A, Ahost, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(x, xhost, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&b, M * sizeof(int));
	    
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    kernelb<<<gs, bs>>>(A, x, b, N);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
	//printf("c) Tiempo GPU: %f[ms]\n", dt);

	bhost = new int[M];
	cudaMemcpy(bhost, b, M * sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", bhost[1]);
    //for(int i = 0; i < N*N; i++){
    //    printf("%d ", bhost[i]);    
    //}
    //printf("\n-----------------------------------------------------------------------\n");
	delete[] bhost;
	cudaFree(b); cudaFree(A); cudaFree(x);

    //inciso D
    gs = (int)ceil((float)N*N / bs);
    cudaMalloc((void**)&A, N * M * sizeof(int));
    cudaMalloc((void**)&x, N * sizeof(int));
    cudaMemcpy(A, Ahost, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(x, xhost, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&b, M * sizeof(int));
	cudaMemset(b, 0, M);

    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    printf("gs: %d  bs:%d\n", gs, bs);
    kernelRed<<<gs, bs>>>(A, x, b, N);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
	//printf("d) Tiempo GPU: %f[ms]\n", dt);

	bhost = new int[M]();
	cudaMemcpy(bhost, b, M * sizeof(int), cudaMemcpyDeviceToHost);
    //for(int i = 0; i < N*N; i++){
    ///    printf("%d ", bhost[i]);    
    //}
    //printf("\n-----------------------------------------------------------------------\n");
    printf("%d\n", bhost[1]);
	delete[] bhost;
	cudaFree(b); cudaFree(A); cudaFree(x);

    return 0;
}