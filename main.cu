#include <time.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define BS 256
/*
 *  Generador de matriz
 */
void genMatrix(int** matrix, int N, int M) {   
    int* matrix1 = new int[M*N];
    srand(1);
	for(int i = 0; i < M*N; i++)
		matrix1[i] = rand() % 1000 +1;
        // matrix1[i] = 1;
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
    if (tid < N){
        extern __shared__ int ax[];
        int size;
        for(int i = 0; i < N; i++){
            ax[threadIdx.x] = A[i*N + tid]*x[tid];
            __syncthreads();
            // for(int j = 1; j < log2(BS); j++){
            //     if(threadIdx.x < (BS/(2**j))){
            //         atomicAdd(ax[threadIdx.x % (BS/(2**j))],ax[threadIdx.x]);
            //     }
            // }
            for (size = 256/2; size>0; size/=2) {
                if (threadIdx.x<size) atomicAdd(&ax[threadIdx.x], ax[threadIdx.x+size]);
                __syncthreads();
            }
            if (threadIdx.x == 0){
                atomicAdd(&b[i], ax[0]);
            }
        }
    }
}

// bs = 2
// 1 2 3 4         10
// 4 2 3 4         x2
// 2 2 3 4         x3
// 6 2 3 4         x4



/*
  *  Kernel inciso E
  */
__global__ void kernelSM(int *A,int *x, int *b, int N){
    //calcular por columna de A y por filas de x
    //guardar el valor en variable local y al final hacer un atomicAdd
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ int vx[];
	if (tid < N){
        int total = 0;
        for(int i = 0; i < N; i++){
            vx[threadIdx.x] = x[i];
            __syncthreads();
            total += A[tid*N + i]*vx[threadIdx.x];
            // __syncthreads();
        }
        atomicAdd(&b[tid], total);
	}
}

/*
  *  Kernel inciso F
  */
__constant__ int x[10000];
__global__ void kernelCM(int *A, int *b, int N){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N){
        int total = 0;
        for(int i = 0; i < N; i++){
            total += A[i+tid*N]*x[i];
        }
        b[tid] = total;
    }
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
	printf("c) Tiempo GPU: %f[ms]\n", dt);

	bhost = new int[M];
	cudaMemcpy(bhost, b, M * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("%d\n", bhost[1]);
    for(int i = 0; i < 10; i++){
       printf("%d ", bhost[i]);    
    }
    printf("\n-----------------------------------------------------------------------\n");
	delete[] bhost;
	cudaFree(b); cudaFree(A); cudaFree(x);

    //inciso D
    gs = (int)ceil((float)N / bs);
    cudaMalloc((void**)&A, N * M * sizeof(int));
    cudaMalloc((void**)&x, N * sizeof(int));
    cudaMemcpy(A, Ahost, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(x, xhost, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&b, M * sizeof(int));
	cudaMemset(b, 0, M);

    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    //printf("gs: %d  bs:%d\n", gs, bs);
    kernelRed<<<gs, bs, bs*sizeof(int)>>>(A, x, b, N);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
	printf("d) Tiempo GPU: %f[ms]\n", dt);

	bhost = new int[M]();
	cudaMemcpy(bhost, b, M * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 10; i++){
        printf("%d ", bhost[i]);    
    }
    printf("\n-----------------------------------------------------------------------\n");
    // printf("%d\n", bhost[1]);
	delete[] bhost;
	cudaFree(b); cudaFree(A); cudaFree(x);


    //inciso E
    gs = (int)ceil((float)N / bs);
    cudaMalloc((void**)&A, N * M * sizeof(int));
    cudaMalloc((void**)&x, N * sizeof(int));
    cudaMemcpy(A, Ahost, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(x, xhost, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&b, M * sizeof(int));
	cudaMemset(b, 0, M);

    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    //printf("gs: %d  bs:%d\n", gs, bs);
    kernelSM<<<gs, bs, bs*sizeof(int)>>>(A, x, b, N);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
	printf("e) Tiempo GPU: %f[ms]\n", dt);

	bhost = new int[M]();
	cudaMemcpy(bhost, b, M * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 10; i++){
        printf("%d ", bhost[i]);    
    }
     printf("\n-----------------------------------------------------------------------\n");
    // printf("%d\n", bhost[1]);
	delete[] bhost;
	cudaFree(b); cudaFree(A); cudaFree(x);

    //inciso F
    gs = (int)ceil((float)N / bs);
    cudaMalloc((void**)&A, N * M * sizeof(int));
    //cudaMalloc((void**)&x, N * sizeof(int));
    cudaMemcpy(A, Ahost, N * M * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(x, xhost, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&b, M * sizeof(int));
	cudaMemset(b, 0, M);
    cudaMemcpyToSymbol(x, xhost, N*sizeof(int), 0, cudaMemcpyHostToDevice);

    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    //printf("gs: %d  bs:%d\n", gs, bs);
    kernelCM<<<gs, bs>>>(A, b, N);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
	printf("f) Tiempo GPU: %f[ms]\n", dt);

	bhost = new int[M]();
	cudaMemcpy(bhost, b, M * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 10; i++){
        printf("%d ", bhost[i]);    
    }
    // printf("\n-----------------------------------------------------------------------\n");
    // printf("%d\n", bhost[1]);
	delete[] bhost;
	cudaFree(b); cudaFree(A); cudaFree(x);

    return 0;
}