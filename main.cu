#include <time.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define BS 256
#define T 16
/*
 *  Generador de matriz
 */
void genMatrix(int** matrix, int N, int M) {   
    int* matrix1 = new int[M*N];
    srand(1);
	for(int i = 0; i < M*N; i++)
		//matrix1[i] = rand() % 1000 +1;
        matrix1[i] = 1;
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
        __shared__ int ax[BS];
        for(int i = 0; i < N; i++){
            ax[threadIdx.x] = A[i*N + tid]*x[tid];
            __syncthreads();
            // for(int j = 1; j < log2(BS); j++){
            //     if(threadIdx.x < (BS/(2**j))){
            //         atomicAdd(ax[threadIdx.x % (BS/(2**j))],ax[threadIdx.x]);
            //     }
            // }
            for (int size = BS/2; size>0; size/=2) {
                if (threadIdx.x<size) atomicAdd(&ax[threadIdx.x], ax[threadIdx.x+size]);
                __syncthreads();
            }
            if (threadIdx.x == 0){
                atomicAdd(&b[i], ax[0]);
            }
        }
    }
}


/*
  *  Kernel inciso E
  */
// __global__ void kernelSM(int *A, int *x, int *b, int N){
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     __shared__ int vx[BS];
// 	if (tid < N){
//         vx[threadIdx.x] = x[tid];
//         __syncthreads();
//         int total = 0;
//         for(int i = 0; i < BS; i++){
//             atomicAdd(total, A[i+tid*BS]*vx[]);
//             // atomicAdd(total, A[tid+i*N]*vx[tid]);
//             // total += A[i]*vx[i];
//         }
// 	}
// }


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
    //printf("%d\n", bhost[1]);
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
    kernelRed<<<gs, bs>>>(A, x, b, N);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
	//printf("d) Tiempo GPU: %f[ms]\n", dt);

	bhost = new int[M]();
	cudaMemcpy(bhost, b, M * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 10; i++){
        printf("%d ", bhost[i]);    
    }
    printf("\n-----------------------------------------------------------------------\n");
    // printf("%d\n", bhost[1]);
	delete[] bhost;
	cudaFree(b); cudaFree(A); cudaFree(x);

    return 0;
}