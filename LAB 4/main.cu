#include <time.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <random>
#define BS 256
#define M 10000
#define N 10000
using namespace std;

//j fila
//i columna
void cudaCheckError(int i) {
    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess) {
        printf("%d.- Cuda failure %s:%d: '%s'\n", i,__FILE__,__LINE__,cudaGetErrorString(e));
        exit(0);
    }
}

double * createCopy(double * arr){
    double * newArr = new double[N*M];
    for(int i = 0; i < N*M; i++){
        newArr[i] = arr[i];
    }
    return newArr;
}

// Kernel de la pregunta 1
void preguntaUno(double *arr, float dx){
    double first, prev, next;
    for(int t = 1; t <= 10; t++){ //itera el t 
        for(int i = 0; i < N*M; i++){ //recorre matriz
            if(i%N == 0 ){
                first = arr[i];
                //es el primero de la fila = primera columna
                prev = (arr[i+1] - arr[i+N-1])/(2*dx);
            }
            else if (i%N == N-1){
                //es el ultimo de la fila = ultima columna
                arr[i] = (first - arr[i-1])/(2*dx);
                arr[i-1] = prev;
            }
            else{
                next = (arr[i+1] - arr[i-1])/(2*dx);
                arr[i-1] = prev;
                prev = next;
            }
        }
    }
}

// kernel de la pregunta 2
__global__ void preguntaDos(double *arr, float dx){
    double first, prev, next;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < M){
        for(int t = 1; t <= 10; t++){ //itera el t 
            for(int i = 0; i < N; i++){ //recorre matriz
                if(i == 0){
                    first = arr[i+tid*M];
                    //es el primero de la fila = primera columna
                    prev = (arr[i+1+tid*M] - arr[i+N-1+tid*M])/(2*dx);
                }
                else if (i == N-1){
                    //es el ultimo de la fila = ultima columna
                    arr[i+tid*M] = (first - arr[i-1+tid*M])/(2*dx);
                    arr[i-1+tid*M] = prev;
                }
                else{
                    next = (arr[i+1+tid*M] - arr[i-1+tid*M])/(2*dx);
                    arr[i-1+tid*M] = prev;
                    prev = next;
                }
            }
        }
	}
}

// kernel de la pregunta 3
__global__ void preguntaTres(double *arr, float dx){
    double first, prev, next;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < M/4){
        for(int t = 1; t <= 10; t++){ //itera el t 
            for(int i = 0; i < N; i++){ //recorre matriz
                if(i == 0){
                    first = arr[i+tid*M];
                    //es el primero de la fila = primera columna
                    prev = (arr[i+1+tid*M] - arr[i+N-1+tid*M])/(2*dx);
                }
                else if (i == N-1){
                    //es el ultimo de la fila = ultima columna
                    arr[i+tid*M] = (first - arr[i-1+tid*M])/(2*dx);
                    arr[i-1+tid*M] = prev;
                }
                else{
                    next = (arr[i+1+tid*M] - arr[i-1+tid*M])/(2*dx);
                    arr[i-1+tid*M] = prev;
                    prev = next;
                }
            }
        }
	}
}

// kernel de la pregunta 4
__device__ double lectura[M*N];
__device__ double escritura[M*N];
__global__ void preguntaCuatro(float dx, int col){ //col = 0 | 2500 | 5000 | 7500
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < M){
        for(int i = col; i < col+N/4; i++){ //recorre matriz
            if(i == 0){
                escritura[i+tid*M] =  (lectura[i+1+tid*M] - lectura[N-1+tid*M])/(2*dx);
            }
            else if (i == N-1){
                escritura[i+tid*M] = (lectura[tid*M] - lectura[i-1+tid*M])/(2*dx);
            }
            else{
                escritura[i+tid*M] = (lectura[i+1+tid*M] - lectura[i-1+tid*M])/(2*dx);
            }
        }
	}
}

// Kernel que copia los valores de la matriz "escritura" a la matriz "lectura"
__global__ void copiar(){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < N*M){
        lectura[tid] = escritura[tid];
    }
}


// Llamadas al kernel de la pregunta 1
void callP1(double *arr){
    float dx = 0.001;
    double *arrP1 = createCopy(arr);
    clock_t t1, t2;
    double ms;
    t1 = clock();
    preguntaUno(arrP1, dx);
    t2 = clock();
    ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
    cout << "1. Tiempo empleado es: "<< ms << "[ms]" << endl;
}

// Llamadas al kernel de la pregunta 2
void callP2(double *arr){
    float dx = 0.001;
    double *arrP2 = createCopy(arr);
    cudaEvent_t ct1, ct2;
    double *arrCUDA;
    float dt;
    int gs = (int)ceil((float)M / BS);
    cudaMalloc((void **)&arrCUDA, N * M * sizeof(double));
    cudaMemcpy(arrCUDA, arrP2, N * M * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    preguntaDos<<<gs, BS>>>(arrCUDA, dx);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
	printf("2. Tiempo GPU: %f[ms]\n", dt);
    cudaMemcpy(arrP2, arrCUDA, N * M * sizeof(double), cudaMemcpyDeviceToHost);
}

// Llamadas al kernel de la pregunta 3
void callP3(double *arr){
    double *arrP3, *arrCUDA1, *arrCUDA2, *arrCUDA3, *arrCUDA4;
    cudaMallocHost(&arrP3, M*N*sizeof(double));

    for(int i = 0; i < N*M; i++){
        arrP3[i] = arr[i];
    }
    
    cudaStream_t stream1, stream2, stream3, stream4;
    int gs = (int)ceil((float)(M/4) / BS);
    cudaEvent_t ct1, ct2;
    float dx = 0.001;

    float dt;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    
    cudaMalloc((void **)&arrCUDA1, (M/4)*N * sizeof(double));
    cudaMalloc((void **)&arrCUDA2, (M/4)*N * sizeof(double));
    cudaMalloc((void **)&arrCUDA3, (M/4)*N * sizeof(double));
    cudaMalloc((void **)&arrCUDA4, (M/4)*N * sizeof(double));

    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);
    //stream 1
    cudaMemcpyAsync(&arrCUDA1[0], &arrP3[0],(M/4)*N*sizeof(double), cudaMemcpyDeviceToHost, stream1);
    preguntaTres<<<gs, BS, 0, stream1>>>(arrCUDA1, dx);
    cudaMemcpyAsync(&arrP3[0], &arrCUDA1[0], (M/4)*N*sizeof(double), cudaMemcpyHostToDevice, stream1);

    //stream 2
    cudaMemcpyAsync(arrCUDA2, &arrP3[M/4], (M/4)*N*sizeof(double), cudaMemcpyDeviceToHost, stream2);
    preguntaTres<<<gs, BS, 0, stream2>>>(arrCUDA2, dx);
    cudaMemcpyAsync(&arrP3[M*N/4], arrCUDA2, (M/4)*N*sizeof(double), cudaMemcpyHostToDevice, stream2);
    
    //stream 3
    cudaMemcpyAsync(arrCUDA3, &arrP3[M/2], (M/4)*N*sizeof(double), cudaMemcpyDeviceToHost, stream3);
    preguntaTres<<<gs, BS, 0, stream3>>>(arrCUDA3, dx);
    cudaMemcpyAsync(&arrP3[M*N/2], arrCUDA3, (M/4)*N*sizeof(double), cudaMemcpyHostToDevice, stream3);

    //stream 4
    cudaMemcpyAsync(arrCUDA4, &arrP3[3*M/4], (M/4)*N*sizeof(double), cudaMemcpyDeviceToHost, stream4);
    preguntaTres<<<gs, BS, 0, stream4>>>(arrCUDA4, dx);
    cudaMemcpyAsync(&arrP3[3*M*N/4], arrCUDA4, (M/4)*N*sizeof(double), cudaMemcpyHostToDevice, stream4);

    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
	printf("3. Tiempo GPU: %f[ms]\n", dt);
}

// Llamadas al kernel de la pregunta 4
void callP4(double *arr){

    cudaStream_t stream1, stream2, stream3, stream4;
    int gs = (int)ceil((float)(M) / BS);
    int grid_size = (int)ceil((float)(M*N) / BS);
    cudaEvent_t ct1, ct2;
    float dx = 0.001;
    float dt;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    // Copiar la matriz a memoria global.
    cudaMemcpyToSymbol(lectura, arr, (M*N)*sizeof(double));

    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);

    for(int t = 0; t < 10; t++){
        //stream 1
        preguntaCuatro<<<gs, BS, 0, stream1>>>(dx, 0);    

        //stream 2
        preguntaCuatro<<<gs, BS, 0, stream2>>>(dx, 2500);
        
        //stream 3
        preguntaCuatro<<<gs, BS, 0, stream3>>>(dx, 5000);

        //stream 4
        preguntaCuatro<<<gs, BS, 0, stream4>>>(dx, 7500);
        
        cudaDeviceSynchronize(); //espera al resto de streams
        // kernel que copia la matriz escritura a la lectura.
        copiar<<<grid_size, BS>>>(); //copiando con N*M hebras
        cudaDeviceSynchronize(); //espera hasta que se termine de ejecutar el kernel anterior
    }

    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
	printf("4. Tiempo GPU: %f[ms]\n", dt);

    //copiar la matriz lectura de gpu a cpu
    double *matriz = new double[M*N];
    cudaMemcpyFromSymbol(matriz, lectura, (M*N)*sizeof(double));

}

int main(){
    //inicializacion del arreglo con valores aleatorios entre 0 y 1
    random_device rd;
    mt19937 gen(rd()); 
    uniform_real_distribution<> dis(0.0, 1.0);
    
    double *arr = new double[N*M];
    for(int i = 0; i < N*M; i++){
        arr[i] = dis(gen);
    }

    //pregunta uno
    callP1(arr);

    //pregunta dos
    callP2(arr);

    //pregunta tres
    callP3(arr);

    //pregunta 4
    callP4(arr);

}