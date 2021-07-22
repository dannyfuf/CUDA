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

__global__ void preguntaTres(double *arr, float dx){
        printf("------------------------------------------\n");
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
    printf("------------------------------------------\n");
}

double * createCopy(double * arr){
    double * newArr = new double[N*M];
    for(int i = 0; i < N*M; i++){
        newArr[i] = arr[i];
    }
    return newArr;
}

int main(){
    //inicializacion del arreglo
    random_device rd;
    mt19937 gen(rd()); 
    uniform_real_distribution<> dis(0.0, 1.0);
    float dx = 0.001;
    double *arr = new double[N*M];
    for(int i = 0; i < N*M; i++){
        arr[i] = dis(gen);
    }

    for(int i = 0; i < 20; i++){
        cout << arr[i] << " ";
    }
    cout << endl;

    //pregunta uno
    // double *arrP1 = createCopy(arr);
    // clock_t t1, t2;
    // int t = 10;
    // double ms;
    // t1 = clock();
    // preguntaUno(arrP1, dx);
    // t2 = clock();
    // ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
    // cout << "1. Tiempo empleado es: "<< ms << "[ms]" << endl;
    // for(int i = 0; i < 20; i++){
    //     cout << arrP1[i] << " ";
    // }
    // cout << endl;

    //pregunta dos
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
    // for(int i = 0; i < 20; i++){
    //    cout << arrP2[i] << " ";
    // }
    // cout << endl;

    //pregunta tres
    double *arrP3, *arrCUDA1, *arrCUDA2, *arrCUDA3, *arrCUDA4;
    cudaMallocHost(&arrP3, M*N*sizeof(double));

    for(int i = 0; i < N*M; i++){
        arrP3[i] = arr[i];
    }
    
    cudaStream_t stream1, stream2, stream3, stream4;
    gs = (int)ceil((float)(M/4) / BS);
    
    //cudaStream_t stream[4];
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
    cudaMemcpyAsync(&arrCUDA1, &arrP3,(M/4)*N*sizeof(double), cudaMemcpyDeviceToHost, stream1);
    preguntaTres<<<gs, BS, 0, stream1>>>(arrCUDA1, dx);
    cudaMemcpyAsync(&arrP3[0], &arrCUDA1[0], (M/4)*N*sizeof(double), cudaMemcpyHostToDevice, stream1); 

    //stream 2
    cudaMemcpyAsync(&arrCUDA2, &arrP3[M/4], (M/4)*N*sizeof(double), cudaMemcpyDeviceToHost, stream2);
    preguntaTres<<<gs, BS, 0, stream2>>>(arrCUDA2, dx);
    cudaMemcpyAsync(&arrP3[M*N/4], &arrCUDA2, (M/4)*N*sizeof(double), cudaMemcpyHostToDevice, stream2);

    //stream 3
    cudaMemcpyAsync(&arrCUDA3, &arrP3[M/2], (M/4)*N*sizeof(double), cudaMemcpyDeviceToHost, stream3);
    preguntaTres<<<gs, BS, 0, stream3>>>(arrCUDA3, dx);
    cudaMemcpyAsync(&arrP3[M*N/2], &arrCUDA3, (M/4)*N*sizeof(double), cudaMemcpyHostToDevice, stream3);

    //stream 4
    cudaMemcpyAsync(&arrCUDA4, &arrP3[3*M/4], (M/4)*N*sizeof(double), cudaMemcpyDeviceToHost, stream4);
    preguntaTres<<<gs, BS, 0, stream4>>>(arrCUDA4, dx);
    cudaMemcpyAsync(&arrP3[3*M*N/4], &arrCUDA4, (M/4)*N*sizeof(double), cudaMemcpyHostToDevice, stream4);

    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
	printf("3. Tiempo GPU: %f[ms]\n", dt);

    for(int i = 0; i < 20; i++){
        cout << arrP3[i] << " ";
    }
    cout << endl;

}