#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <string>
#include <sys/types.h>
using namespace std;

#define BS 256
#define X 3
#define Y 3
#define PYTHON_COMMAND 0 // 0 si el comando de python NO lleva el 3. cualquier otro valor si lo lleva.
extern int EME;
extern int ENE;

//
// Funcion que ejecuta el comando de python para convertir la imagen en formato txt
// en formato txt a PNG.
// Hay que cambiar PYTHON_COMMAND dependiendo de como se ejecuta el comando en el PC.
//
int TXTtoRGB(){
    if (PYTHON_COMMAND == 0){
        system("python TXTtoRGB.py");
        return 1;
    }
    else if(system("python3 TXTtoRGB.py")){
        return 1;
    }
    else{
        cout << "Error al convertir el txt a imagen" << endl;
        return 0;
    }
}

//
// Funcion que imprime el ultimo error arrojado por CUDA
//
void cudaCheckError(int i) {
    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess) {
        printf("%d.- Cuda failure %s:%d: '%s'\n", i,__FILE__,__LINE__,cudaGetErrorString(e));
        exit(0);
    }
}

/*
 *  Escritura Archivo txt
    Funcion extraida de actividad de curso
 */
void Write(float* R, float* G, float* B, 
	       int M, int N, const char *filename) {
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", M, N);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", R[i]);
    fprintf(fp, "%f\n", R[M*N-1]);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", G[i]);
    fprintf(fp, "%f\n", G[M*N-1]);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", B[i]);
    fprintf(fp, "%f\n", B[M*N-1]);
    fclose(fp);
}

/*
 *  Lectura Archivo txt
 */
void Read(float** R, float** G, float** B, int *M, int *N, 
	      const char *filename) {    
    FILE *fp;
    fp = fopen(filename, "r");
    fscanf(fp, "%d %d\n", M, N);

    int imsize = (*M) * (*N);
    float* R1 = new float[imsize];
    float* G1 = new float[imsize];
    float* B1 = new float[imsize];

    for(int i = 0; i < imsize; i++)
        fscanf(fp, "%f ", &(R1[i]));
    for(int i = 0; i < imsize; i++)
        fscanf(fp, "%f ", &(G1[i]));
    for(int i = 0; i < imsize; i++)
        fscanf(fp, "%f ", &(B1[i]));
    
    fclose(fp);
    *R = R1; *G = G1; *B = B1;
}

// __global__ void kernelStream(float* R, float* Rx, float *Ry, int M, int N, int Mout, int Nout, int *k1, int *k2){
// 	int tid = threadIdx.x + blockDim.x * blockIdx.x;
// 	if (tid < (Mout*Nout)/4){
//         float v1 = 0, v2 = 0;
//         int fila = tid + (tid/Nout)*2;
//         for(int i = 0; i<Y ; i++){
//             for(int j = 0; j<X ; j++){
//                 v1 += R[j+i*N+fila]*k1[j+i*Y];
//                 v2 += R[j+i*N+fila]*k2[j+i*Y];
//             }
//         }
//         Rx[tid] = v1;
//         Ry[tid] = v2;
//     }
// }


__global__ void kernelConvolucion(float* R, float* Rx, float *Ry, int M, int N, int Mout, int Nout, int *k1, int *k2){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < Mout*Nout){
        float v1 = 0, v2 = 0;
        int fila = tid + (tid/Nout)*2;
        for(int i = 0; i<Y ; i++){
            for(int j = 0; j<X ; j++){
                v1 += R[j+i*N+fila]*k1[j+i*Y];
                v2 += R[j+i*N+fila]*k2[j+i*Y];
            }
        }
        Rx[tid] = v1;
        Ry[tid] = v2;
    }
}


__global__ void kernelFila(float* R, float* Rx, float *Ry, int M, int N, int Mout, int Nout, int *k1, int *k2){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < Mout){
        float v1, v2;
        for(int k=0; k<Nout ; k++){
            v1 = 0;
            v2 = 0;
            for(int i = 0; i<Y ; i++){
                for(int j = 0; j<X ; j++){
                    v1 += R[j+k+i*N+tid*N]*k1[j+i*Y];
                    v2 += R[j+k+i*N+tid*N]*k2[j+i*Y];
                }
            }
            Rx[k+tid*Nout] = v1;
            Ry[k+tid*Nout] = v2;
        }
    }
}

void kernelCPU(float* R, float* Rx, float *Ry, int M, int N, int Mout, int Nout, int *k1, int *k2){
    float v1, v2;
    for(int h=0 ; h<Mout ; h++){
        for(int k=0; k<Nout ; k++){
            v1 = 0;
            v2 = 0;
            for(int i = 0; i<Y ; i++){
                for(int j = 0; j<X ; j++){
                    v1 += R[j+k+i*N+h*N]*k1[j+i*Y];
                    v2 += R[j+k+i*N+h*N]*k2[j+i*Y];
                }
            }
            Rx[k+h*Nout] = v1;
            Ry[k+h*Nout] = v2;
        }
    }
}

void blancoynegro(float* R, float* G, float* B, int M, int N){
    float prom = 0;
    int imsize = (M) * (N);
    for(int i = 0; i < imsize; i++){
        prom = (R[i]+G[i]+B[i] )/ 3;
        (R[i]) = prom;
        (G[i]) = prom;
        (B[i]) = prom;
    }
}

void copiar(float *Rhost, float *Ghost, float *Bhost, float *Rxhost, float *Ryhost, int Mout, int Nout){
    float tmp;
    //norma
    for(int i = 0; i < Mout*Nout; i++){
        tmp = sqrt( (pow(Rxhost[i], 2)+ pow(Ryhost[i], 2)) );
        if(tmp > 1) tmp = 1;
        Rhost[i] = tmp;
        Ghost[i] = tmp;
        Bhost[i] = tmp;
    }

    Write(Rhost, Ghost, Bhost, Mout, Nout, "salida.txt");
}

void callKernelFila(float * Rhost, int N, int M, int Mout, int Nout, int * k1, int * k2){
    float *Rdev, *Rx, *Ry, *Rxhost, *Ryhost, dt;
    cudaEvent_t ct1, ct2;
    int *k1dev, *k2dev;
    
    int GS = (int)ceil((float) Mout / BS);

    cudaMalloc((void**)&Rdev, M * N * sizeof(float));
    //cudaMemcpy(Rdev, Rhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Rdev, Rhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(1);

    cudaMalloc((void**)&k1dev, X * Y * sizeof(int));
    cudaMalloc((void**)&k2dev, X * Y * sizeof(int));
    cudaMemcpy(k1dev, k1, X * Y * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(k2dev, k2, X * Y * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError(2);

    cudaMalloc((void**)&Rx, Mout * Nout * sizeof(float));
    cudaMalloc((void**)&Ry, Mout * Nout * sizeof(float));
    cudaCheckError(3);

    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);

    kernelFila<<<GS, BS>>>(Rdev, Rx, Ry, M, N, Mout, Nout, k1dev, k2dev);
    cudaCheckError(4);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    cout << "Tiempo GPU: " << dt << "[ms]" << endl;

    Rxhost = new float[Mout*Nout];
	Ryhost = new float[Mout*Nout];
    cudaMemcpy(Rxhost, Rx, Mout * Nout * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(Ryhost, Ry, Mout * Nout * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError(5);

    float *Rfinal= new float[Mout*Nout];
    float *Gfinal = new float[Mout*Nout];
    float *Bfinal = new float[Mout*Nout];
    copiar(Rfinal, Gfinal, Bfinal, Rxhost, Ryhost, Mout, Nout);

}

void callCPU(float * Rhost, int N, int M, int Mout, int Nout, int * k1, int * k2){
    clock_t t1, t2;
    double ms;
    t1 = clock();
    float *Rx = new float[Mout*Nout];
    float *Ry = new float[Mout*Nout];
    kernelCPU(Rhost, Rx, Ry, M, N, Mout, Nout, k1, k2);
    t2 = clock();
    ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
    cout << "Tiempo CPU es: "<< ms << "[ms]" << endl;

    float *Rfinal= new float[Mout*Nout];
    float *Gfinal = new float[Mout*Nout];
    float *Bfinal = new float[Mout*Nout];
    copiar(Rfinal, Gfinal, Bfinal, Rx, Ry, Mout, Nout);
}

void callKernelConv(float * Rhost, int N, int M, int Mout, int Nout, int * k1, int * k2){
    float *Rdev, *Rx, *Ry, *Rxhost, *Ryhost, dt;
    cudaEvent_t ct1, ct2;
    int *k1dev, *k2dev;
    
    int GS = (int)ceil((float) Mout*Nout / BS);

    cudaMalloc((void**)&Rdev, M * N * sizeof(float));
    //cudaMemcpy(Rdev, Rhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Rdev, Rhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(1);

    cudaMalloc((void**)&k1dev, X * Y * sizeof(int));
    cudaMalloc((void**)&k2dev, X * Y * sizeof(int));
    cudaMemcpy(k1dev, k1, X * Y * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(k2dev, k2, X * Y * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError(2);

    cudaMalloc((void**)&Rx, Mout * Nout * sizeof(float));
    cudaMalloc((void**)&Ry, Mout * Nout * sizeof(float));
    cudaCheckError(3);

    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);

    kernelConvolucion<<<GS, BS>>>(Rdev, Rx, Ry, M, N, Mout, Nout, k1dev, k2dev);
    cudaCheckError(4);
    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    cout << "Tiempo GPU: " << dt << "[ms]" << endl;

    Rxhost = new float[Mout*Nout];
	Ryhost = new float[Mout*Nout];
    cudaMemcpy(Rxhost, Rx, Mout * Nout * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(Ryhost, Ry, Mout * Nout * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError(5);

    float *Rfinal= new float[Mout*Nout];
    float *Gfinal = new float[Mout*Nout];
    float *Bfinal = new float[Mout*Nout];
    copiar(Rfinal, Gfinal, Bfinal, Rxhost, Ryhost, Mout, Nout);
}

// void callKernelStream(float * Rhost, int N, int M, int Mout, int Nout, int * k1, int * k2){
//     cudaStream_t stream1, stream2, stream3, stream4;
//     float *Rdev, *Rx, *Ry, *Rxhost, *Ryhost, dt;
//     cudaEvent_t ct1, ct2;
//     int *k1dev, *k2dev;
    
//     int cantStream = 4;
//     int GS = (int)ceil((float) (Mout/cantStream)*Nout / BS); // S1 - S2 - S3 - S4+7%4
//     int GS4 = (int)ceil((float) ((Mout+Mout%cantStream)/cantStream)*Nout / BS);

//     cudaStreamCreate(&stream1);
//     cudaStreamCreate(&stream2);
//     cudaStreamCreate(&stream3);
//     cudaStreamCreate(&stream4);

//     cudaMalloc((void**)&Rdev, M * N * sizeof(float));
//     cudaMemcpy(Rdev, Rhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
//     cudaCheckError(1);

//     cudaMalloc((void**)&k1dev, X * Y * sizeof(int));
//     cudaMalloc((void**)&k2dev, X * Y * sizeof(int));
//     cudaMemcpy(k1dev, k1, X * Y * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(k2dev, k2, X * Y * sizeof(int), cudaMemcpyHostToDevice);
//     cudaCheckError(2);

//     cudaMalloc((void**)&Rx, Mout * Nout * sizeof(float));
//     cudaMalloc((void**)&Ry, Mout * Nout * sizeof(float));
//     cudaCheckError(3);

//     cudaEventCreate(&ct1);
//     cudaEventCreate(&ct2);
//     cudaEventRecord(ct1);

//     kernelConvolucion<<<GS, BS>>>(Rdev, Rx, Ry, M, N, Mout, Nout, k1dev, k2dev);
//     cudaCheckError(4);
//     cudaEventRecord(ct2);
//     cudaEventSynchronize(ct2);
//     cudaEventElapsedTime(&dt, ct1, ct2);
//     cout << "Tiempo GPU: " << dt << "[ms]" << endl;

//     Rxhost = new float[Mout*Nout];
// 	Ryhost = new float[Mout*Nout];
//     cudaMemcpy(Rxhost, Rx, Mout * Nout * sizeof(float), cudaMemcpyDeviceToHost);
// 	cudaMemcpy(Ryhost, Ry, Mout * Nout * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaCheckError(5);

//     float *Rfinal= new float[Mout*Nout];
//     float *Gfinal = new float[Mout*Nout];
//     float *Bfinal = new float[Mout*Nout];
//     copiar(Rfinal, Gfinal, Bfinal, Rxhost, Ryhost, Mout, Nout);
// }

int main(){

    //inicializacion
    //se convierte la imagen a blanco y negro
    float *Rhost, *Ghost, *Bhost;
    int M, N, Mout, Nout; //M filas, N columnas
    Read(&Rhost, &Ghost, &Bhost, &M, &N, "imgG.txt"); 
    blancoynegro(Rhost, Ghost, Bhost, M, N);
    Nout = N - 2;
	Mout = M - 2;
    //probar esto
    EME = M;
    ENE = N;
    
    int *k1{ new int[9]{ -1, 0, 1, -2, 0, 2, -1, 0, 1 } };
    int *k2{ new int[9]{ -1, -2, -1, 0, 0, 0, 1, 2, 1 } };

    // llamada a la implementacion de cpu
    // callCPU(Rhost, N, M, Mout, Nout, k1, k2);
    // TXTtoRGB();

    // llamada a la implementación del kernel usando una hebra por fila. 
    callKernelFila(Rhost, N, M, Mout, Nout, k1, k2);
    TXTtoRGB();

    // llamada a la implementación del kernel usando una hebra por fila. 
    // callKernelConv(Rhost, N, M, Mout, Nout, k1, k2);
    // TXTtoRGB();

    return 0;
}