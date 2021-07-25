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

__global__ void kernelFila(float* R, float* Rx, float *Ry, int M, int N, int Mout, int Nout, int X, int Y, int *k1, int *k2){
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


//void kernelCPU(float* R, float* Rx, float *Ry, int M, int N, int Mout, int Nout, int X, int Y, int *k1, int *k2){
//    float v1, v2;
//    for(int h=0 ; h<Mout ; h++){
//        for(int k=0; k<Nout ; k++){
//            v1 = 0;
//            v2 = 0;
//            for(int i = 0; i<Y ; i++){
//                for(int j = 0; j<X ; j++){
//                    v1 += R[j+k+i*N+h*N]*k1[j+i*Y];
//                    v2 += R[j+k+i*N+h*N]*k2[j+i*Y];
//                }
//            }
//            Rx[k+h*Nout] = v1;
//            Ry[k+h*Nout] = v2;
//        }
//    }
//}



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

    kernelFila<<<GS, BS>>>(Rdev, Rx, Ry, M, N, Mout, Nout, X, Y, k1dev, k2dev);
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

int main(){

    //inicializacion
    //se convierte la imagen a blanco y negro
    float *Rhost, *Ghost, *Bhost;
    int M, N, Mout, Nout; //M filas, N columnas
    Read(&Rhost, &Ghost, &Bhost, &M, &N, "imgG.txt"); 
    blancoynegro(Rhost, Ghost, Bhost, M, N);
    Nout = N - 2;
	Mout = M - 2;
    
    int *k1{ new int[9]{ -1, 0, 1, -2, 0, 2, -1, 0, 1 } };
    // int *k1{ new int[9]{ 1, 0, -1, 1, 0, -1, 1, 0, -1 } };
    int *k2{ new int[9]{ -1, -2, -1, 0, 0, 0, 1, 2, 1 } };

    //ejemplo
    
    // float *ejemplo{ new float[36]{ 3,0,1,2,7,4,
    //                             1,5,8,9,3,1,
    //                             2,7,2,5,1,3,
    //                             0,1,3,1,7,8,
    //                             4,2,1,6,2,8,
    //                             2,4,5,2,3,9} };



    // int GS = (int)ceil((float) Mout * Nout / BS);

    callKernelFila(Rhost, N, M, Mout, Nout, k1, k2);


    
}

