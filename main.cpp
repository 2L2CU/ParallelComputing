#include <iostream>
#include <cassert>
#include <omp.h>

using namespace std;

#define  MATRIX_SIZE 900


double** create_matrix(int rows, int cols) //dynamic allocation of 2d array 
{
    double** mat = new double* [rows]; 
    for (int i = 0; i < rows; ++i)
    {
        mat[i] = new double[cols]();
    }
    return mat;
}

void destroy_matrix(double** &mat, int rows)  //deletes 2d array, prevents memory leaks 
{
    if (mat)
    {
        for (int i = 0; i < rows; ++i)
        {
            delete[] mat[i]; 
        }

        delete[] mat; 
        mat = nullptr;
    }
}

void randomizeMatrixValues(double** &mat, int rows){ //fill 2d matrix with random numbers 0-9 
    if (mat){
        for(int i = 0; i < rows; i++){
            for (int j = 0; j < rows; j++){
                mat[i][j] = rand()%10;
            }
        }
    }
}

void printMatrix(double** mat, int rows, int cols){ 
     for(int i = 0; i < rows; i++){
      for (int j = 0; j < cols; j++){
        printf(" %f  ",mat[i][j]);
      }
      printf("\n");
    }

}

int main(){
    double t_Start, t_Stop;  // needed to calc duration of task

    int rowsA = MATRIX_SIZE; // size of matrix, simplified by making it square
    int colsA= MATRIX_SIZE; 
    double** matA = create_matrix(rowsA, colsA);
    randomizeMatrixValues(matA, rowsA);

    int rowsB = MATRIX_SIZE; 
    int colsB = MATRIX_SIZE; 
    double** matB = create_matrix(rowsB, colsB);
    randomizeMatrixValues(matB,rowsB);

    //checks if the multiplication is possible 
    assert(colsA == rowsB); 

    double** matC = create_matrix(rowsA, colsB);
    int i, j, k; // declaration of iterators before loop lest OpenMP make a copy for every thread 

    omp_set_num_threads(omp_get_max_threads()); //sets number of threads to m

    printf("Maximum number of threads  = %d\n\n\n", omp_get_max_threads() );


//***********************************************************Sequential**************************************************
    t_Start = omp_get_wtime();
    for(i = 0; i < rowsA; ++i)   {                                                                                                   
        for(j = 0; j < colsB; ++j){
            for(k = 0; k < colsA; ++k) {
                matC[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
     t_Stop = omp_get_wtime() - t_Start;
    printf("\nSequential Multiplication Time: %lf seconds\n", t_Stop);


//*********************************************************Static *******************************************************

    t_Start = omp_get_wtime();
    #pragma omp parallel for schedule(static) private(i,j,k) shared(matA, matB, matC)
    for(i = 0; i < rowsA; ++i)   {                                                                                                   
        for(j = 0; j < colsB; ++j){
            for(k = 0; k < colsA; ++k) {
                matC[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
     t_Stop = omp_get_wtime() - t_Start;
    printf("\nStatic Parallel Multiplication   Time: %lf seconds\n", t_Stop);

//*********************************************************Static with chunk*******************************************************

    t_Start = omp_get_wtime();
    #pragma omp parallel for schedule(static, 4) private(i,j,k) shared(matA, matB, matC)
    for(i = 0; i < rowsA; ++i)   {                                                                                                   
        for(j = 0; j < colsB; ++j){
            for(k = 0; k < colsA; ++k) {
                matC[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
     t_Stop = omp_get_wtime() - t_Start;
    printf("\nStatic with chunk Parallel Multiplication   Time: %lf seconds\n", t_Stop);

//********************************************************Dynamic*******************************************************
    t_Start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic) private(i,j,k) shared(matC, matA, matB)
    for(i = 0; i < rowsA; ++i)   {                                                                                                   
        for(j = 0; j < colsB; ++j){
            for(k = 0; k < colsA; ++k) {
                matC[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
     t_Stop = omp_get_wtime() - t_Start;
    printf("\nDynamic Parallel Multiplication Time: %lf seconds\n", t_Stop);


//*********************************************************Dynamic with chunk********************************************
t_Start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 4) private(i,j,k) shared(matC, matA, matB)
    for(i = 0; i < rowsA; ++i)   {                                                                                                   
        for(j = 0; j < colsB; ++j){
            for(k = 0; k < colsA; ++k) {
                matC[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
     t_Stop = omp_get_wtime() - t_Start;
     printf("\nDynamic with chunk Parallel Multiplication Time: %lf seconds\n", t_Stop); 

//***************************************************Auto********************************************************************
    t_Start = omp_get_wtime();
    #pragma omp parallel for schedule(auto) private(i,j,k) shared(matA, matB, matC)
    for(i = 0; i < rowsA; ++i){                                                                                                   
        for(j = 0; j < colsB; ++j){
            for(k = 0; k < colsA; ++k) {
                matC[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
     t_Stop = omp_get_wtime() - t_Start;
    printf("\nAuto Parallel Multiplication   Time: %lf seconds\n", t_Stop);

//***************************************************Runtime********************************************************************
    t_Start = omp_get_wtime();
    #pragma omp parallel for schedule(runtime) private(i,j,k) shared(matA, matB, matC)
    for(i = 0; i < rowsA; ++i){                                                                                                   
        for(j = 0; j < colsB; ++j){
            for(k = 0; k < colsA; ++k) {
                matC[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
     t_Stop = omp_get_wtime() - t_Start;
    printf("\nRuntime Parallel Multiplication   Time: %lf seconds\n\n\n", t_Stop);



    destroy_matrix(matA, rowsA);
    destroy_matrix(matB, rowsB);
    destroy_matrix(matC, rowsA);
}