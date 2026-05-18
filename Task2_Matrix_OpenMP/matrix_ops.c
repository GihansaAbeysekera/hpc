#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

Matrix createMatrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;

    m.data = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        m.data[i] = (double *)malloc(cols * sizeof(double));
    }

    return m;
}

void freeMatrix(Matrix m) {
    for (int i = 0; i < m.rows; i++) {
        free(m.data[i]);
    }
    free(m.data);
}

int readMatrix(FILE *file, Matrix *m) {
    int rows, cols;

    if (fscanf(file, "%d,%d", &rows, &cols) != 2) {
        if (fscanf(file, "%d %d", &rows, &cols) != 2) {
            return 0;
        }
    }

    *m = createMatrix(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%lf,", &m->data[i][j]) != 1) {
                if (fscanf(file, "%lf", &m->data[i][j]) != 1) {
                    printf("Error: Invalid matrix data.\n");
                    return 0;
                }
            }
        }
    }

    return 1;
}

void printMatrix(FILE *out, Matrix m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            fprintf(out, "%.2f", m.data[i][j]);
            if (j < m.cols - 1) {
                fprintf(out, ", ");
            }
        }
        fprintf(out, "\n");
    }
}

void addMatrices(FILE *out, Matrix A, Matrix B, int threads) {
    if (A.rows != B.rows || A.cols != B.cols) {
        fprintf(out, "Addition cannot be done - different size.\n\n");
        return;
    }

    Matrix C = createMatrix(A.rows, A.cols);

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            C.data[i][j] = A.data[i][j] + B.data[i][j];
        }
    }

    fprintf(out, "Addition - %d,%d\n", C.rows, C.cols);
    printMatrix(out, C);
    fprintf(out, "\n");

    freeMatrix(C);
}

void subtractMatrices(FILE *out, Matrix A, Matrix B, int threads) {
    if (A.rows != B.rows || A.cols != B.cols) {
        fprintf(out, "Subtraction cannot be done - different size.\n\n");
        return;
    }

    Matrix C = createMatrix(A.rows, A.cols);

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            C.data[i][j] = A.data[i][j] - B.data[i][j];
        }
    }

    fprintf(out, "Subtraction - %d,%d\n", C.rows, C.cols);
    printMatrix(out, C);
    fprintf(out, "\n");

    freeMatrix(C);
}

void elementMultiply(FILE *out, Matrix A, Matrix B, int threads) {
    if (A.rows != B.rows || A.cols != B.cols) {
        fprintf(out, "Element-wise multiplication cannot be done - different size.\n\n");
        return;
    }

    Matrix C = createMatrix(A.rows, A.cols);

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            C.data[i][j] = A.data[i][j] * B.data[i][j];
        }
    }

    fprintf(out, "Element-wise Multiplication - %d,%d\n", C.rows, C.cols);
    printMatrix(out, C);
    fprintf(out, "\n");

    freeMatrix(C);
}

void elementDivide(FILE *out, Matrix A, Matrix B, int threads) {
    if (A.rows != B.rows || A.cols != B.cols) {
        fprintf(out, "Element-wise division cannot be done - different size.\n\n");
        return;
    }

    Matrix C = createMatrix(A.rows, A.cols);

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            if (B.data[i][j] == 0) {
                C.data[i][j] = NAN;
            } else {
                C.data[i][j] = A.data[i][j] / B.data[i][j];
            }
        }
    }

    fprintf(out, "Element-wise Division - %d,%d\n", C.rows, C.cols);
    printMatrix(out, C);
    fprintf(out, "\n");

    freeMatrix(C);
}

void transposeMatrix(FILE *out, Matrix A, char name, int threads) {
    Matrix T = createMatrix(A.cols, A.rows);

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            T.data[j][i] = A.data[i][j];
        }
    }

    fprintf(out, "Transpose %c - %d,%d\n", name, T.rows, T.cols);
    printMatrix(out, T);
    fprintf(out, "\n");

    freeMatrix(T);
}

void multiplyMatrices(FILE *out, Matrix A, Matrix B, int threads) {
    if (A.cols != B.rows) {
        fprintf(out, "Matrix multiplication cannot be done - A columns must equal B rows.\n\n");
        return;
    }

    Matrix C = createMatrix(A.rows, B.cols);

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            C.data[i][j] = 0;

            for (int k = 0; k < A.cols; k++) {
                C.data[i][j] += A.data[i][k] * B.data[k][j];
            }
        }
    }

    fprintf(out, "Matrix Multiplication - %d,%d\n", C.rows, C.cols);
    printMatrix(out, C);
    fprintf(out, "\n");

    freeMatrix(C);
}

int getThreadLimit(Matrix A, Matrix B, int requestedThreads) {
    int maxDim = A.rows;

    if (A.cols > maxDim) maxDim = A.cols;
    if (B.rows > maxDim) maxDim = B.rows;
    if (B.cols > maxDim) maxDim = B.cols;

    if (requestedThreads > maxDim) {
        return maxDim;
    }

    return requestedThreads;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_file> <number_of_threads>\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];
    int requestedThreads = atoi(argv[2]);

    if (requestedThreads <= 0) {
        printf("Error: Number of threads must be greater than 0.\n");
        return 1;
    }

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Cannot open input file.\n");
        return 1;
    }

    FILE *out = fopen("results.txt", "w");
    if (out == NULL) {
        printf("Error: Cannot create results.txt.\n");
        fclose(file);
        return 1;
    }

    Matrix A, B;
    int pairNumber = 1;

    while (1) {
        if (!readMatrix(file, &A)) {
            break;
        }

        if (!readMatrix(file, &B)) {
            fprintf(out, "Error: Matrix pair %d is incomplete.\n", pairNumber);
            freeMatrix(A);
            break;
        }

        int threads = getThreadLimit(A, B, requestedThreads);

        fprintf(out, "=====================================\n");
        fprintf(out, "Matrix Pair %d\n", pairNumber);
        fprintf(out, "Matrix A Size: %d,%d\n", A.rows, A.cols);
        fprintf(out, "Matrix B Size: %d,%d\n", B.rows, B.cols);
        fprintf(out, "Threads Used: %d\n", threads);
        fprintf(out, "=====================================\n\n");

        addMatrices(out, A, B, threads);
        subtractMatrices(out, A, B, threads);
        elementMultiply(out, A, B, threads);
        elementDivide(out, A, B, threads);
        transposeMatrix(out, A, 'A', threads);
        transposeMatrix(out, B, 'B', threads);
        multiplyMatrices(out, A, B, threads);

        freeMatrix(A);
        freeMatrix(B);

        pairNumber++;
    }

    fclose(file);
    fclose(out);

    printf("Matrix operations completed. Results saved in results.txt\n");

    return 0;
}
