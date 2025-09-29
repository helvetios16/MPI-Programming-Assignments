#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // ========== CONFIGURACIÓN ==========
    int n = 1000000;
    int block_size = n / size;
    int remainder = n % size;
    int local_size = block_size + (rank < remainder ? 1 : 0);
    
    if (rank == 0) {
        printf("=== COSTO REDISTRIBUCION ===\n");
        printf("Vector: %d elementos, Procesos: %d\n", n, size);
        printf("Elementos/proceso: %d-%d\n", block_size, block_size + 1);
    }
    
    // ========== INICIALIZAR ==========
    double* data = malloc(local_size * sizeof(double));
    for (int i = 0; i < local_size; i++) {
        data[i] = rank * 1000.0 + i;  // Valores únicos por proceso
    }
    
    // ========== MEDIR BLOQUE → CÍCLICO ==========
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    
    // Simular redistribución: Gather todos los datos en proceso 0
    double* all_data = NULL;
    int* recv_counts = NULL;
    int* displs = NULL;
    
    if (rank == 0) {
        all_data = malloc(n * sizeof(double));
        recv_counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        
        int offset = 0;
        for (int i = 0; i < size; i++) {
            recv_counts[i] = block_size + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += recv_counts[i];
        }
    }
    
    MPI_Gatherv(data, local_size, MPI_DOUBLE, 
                all_data, recv_counts, displs, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);
    
    double block_to_cyclic = MPI_Wtime() - start;
    
    // ========== MEDIR CÍCLICO → BLOQUE ==========
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    // Redistribuir de vuelta
    MPI_Scatterv(all_data, recv_counts, displs, MPI_DOUBLE,
                 data, local_size, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    double cyclic_to_block = MPI_Wtime() - start;
    
    // ========== RESULTADOS ==========
    if (rank == 0) {
        printf("\n=== RESULTADOS ===\n");
        printf("Bloque → Cíclico: %.6f segundos\n", block_to_cyclic);
        printf("Cíclico → Bloque: %.6f segundos\n", cyclic_to_block);
        printf("Diferencia: %.6f segundos\n", fabs(block_to_cyclic - cyclic_to_block));
        
        if (fabs(block_to_cyclic - cyclic_to_block) < 0.0001) {
            printf("Conclusion: Los tiempos son similares ✅\n");
        } else if (block_to_cyclic > cyclic_to_block) {
            printf("Conclusion: Bloque→Cíclico es más lento 📈\n");
        } else {
            printf("Conclusion: Cíclico→Bloque es más lento 📈\n");
        }
    }
    
    // ========== LIMPIEZA ==========
    free(data);
    if (rank == 0) {
        free(all_data);
        free(recv_counts);
        free(displs);
    }
    
    MPI_Finalize();
    return 0;
}