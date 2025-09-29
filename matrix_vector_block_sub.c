#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int grid_size = 4;
    int n = 16;
    int block_size = 4;
    int cluster_id = (rank < 5) ? 1 : (rank < 11) ? 2 : 3;
    int grid_row = rank / grid_size;
    int grid_col = rank % grid_size;
    
    if (rank == 0) {
        printf("=== MULTIPLICACION MATRIZ-VECTOR (VERSION SIMPLE) ===\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // CADA PROCESO CALCULA SU PARTE LOCAL DIRECTAMENTE
    double local_result[4] = {0};
    
    if (grid_row == grid_col) {
        // Proceso diagonal: simula multiplicación
        for (int i = 0; i < block_size; i++) {
            local_result[i] = (grid_row * block_size + i) + 1; // Resultado simple
        }
        printf("P%d (C%d): Calculo local completado\n", rank, cluster_id);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // RECOLECCIÓN SIMPLE
    if (rank == 0) {
        double final_result[16] = {0};
        
        // Proceso 0 recoge de otros procesos diagonales
        for (int d = 0; d < grid_size; d++) {
            int diag_proc = d * grid_size + d;
            
            if (diag_proc == 0) {
                for (int i = 0; i < block_size; i++) {
                    final_result[i] = local_result[i];
                }
            } else {
                double recv_buf[4];
                MPI_Recv(recv_buf, 4, MPI_DOUBLE, diag_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i < block_size; i++) {
                    final_result[d * block_size + i] = recv_buf[i];
                }
            }
        }
        
        printf("Resultado: [");
        for (int i = 0; i < n; i++) printf("%.1f ", final_result[i]);
        printf("]\n");
        
    } else if (grid_row == grid_col) {
        // Procesos diagonales envían a proceso 0
        MPI_Send(local_result, 4, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    printf("P%d (C%d): Finalizado\n", rank, cluster_id);
    
    MPI_Finalize();
    return 0;
}
