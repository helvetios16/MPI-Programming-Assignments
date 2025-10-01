#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Solo 3 procesos (uno por cluster)
  if (rank > 2) {
    MPI_Finalize();
    return 0;
  }

  // ========== CONFIGURACIÓN ==========
  int cluster_id = rank + 1; // Cluster 1, 2, 3
  int trials = 10;
  int data = 42, recv_data;

  if (rank == 0) {
    printf("=== PING-PONG ENTRE 3 CLUSTERS ===\n");
    printf("P0: Cluster 1, P1: Cluster 2, P2: Cluster 3\n");
    printf("Pruebas: %d\n\n", trials);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // ========== 1. CADENA: C1 → C2 → C3 → C1 ==========
  double start = MPI_Wtime();

  for (int i = 0; i < trials; i++) {
    if (rank == 0) {
      MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(&recv_data, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
      MPI_Recv(&recv_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&data, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
    } else if (rank == 2) {
      MPI_Recv(&recv_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }

  double time_per_hop = (MPI_Wtime() - start) / (3.0 * trials);
  printf("P%d (Cluster %d): Cadena = %.9f seg/salto\n", rank, cluster_id,
         time_per_hop);

  MPI_Barrier(MPI_COMM_WORLD);

  // ========== 2. PING-PONG DIRECTOS ==========
  if (rank == 0)
    printf("\n=== PING-PONG DIRECTOS ===\n");
  MPI_Barrier(MPI_COMM_WORLD);

  // Cluster 1 ↔ Cluster 2
  if (rank == 0 || rank == 1) {
    int partner = 1 - rank;

    start = MPI_Wtime();
    for (int i = 0; i < trials; i++) {
      if (rank == 0) {
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&recv_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(&recv_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Send(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      }
    }
    double time_msg = (MPI_Wtime() - start) / (2.0 * trials);
    printf("P%d (Cluster %d): C1↔C2 = %.9f seg/msg\n", rank, cluster_id,
           time_msg);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Cluster 2 ↔ Cluster 3
  if (rank == 1 || rank == 2) {
    int partner = (rank == 1) ? 2 : 1;

    start = MPI_Wtime();
    for (int i = 0; i < trials; i++) {
      if (rank == 1) {
        MPI_Send(&data, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
        MPI_Recv(&recv_data, 1, MPI_INT, 2, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(&recv_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
      }
    }
    double time_msg = (MPI_Wtime() - start) / (2.0 * trials);
    printf("P%d (Cluster %d): C2↔C3 = %.9f seg/msg\n", rank, cluster_id,
           time_msg);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Cluster 3 ↔ Cluster 1
  if (rank == 2 || rank == 0) {
    int partner = (rank == 2) ? 0 : 2;

    start = MPI_Wtime();
    for (int i = 0; i < trials; i++) {
      if (rank == 2) {
        MPI_Send(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Recv(&recv_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(&recv_data, 1, MPI_INT, 2, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Send(&data, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
      }
    }
    double time_msg = (MPI_Wtime() - start) / (2.0 * trials);
    printf("P%d (Cluster %d): C3↔C1 = %.9f seg/msg\n", rank, cluster_id,
           time_msg);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  printf("P%d (Cluster %d): Finalizado\n", rank, cluster_id);

  MPI_Finalize();
  return 0;
}
