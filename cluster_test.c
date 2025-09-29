#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  char hostname[256];
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  int node_id;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Get_processor_name(processor_name, &name_len);
  gethostname(hostname, 255);

  // Simular 3 nodos diferentes basado en el rango del proceso
  // Ahora con 16 procesos totales
  if (rank < 5) {
    node_id = 1; // Nodo 1: procesos 0-4 (5 procesos)
    strcpy(processor_name, "node1");
  } else if (rank < 10) {
    node_id = 2; // Nodo 2: procesos 5-9 (5 procesos)
    strcpy(processor_name, "node2");
  } else {
    node_id = 3; // Nodo 3: procesos 10-15 (6 procesos)
    strcpy(processor_name, "node3");
  }

  printf("🖥️  Proceso %2d/%d | Nodo: %d (%s) | Host real: %s\n", rank, size,
         node_id, processor_name, hostname);

  // Sincronizar
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    printf("\n");
    printf("=========================================\n");
    printf("✅ CLUSTER DE 3 NODOS SIMULADO\n");
    printf("=========================================\n");
    printf("Nodo 1: Procesos 0-4 (5 cores)\n");
    printf("Nodo 2: Procesos 5-9 (5 cores)\n");
    printf("Nodo 3: Procesos 10-15 (6 cores)\n");
    printf("Total de procesos: %d\n", size);
  }

  MPI_Finalize();
  return 0;
}
