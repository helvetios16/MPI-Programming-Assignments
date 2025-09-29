#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Generador de números aleatorios
static unsigned long next = 1;

int rand(void) {
  next = next * 1103515245 + 12345;
  return (unsigned int)(next / 65536) % 32768;
}

double random_double(double min, double max) {
  return min + ((double)rand() / 32767.0) * (max - min);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  long long int total_tosses = 10000000LL;

  if (rank == 0) {
    if (argc == 2) {
      total_tosses = atoll(argv[1]);
    }
    printf("=== MONTE CARLO π (OPTIMIZADO) ===\n");
    printf("Total tosses: %lld\n", total_tosses);
    printf("Processes: %d\n", size);
  }

  MPI_Bcast(&total_tosses, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

  long long int local_tosses = total_tosses / size;
  long long int local_number_in_circle = 0;

  // Semilla más simple y rápida
  next = (unsigned long)time(NULL) + rank;

  int cluster_id;
  if (rank < size / 3) {
    cluster_id = 1;
  } else if (rank < 2 * (size / 3)) {
    cluster_id = 2;
  } else {
    cluster_id = 3;
  }

  printf("Process %d (Cluster %d): Starting %lld tosses...\n", rank, cluster_id,
         local_tosses);
  double start_time = MPI_Wtime();

  // BUCLE OPTIMIZADO - MÁS RÁPIDO
  for (long long int i = 0; i < local_tosses; i++) {
    // Generación de números aleatorios
    double x = -1.0 + 2.0 * ((double)rand() / 32767.0);
    double y = -1.0 + 2.0 * ((double)rand() / 32767.0);

    // Cálculo optimizado
    if (x * x + y * y <= 1.0) {
      local_number_in_circle++;
    }

    // Mostrar progreso cada 1M de iteraciones (solo para rank 0)
    if (rank == 0 && i > 0 && i % 1000000 == 0) {
      printf("Progress: %lld/%lld (%.1f%%)\n", i, local_tosses,
             (double)i / local_tosses * 100);
    }
  }

  double local_time = MPI_Wtime() - start_time;
  printf("Process %d (Cluster %d): Completed in %.2f seconds\n", rank,
         cluster_id, local_time);

  // Reducción rápida
  long long int global_number_in_circle;
  MPI_Reduce(&local_number_in_circle, &global_number_in_circle, 1,
             MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    double pi_estimate = 4.0 * global_number_in_circle / (double)total_tosses;
    double total_time = MPI_Wtime() - start_time;

    printf("\n=== RESULTS ===\n");
    printf("Total tosses: %lld\n", total_tosses);
    printf("Points in circle: %lld\n", global_number_in_circle);
    printf("π estimate: %.10f\n", pi_estimate);
    printf("Actual π:    %.10f\n", M_PI);
    printf("Error: %.10f\n", fabs(M_PI - pi_estimate));
    printf("Total time: %.2f seconds\n", total_time);

    // Simulación simple de 3 clusters
    printf("\n=== 3-CLUSTER SIMULATION ===\n");
    int processes_per_cluster = size / 3;
    printf("Cluster 1: processes 0-%d\n", processes_per_cluster - 1);
    printf("Cluster 2: processes %d-%d\n", processes_per_cluster,
           2 * processes_per_cluster - 1);
    printf("Cluster 3: processes %d-%d\n", 2 * processes_per_cluster, size - 1);
  }

  MPI_Finalize();
  return 0;
}
