#include <math.h>
#include <mpi.h>
#include <stdio.h>

// Función para verificar si un número es potencia de dos
int is_power_of_two(int x) { return (x > 0) && ((x & (x - 1)) == 0); }

// Versión para potencias de dos
double tree_sum_power_of_two(double local_val, int rank, int size,
                             MPI_Comm comm) {
  double sum = local_val;
  int partner;
  int current_size = size;

  while (current_size > 1) {
    int half = current_size / 2;

    if (rank < half) {
      // Proceso receptor
      partner = rank + half;
      double received_val;
      MPI_Recv(&received_val, 1, MPI_DOUBLE, partner, 0, comm,
               MPI_STATUS_IGNORE);
      sum += received_val;
    } else if (rank < current_size) {
      // Proceso emisor
      partner = rank - half;
      MPI_Send(&sum, 1, MPI_DOUBLE, partner, 0, comm);
      return 0.0;
    }

    current_size = half;
  }

  return sum;
}

// Versión para cualquier tamaño
double tree_sum_any_size(double local_val, int rank, int size, MPI_Comm comm) {
  double sum = local_val;
  int mask = 1;

  while (mask < size) {
    int partner = rank ^ mask;

    if (partner < size) {
      if (rank < partner) {
        // Receptor
        double received_val;
        MPI_Recv(&received_val, 1, MPI_DOUBLE, partner, 0, comm,
                 MPI_STATUS_IGNORE);
        sum += received_val;
      } else {
        // Emisor
        MPI_Send(&sum, 1, MPI_DOUBLE, partner, 0, comm);
        return 0.0;
      }
    }

    mask <<= 1;
  }

  return sum;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Valor local de cada proceso
  double local_value = (double)(rank + 1);
  double tree_sum_result, mpi_sum_result;

  // ========== FASE 1: INFORMACIÓN INICIAL ==========
  if (rank == 0) {
    printf("=== SUMA GLOBAL EN ÁRBOL ===\n");
    printf("Procesos: %d (%s potencia de 2)\n", size,
           is_power_of_two(size) ? "es" : "NO es");
    printf("Valor local de cada proceso: ");
    for (int i = 0; i < size; i++) {
      printf("%.0f ", (double)(i + 1));
    }
    printf("\nSuma teórica: %.0f\n\n", (double)size * (size + 1) / 2);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // ========== FASE 2: SIMULACIÓN DE 3 CLUSTERS ==========
  int cluster_id;
  if (rank < size / 3)
    cluster_id = 1;
  else if (rank < 2 * (size / 3))
    cluster_id = 2;
  else
    cluster_id = 3;

  printf("Proceso %2d (Cluster %d): valor local = %.0f\n", rank, cluster_id,
         local_value);

  MPI_Barrier(MPI_COMM_WORLD);

  // ========== FASE 3: SUMA CON MPI_REDUCE (REFERENCIA) ==========
  MPI_Reduce(&local_value, &mpi_sum_result, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  // ========== FASE 4: SUMA EN ÁRBOL (ALGORITMO PROPIO) ==========
  double start_time = MPI_Wtime();

  if (is_power_of_two(size)) {
    tree_sum_result =
        tree_sum_power_of_two(local_value, rank, size, MPI_COMM_WORLD);
  } else {
    tree_sum_result =
        tree_sum_any_size(local_value, rank, size, MPI_COMM_WORLD);
  }

  double tree_sum_time = MPI_Wtime() - start_time;

  // ========== FASE 5: RESULTADOS Y VERIFICACIÓN ==========
  if (rank == 0) {
    printf("\n=== RESULTADOS ===\n");
    printf("MPI_Reduce:    %.0f\n", mpi_sum_result);
    printf("Tree Sum:      %.0f\n", tree_sum_result);
    printf("Tiempo árbol:  %.6f segundos\n", tree_sum_time);

    if (fabs(tree_sum_result - mpi_sum_result) < 1e-10) {
      printf("✅ SUMA CORRECTA\n");
    } else {
      printf("❌ ERROR: diferencia = %.10f\n",
             fabs(tree_sum_result - mpi_sum_result));
    }

    // Información de los clusters
    printf("\n=== CONFIGURACIÓN DE CLUSTERS ===\n");
    int c1_end = size / 3 - 1;
    int c2_start = size / 3;
    int c2_end = 2 * (size / 3) - 1;
    int c3_start = 2 * (size / 3);

    printf("Cluster 1: procesos 0-%d\n", c1_end);
    printf("Cluster 2: procesos %d-%d\n", c2_start, c2_end);
    printf("Cluster 3: procesos %d-%d\n", c3_start, size - 1);
    printf("El árbol de comunicación conecta todos los clusters\n");
  }

  MPI_Finalize();
  return 0;
}
