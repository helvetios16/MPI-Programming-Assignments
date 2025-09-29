#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Función para encontrar el bin de un valor
int find_bin(double value, double *bin_maxes, int bin_count, double min_meas) {
  if (value < min_meas)
    return -1; // Valor por debajo del mínimo

  // Búsqueda binaria para eficiencia con muchos bins
  int left = 0, right = bin_count - 1;

  while (left <= right) {
    int mid = left + (right - left) / 2;

    if (mid == 0) {
      // Caso especial para el primer bin
      if (value >= min_meas && value < bin_maxes[0]) {
        return 0;
      }
      left = mid + 1;
    } else {
      if (value >= bin_maxes[mid - 1] && value < bin_maxes[mid]) {
        return mid;
      } else if (value < bin_maxes[mid - 1]) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
  }

  // Si llegamos aquí, el valor está en el último bin o por encima
  if (value >= bin_maxes[bin_count - 1]) {
    return bin_count; // Por encima del máximo
  }

  return -1; // No debería ocurrir
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Parámetros del histograma (los define el proceso 0)
  int data_count = 0;
  double *data = NULL;
  double min_meas = 0.0, max_meas = 0.0;
  int bin_count = 0;
  double *bin_maxes = NULL;

  // Variables locales para cada proceso
  int local_data_count;
  double *local_data = NULL;
  int *local_bin_counts = NULL;
  int *global_bin_counts = NULL;

  // ========== FASE 1: PROCESO 0 LEE LOS DATOS ==========
  if (rank == 0) {
    // Datos de ejemplo del libro (página 67)
    double sample_data[] = {1.3, 2.9, 0.4, 0.3, 1.3, 4.4, 1.7, 0.4, 3.2, 0.3,
                            4.9, 2.4, 3.1, 4.4, 3.9, 0.4, 4.2, 4.5, 4.9, 0.9};

    data_count = sizeof(sample_data) / sizeof(sample_data[0]);
    min_meas = 0.0;
    max_meas = 5.0;
    bin_count = 5;

    data = (double *)malloc(data_count * sizeof(double));
    memcpy(data, sample_data, data_count * sizeof(double));

    printf("=== DATOS DEL HISTOGRAMA ===\n");
    printf("Cantidad de datos: %d\n", data_count);
    printf("Rango: [%.1f, %.1f]\n", min_meas, max_meas);
    printf("Número de bins: %d\n", bin_count);
    printf("Datos: ");
    for (int i = 0; i < data_count; i++) {
      printf("%.1f ", data[i]);
    }
    printf("\n\n");

    // Calcular los límites de los bins
    double bin_width = (max_meas - min_meas) / bin_count;
    bin_maxes = (double *)malloc(bin_count * sizeof(double));
    for (int b = 0; b < bin_count; b++) {
      bin_maxes[b] = min_meas + bin_width * (b + 1);
    }
  }

  // ========== FASE 2: DISTRIBUIR PARÁMETROS A TODOS LOS PROCESOS ==========
  MPI_Bcast(&data_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&min_meas, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&max_meas, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bin_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Proceso 0 envía bin_maxes a todos
  if (rank != 0) {
    bin_maxes = (double *)malloc(bin_count * sizeof(double));
  }
  MPI_Bcast(bin_maxes, bin_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // ========== FASE 3: DISTRIBUIR LOS DATOS ENTRE PROCESOS ==========
  // Calcular cuántos datos le tocan a cada proceso
  int base_count = data_count / size;
  int extra = data_count % size;

  if (rank < extra) {
    local_data_count = base_count + 1;
  } else {
    local_data_count = base_count;
  }

  local_data = (double *)malloc(local_data_count * sizeof(double));

  // Proceso 0 distribuye los datos
  int *sendcounts = NULL;
  int *displs = NULL;

  if (rank == 0) {
    sendcounts = (int *)malloc(size * sizeof(int));
    displs = (int *)malloc(size * sizeof(int));

    int offset = 0;
    for (int i = 0; i < size; i++) {
      sendcounts[i] = (i < extra) ? base_count + 1 : base_count;
      displs[i] = offset;
      offset += sendcounts[i];
    }
  }

  MPI_Scatterv(data, sendcounts, displs, MPI_DOUBLE, local_data,
               local_data_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // ========== FASE 4: CADA PROCESO CALCULA SU HISTOGRAMA LOCAL ==========
  local_bin_counts = (int *)calloc(bin_count, sizeof(int));

  for (int i = 0; i < local_data_count; i++) {
    int bin = find_bin(local_data[i], bin_maxes, bin_count, min_meas);
    if (bin >= 0 && bin < bin_count) {
      local_bin_counts[bin]++;
    }
  }

  // ========== FASE 5: COMBINAR HISTOGRAMAS LOCALES ==========
  if (rank == 0) {
    global_bin_counts = (int *)malloc(bin_count * sizeof(int));
  }

  MPI_Reduce(local_bin_counts, global_bin_counts, bin_count, MPI_INT, MPI_SUM,
             0, MPI_COMM_WORLD);

  // ========== FASE 6: PROCESO 0 IMPRIME EL RESULTADO ==========
  if (rank == 0) {
    printf("=== RESULTADO DEL HISTOGRAMA ===\n");

    double bin_width = (max_meas - min_meas) / bin_count;

    for (int b = 0; b < bin_count; b++) {
      double bin_start = (b == 0) ? min_meas : bin_maxes[b - 1];
      double bin_end = bin_maxes[b];

      printf("Bin %d [%5.1f - %5.1f): ", b, bin_start, bin_end);

      // Imprimir barra del histograma
      int bar_length = global_bin_counts[b] * 3; // Escala para visualización
      for (int i = 0; i < bar_length; i++) {
        printf("█");
      }
      printf(" %d elementos\n", global_bin_counts[b]);
    }

    // Información de distribución
    printf("\n=== DISTRIBUCIÓN EN CLUSTERS ===\n");
    printf("Total de procesos: %d\n", size);
    printf("Cluster 1 (procesos 0-%d): %d procesos\n", (size / 3) - 1,
           (size / 3));
    printf("Cluster 2 (procesos %d-%d): %d procesos\n", (size / 3),
           2 * (size / 3) - 1, (size / 3));
    printf("Cluster 3 (procesos %d-%d): %d procesos\n", 2 * (size / 3),
           size - 1, size - 2 * (size / 3));
  }

  // ========== FASE 7: INFORMACIÓN POR CLUSTER ==========
  // Simular los 3 clusters
  int cluster_id;
  if (rank < size / 3) {
    cluster_id = 1;
  } else if (rank < 2 * (size / 3)) {
    cluster_id = 2;
  } else {
    cluster_id = 3;
  }

  // Cada proceso imprime su información local
  printf("Proceso %2d (Cluster %d): procesé %2d elementos, "
         "bin más frecuente: ",
         rank, cluster_id, local_data_count);

  int max_local_bin = 0;
  int max_local_count = 0;
  for (int b = 0; b < bin_count; b++) {
    if (local_bin_counts[b] > max_local_count) {
      max_local_count = local_bin_counts[b];
      max_local_bin = b;
    }
  }
  printf("bin %d con %d elementos\n", max_local_bin, max_local_count);

  // ========== FASE 8: LIMPIEZA ==========
  free(local_data);
  free(local_bin_counts);
  if (rank == 0) {
    free(data);
    free(bin_maxes);
    free(sendcounts);
    free(displs);
    free(global_bin_counts);
  } else {
    free(bin_maxes);
  }

  MPI_Finalize();
  return 0;
}
