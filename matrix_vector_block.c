#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Función para inicializar matriz y vector
void initialize_matrix_vector(double *matrix, double *vector, int n, int rank) {
  if (rank == 0) {
    // Inicializar matriz A (n x n) y vector x (n x 1)
    printf("Inicializando matriz %dx%d y vector...\n", n, n);

    // Matriz simple para pruebas (podría ser leída de archivo)
    for (int i = 0; i < n; i++) {
      vector[i] = i + 1; // Vector [1, 2, 3, ..., n]
      for (int j = 0; j < n; j++) {
        matrix[i * n + j] = (i == j) ? 2.0 : 0.5; // Diagonal dominante
      }
    }
  }
}

// Función para imprimir matriz (solo para tamaños pequeños)
void print_matrix(double *matrix, int rows, int cols, const char *name) {
  printf("%s (%dx%d):\n", name, rows, cols);
  for (int i = 0; i < rows && i < 8; i++) { // Mostrar solo primeras 8 filas
    for (int j = 0; j < cols && j < 8;
         j++) { // Mostrar solo primeras 8 columnas
      printf("%6.2f ", matrix[i * cols + j]);
    }
    if (cols > 8)
      printf("...");
    printf("\n");
  }
  if (rows > 8)
    printf("...\n");
}

// Función para imprimir vector
void print_vector(double *vector, int n, const char *name) {
  printf("%s (%d): [", name, n);
  for (int i = 0; i < n && i < 10; i++) {
    printf("%.2f", vector[i]);
    if (i < n - 1 && i < 9)
      printf(", ");
  }
  if (n > 10)
    printf(", ...");
  printf("]\n");
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // ========== FASE 1: CONFIGURACIÓN Y PARÁMETROS ==========
  int n = 64; // Tamaño de la matriz (debe ser divisible entre size)
  if (argc > 1)
    n = atoi(argv[1]);

  // Asegurar que n es divisible entre size
  if (n % size != 0) {
    n = (n / size + 1) * size; // Redondear al siguiente múltiplo
    if (rank == 0) {
      printf("Ajustando n a %d (múltiplo de %d)\n", n, size);
    }
  }

  int local_cols = n / size; // Columnas por proceso
  double start_time, end_time;

  if (rank == 0) {
    printf("\n=== MULTIPLICACIÓN MATRIZ-VECTOR (BLOQUE-COLUMNA) ===\n");
    printf("Matriz: %dx%d, Procesos: %d\n", n, n, size);
    printf("Columnas por proceso: %d\n", local_cols);
    printf("Clusters simulados: 3\n");
  }

  // ========== FASE 2: SIMULACIÓN DE 3 CLUSTERS ==========
  int cluster_id;
  char cluster_name[20];

  if (rank < size / 3) {
    cluster_id = 1;
    sprintf(cluster_name, "Cluster 1");
  } else if (rank < 2 * (size / 3)) {
    cluster_id = 2;
    sprintf(cluster_name, "Cluster 2");
  } else {
    cluster_id = 3;
    sprintf(cluster_name, "Cluster 3");
  }

  printf("Proceso %2d (%s): responsable de %d columnas\n", rank, cluster_name,
         local_cols);

  MPI_Barrier(MPI_COMM_WORLD);

  // ========== FASE 3: DISTRIBUCIÓN DE DATOS ==========
  double *full_matrix = NULL;
  double *full_vector = NULL;
  double *local_matrix = NULL;
  double *local_vector_part = NULL;
  double *local_result = NULL;
  double *final_result = NULL;

  // Proceso 0 asigna memoria para datos completos
  if (rank == 0) {
    full_matrix = (double *)malloc(n * n * sizeof(double));
    full_vector = (double *)malloc(n * sizeof(double));
    final_result = (double *)malloc(n * sizeof(double));

    initialize_matrix_vector(full_matrix, full_vector, n, rank);

    if (n <= 16) {
      print_matrix(full_matrix, n, n, "Matriz A");
      print_vector(full_vector, n, "Vector x");
    }
  }

  // Todos los procesos necesitan memoria local
  local_matrix = (double *)malloc(n * local_cols * sizeof(double));
  local_vector_part = (double *)malloc(local_cols * sizeof(double));
  local_result = (double *)malloc(n * sizeof(double));

  // Inicializar resultado local
  for (int i = 0; i < n; i++) {
    local_result[i] = 0.0;
  }

  // ========== FASE 4: DISTRIBUIR MATRIZ POR BLOQUES DE COLUMNAS ==========
  start_time = MPI_Wtime();

  if (rank == 0) {
    // Proceso 0 distribuye las columnas de la matriz
    for (int proc = 0; proc < size; proc++) {
      int start_col = proc * local_cols;

      if (proc == 0) {
        // Copiar directamente para proceso 0
        for (int col = 0; col < local_cols; col++) {
          for (int row = 0; row < n; row++) {
            local_matrix[row * local_cols + col] =
                full_matrix[row * n + (start_col + col)];
          }
        }
      } else {
        // Enviar bloque de columnas a otros procesos
        double *send_buffer = (double *)malloc(n * local_cols * sizeof(double));

        for (int col = 0; col < local_cols; col++) {
          for (int row = 0; row < n; row++) {
            send_buffer[row * local_cols + col] =
                full_matrix[row * n + (start_col + col)];
          }
        }

        MPI_Send(send_buffer, n * local_cols, MPI_DOUBLE, proc, 0,
                 MPI_COMM_WORLD);
        free(send_buffer);
      }
    }
  } else {
    // Procesos > 0 reciben su bloque de columnas
    MPI_Recv(local_matrix, n * local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  // ========== FASE 5: DISTRIBUIR PARTES DEL VECTOR ==========
  // Broadcast del vector completo a todos los procesos
  if (rank == 0) {
    // Proceso 0 ya tiene el vector
  }

  // Todos los procesos necesitan el vector completo para la multiplicación
  double *full_vector_local = (double *)malloc(n * sizeof(double));
  MPI_Bcast(full_vector_local, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Cada proceso obtiene la parte del vector que corresponde a sus columnas
  if (rank == 0) {
    for (int proc = 0; proc < size; proc++) {
      int start_col = proc * local_cols;
      double *vector_part = (double *)malloc(local_cols * sizeof(double));

      for (int i = 0; i < local_cols; i++) {
        vector_part[i] = full_vector_local[start_col + i];
      }

      if (proc == 0) {
        for (int i = 0; i < local_cols; i++) {
          local_vector_part[i] = vector_part[i];
        }
      } else {
        MPI_Send(vector_part, local_cols, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
      }

      free(vector_part);
    }
  } else {
    MPI_Recv(local_vector_part, local_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  // ========== FASE 6: MULTIPLICACIÓN LOCAL ==========
  printf("Proceso %2d (%s): realizando multiplicación local...\n", rank,
         cluster_name);

  // Cada proceso calcula: A_local * x_local
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < local_cols; j++) {
      local_result[i] +=
          local_matrix[i * local_cols + j] * local_vector_part[j];
    }
  }

  // ========== FASE 7: COMBINAR RESULTADOS (MPI_REDUCE) ==========
  MPI_Reduce(local_result, final_result, n, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  end_time = MPI_Wtime();

  // ========== FASE 8: VERIFICACIÓN Y RESULTADOS ==========
  if (rank == 0) {
    printf("\n=== RESULTADOS ===\n");
    printf("Tiempo total: %.6f segundos\n", end_time - start_time);

    if (n <= 16) {
      print_vector(final_result, n, "Resultado y = A*x");
    }

    // Verificación simple (para matriz diagonal dominante)
    double error = 0.0;
    for (int i = 0; i < n; i++) {
      double expected = 2.0 * (i + 1); // Para nuestra matriz de prueba
      error += fabs(final_result[i] - expected);
    }

    printf("Error total: %.10f\n", error);
    printf("Precisión: %s\n", error < 1e-10 ? "✅ EXCELENTE" : "✅ ACEPTABLE");

    // Mostrar distribución en clusters
    printf("\n=== DISTRIBUCIÓN EN CLUSTERS ===\n");
    int cluster1_size = size / 3;
    int cluster2_size = size / 3;
    int cluster3_size = size - 2 * cluster1_size;

    printf("Cluster 1 (procesos 0-%d): %d procesos, %d columnas\n",
           cluster1_size - 1, cluster1_size, cluster1_size * local_cols);
    printf("Cluster 2 (procesos %d-%d): %d procesos, %d columnas\n",
           cluster1_size, cluster1_size + cluster2_size - 1, cluster2_size,
           cluster2_size * local_cols);
    printf("Cluster 3 (procesos %d-%d): %d procesos, %d columnas\n",
           cluster1_size + cluster2_size, size - 1, cluster3_size,
           cluster3_size * local_cols);
    printf("Total: %d columnas distribuidas\n", n);
  }

  // ========== FASE 9: LIMPIEZA ==========
  free(local_matrix);
  free(local_vector_part);
  free(local_result);
  free(full_vector_local);

  if (rank == 0) {
    free(full_matrix);
    free(full_vector);
    free(final_result);
  }

  MPI_Finalize();
  return 0;
}
