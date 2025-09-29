#include <math.h>
#include <mpi.h>
#include <stdio.h>

// Función para verificar si un número es potencia de dos
int is_power_of_two(int x) { return (x > 0) && ((x & (x - 1)) == 0); }

// Versión Butterfly para potencias de dos
double butterfly_sum_power_of_two(double local_val, int rank, int size,
                                  MPI_Comm comm) {
  double sum = local_val;
  int partner;

  for (int step = 1; step < size; step *= 2) {
    // Calcular pareja para este paso
    partner = rank ^ step; // XOR para encontrar pareja

    if (partner < size) {
      double received_val;
      // Intercambiar valores con la pareja
      MPI_Sendrecv(&sum, 1, MPI_DOUBLE, partner, 0, &received_val, 1,
                   MPI_DOUBLE, partner, 0, comm, MPI_STATUS_IGNORE);

      // Sumar el valor recibido
      sum += received_val;
    }
  }

  return sum; // Todos los procesos tienen la suma total
}

// Versión Butterfly para cualquier tamaño
double butterfly_sum_any_size(double local_val, int rank, int size,
                              MPI_Comm comm) {
  double sum = local_val;
  int dim = (int)ceil(log2(size)); // Dimensión del hipercubo

  for (int d = 0; d < dim; d++) {
    int mask = 1 << d; // Máscara para esta dimensión
    int partner = rank ^ mask;

    if (partner < size) {
      double received_val;
      // Intercambiar valores con la pareja
      MPI_Sendrecv(&sum, 1, MPI_DOUBLE, partner, 0, &received_val, 1,
                   MPI_DOUBLE, partner, 0, comm, MPI_STATUS_IGNORE);

      sum += received_val;
    }
    // Si la pareja no existe, este proceso simplemente continúa
  }

  return sum;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Valor local de cada proceso (diferente para hacerlo interesante)
  double local_value = (double)((rank + 1) * 10); // 10, 20, 30, ...
  double butterfly_result, mpi_sum_result;
  double butterfly_result_any = 0.0;

  // ========== FASE 1: INFORMACIÓN INICIAL ==========
  if (rank == 0) {
    printf("=== SUMA GLOBAL BUTTERFLY (3 CLUSTERS, %d PROCESOS) ===\n", size);
    printf("Configuración: %s potencia de 2\n",
           is_power_of_two(size) ? "ES" : "NO es");
    printf("\nValores locales por proceso:\n");
    for (int i = 0; i < size; i++) {
      printf("P%d: %.0f  ", i, (double)((i + 1) * 10));
      if ((i + 1) % 8 == 0)
        printf("\n");
    }
    printf("\nSuma teórica: %.0f\n", (double)size * (size + 1) / 2 * 10);
  }

  MPI_Barrier(MPI_COMM_WORLD);

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

  printf("Proceso %2d (%s): valor inicial = %.0f\n", rank, cluster_name,
         local_value);

  MPI_Barrier(MPI_COMM_WORLD);

  // ========== FASE 3: SUMA CON MPI_REDUCE (REFERENCIA) ==========
  MPI_Reduce(&local_value, &mpi_sum_result, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  // ========== FASE 4: BUTTERFLY PARA POTENCIAS DE 2 ==========
  if (is_power_of_two(size)) {
    double start_time = MPI_Wtime();
    butterfly_result =
        butterfly_sum_power_of_two(local_value, rank, size, MPI_COMM_WORLD);
    double butterfly_time = MPI_Wtime() - start_time;

    if (rank == 0) {
      printf("\n=== BUTTERFLY (POTENCIA DE 2) ===\n");
      printf("Resultado: %.0f\n", butterfly_result);
      printf("Tiempo: %.6f segundos\n", butterfly_time);
    }
  }

  // ========== FASE 5: BUTTERFLY PARA CUALQUIER TAMAÑO ==========
  double start_time_any = MPI_Wtime();
  butterfly_result_any =
      butterfly_sum_any_size(local_value, rank, size, MPI_COMM_WORLD);
  double butterfly_time_any = MPI_Wtime() - start_time_any;

  if (rank == 0) {
    printf("\n=== BUTTERFLY (CUALQUIER TAMAÑO) ===\n");
    printf("Resultado: %.0f\n", butterfly_result_any);
    printf("Tiempo: %.6f segundos\n", butterfly_time_any);
  }

  // ========== FASE 6: VERIFICACIÓN Y RESULTADOS ==========
  if (rank == 0) {
    printf("\n=== VERIFICACIÓN ===\n");
    printf("MPI_Reduce:    %.0f\n", mpi_sum_result);

    if (is_power_of_two(size)) {
      printf("Butterfly P2:  %.0f - %s\n", butterfly_result,
             fabs(butterfly_result - mpi_sum_result) < 1e-10 ? "✅ CORRECTO"
                                                             : "❌ ERROR");
    }

    printf("Butterfly Any: %.0f - %s\n", butterfly_result_any,
           fabs(butterfly_result_any - mpi_sum_result) < 1e-10 ? "✅ CORRECTO"
                                                               : "❌ ERROR");

    // Mostrar estructura de comunicación
    printf("\n=== PATRÓN BUTTERFLY EXPLICADO ===\n");
    printf("En Butterfly, TODOS los procesos obtienen el resultado final\n");
    printf("Cada paso duplica la información disponible\n");
    printf("Número de pasos: ⌈log2(%d)⌉ = %d\n", size, (int)ceil(log2(size)));
  }

  // ========== FASE 7: DEMOSTRACIÓN DE LOS PASOS BUTTERFLY ==========
  MPI_Barrier(MPI_COMM_WORLD);

  if (size <= 16) { // Solo mostrar para tamaños manejables
    if (rank == 0)
      printf("\n=== DEMOSTRACIÓN PASO A PASO ===\n");

    double current_value = local_value;
    int steps = (int)ceil(log2(size));

    for (int step = 0; step < steps; step++) {
      MPI_Barrier(MPI_COMM_WORLD);

      int mask = 1 << step;
      int partner = rank ^ mask;

      if (partner < size) {
        double received_val;
        MPI_Sendrecv(&current_value, 1, MPI_DOUBLE, partner, 0, &received_val,
                     1, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

        double old_value = current_value;
        current_value += received_val;

        printf("P%02d (%s): Paso %d - intercambio con P%02d (%.0f + %.0f = "
               "%.0f)\n",
               rank, cluster_name, step + 1, partner, old_value, received_val,
               current_value);
      } else {
        printf("P%02d (%s): Paso %d - sin pareja (mantiene %.0f)\n", rank,
               cluster_name, step + 1, current_value);
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0 && step < steps - 1)
        printf("\n");
    }

    // Verificar que todos tienen el mismo resultado
    MPI_Barrier(MPI_COMM_WORLD);
    if (fabs(current_value - mpi_sum_result) < 1e-10) {
      printf("P%02d (%s): ✅ TIENE LA SUMA CORRECTA: %.0f\n", rank,
             cluster_name, current_value);
    }
  }

  // ========== FASE 8: RESUMEN FINAL ==========
  if (rank == 0) {
    printf("\n=== RESUMEN FINAL ===\n");
    printf("Procesos: %d\n", size);
    printf("Clusters: 3\n");
    printf(" - Cluster 1: procesos 0-%d\n", size / 3 - 1);
    printf(" - Cluster 2: procesos %d-%d\n", size / 3, 2 * (size / 3) - 1);
    printf(" - Cluster 3: procesos %d-%d\n", 2 * (size / 3), size - 1);
    printf("Butterfly conecta procesos entre clusters naturalmente\n");
    printf("Cada proceso tiene acceso al resultado final\n");
  }

  MPI_Finalize();
  return 0;
}
