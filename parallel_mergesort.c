#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Función para mezclar dos arrays ordenados
void merge(int* arr1, int size1, int* arr2, int size2, int* result) {
    int i = 0, j = 0, k = 0;
    while (i < size1 && j < size2) {
        if (arr1[i] < arr2[j]) {
            result[k++] = arr1[i++];
        } else {
            result[k++] = arr2[j++];
        }
    }
    while (i < size1) result[k++] = arr1[i++];
    while (j < size2) result[k++] = arr2[j++];
}

// QuickSort simple para ordenar localmente
void quicksort(int* arr, int left, int right) {
    if (left >= right) return;
    
    int pivot = arr[(left + right) / 2];
    int i = left, j = right;
    
    while (i <= j) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i <= j) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            i++;
            j--;
        }
    }
    
    quicksort(arr, left, j);
    quicksort(arr, i, right);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // ========== CONFIGURACIÓN ==========
    int n = 64;  // Total de elementos
    if (argc > 1) n = atoi(argv[1]);
    
    if (n % size != 0) n = (n / size) * size;  // Ajustar para división exacta
    
    int local_size = n / size;
    int* local_data = malloc(local_size * sizeof(int));
    
    // Proceso 0 lee n y distribuye
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // ========== GENERAR Y ORDENAR DATOS LOCALES ==========
    srand(time(NULL) + rank);  // Semilla diferente por proceso
    
    for (int i = 0; i < local_size; i++) {
        local_data[i] = rand() % 1000;  // Números entre 0-999
    }
    
    // Ordenar localmente
    quicksort(local_data, 0, local_size - 1);
    
    // ========== PROCESO 0 MUESTRA LISTAS LOCALES ==========
    if (rank == 0) {
        printf("=== MERGE SORT PARALELO ===\n");
        printf("Total elementos: %d, Procesos: %d, Elementos/proceso: %d\n\n", n, size, local_size);
        
        // Recolectar y mostrar listas locales
        int* all_local = malloc(n * sizeof(int));
        MPI_Gather(local_data, local_size, MPI_INT, all_local, local_size, MPI_INT, 0, MPI_COMM_WORLD);
        
        printf("Listas locales ordenadas:\n");
        for (int proc = 0; proc < size; proc++) {
            printf("Proceso %d: [", proc);
            for (int i = 0; i < local_size; i++) {
                printf("%d", all_local[proc * local_size + i]);
                if (i < local_size - 1) printf(", ");
            }
            printf("]\n");
        }
        printf("\n");
        free(all_local);
    } else {
        MPI_Gather(local_data, local_size, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    // ========== MERGE SORT PARALELO (ÁRBOL BINARIO) ==========
    int current_size = local_size;
    int* current_data = local_data;
    int step = 1;
    
    while (step < size) {
        int partner = rank ^ step;  // Pareja usando XOR
        
        if (partner < size) {
            if (rank < partner) {
                // Recibir datos del partner y mezclar
                int partner_size;
                MPI_Recv(&partner_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                int* partner_data = malloc(partner_size * sizeof(int));
                MPI_Recv(partner_data, partner_size, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Mezclar
                int new_size = current_size + partner_size;
                int* new_data = malloc(new_size * sizeof(int));
                merge(current_data, current_size, partner_data, partner_size, new_data);
                
                // Actualizar
                if (current_data != local_data) free(current_data);
                free(partner_data);
                current_data = new_data;
                current_size = new_size;
                
            } else {
                // Enviar datos al partner y terminar
                MPI_Send(&current_size, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
                MPI_Send(current_data, current_size, MPI_INT, partner, 0, MPI_COMM_WORLD);
                
                if (current_data != local_data) free(current_data);
                current_data = NULL;
                current_size = 0;
                break;
            }
        }
        
        step *= 2;  // Siguiente nivel del árbol
    }
    
    // ========== PROCESO 0 MUESTRA RESULTADO FINAL ==========
    if (rank == 0) {
        printf("Lista final ordenada (%d elementos):\n[", current_size);
        for (int i = 0; i < current_size; i++) {
            printf("%d", current_data[i]);
            if (i < current_size - 1) printf(", ");
            if (i > 0 && i % 20 == 0) printf("\n ");  // Nueva línea cada 20 elementos
        }
        printf("]\n");
        
        // Verificar que está ordenado
        int sorted = 1;
        for (int i = 1; i < current_size; i++) {
            if (current_data[i] < current_data[i-1]) {
                sorted = 0;
                break;
            }
        }
        printf("\n✅ Lista %sordenada correctamente\n", sorted ? "" : "NO ");
    }
    
    // ========== LIMPIEZA ==========
    if (current_data != NULL && current_data != local_data) {
        free(current_data);
    }
    free(local_data);
    
    MPI_Finalize();
    return 0;
}