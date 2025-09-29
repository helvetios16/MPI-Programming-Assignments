histogram:
	mpicc -o histogram histogram_mpi.c; mpirun --hostfile mpi_hosts ./histogram; rm histogram

monte_carlo:
	mpicc -o monte_carlo monte_carlo_pi.c; mpirun --hostfile mpi_hosts ./monte_carlo; rm monte_carlo

tree_sum:
	mpicc -o tree_sum tree_sum.c; mpirun --hostfile mpi_hosts ./tree_sum; rm tree_sum

fly:
	mpicc -o fly butterfly_sum.c -lm; mpirun --hostfile mpi_hosts ./fly; rm fly

matrix:
	mpicc -o matrix matrix_vector_block.c -lm; mpirun --hostfile mpi_hosts ./matrix; rm matrix

matrix_vector:
	mpicc -o matrix_vector_block_sub matrix_vector_block_sub.c -lm; mpirun --hostfile mpi_hosts -np 16 ./matrix_vector_block_sub; rm matrix_vector_block_sub

ping:
	mpicc -o ping_pong ping_pong.c -lm; mpirun -np 3 ./ping_pong; rm ping_pong

merge:
	mpicc -o merge parallel_mergesort.c -lm; mpirun -np 16 ./merge; rm merge

cost:
	mpicc -o cost redistribution_cost.c -lm; mpirun -np 16 ./cost; rm cost