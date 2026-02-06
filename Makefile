# Makefile for Laplace Equation Solvers
# High Performance Computing - Master's Course

# Compilers
CC = gcc
MPICC = mpicc

# Flags
CFLAGS = -O3 -Wall -Wextra
LDFLAGS = -lm

# Directories
SEQ_DIR = src/sequential
PAR_DIR = src/parallel
BIN_DIR = bin
SOL_DIR = solution

# Sequential targets
SEQ_SOURCES = $(SEQ_DIR)/sor.c
SEQ_TARGETS = $(BIN_DIR)/sor

# Parallel targets
PAR_SOURCES = $(PAR_DIR)/sor_mpi.c
PAR_TARGETS = $(BIN_DIR)/sor_mpi

# Default target
all: dirs sequential parallel

# Create directories
dirs:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(SOL_DIR)

# Sequential programs
sequential: dirs $(SEQ_TARGETS)

$(BIN_DIR)/sor: $(SEQ_DIR)/sor.c $(SEQ_DIR)/laplace_common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Parallel programs (require MPI)
parallel: dirs $(PAR_TARGETS)

$(BIN_DIR)/sor_mpi: $(PAR_DIR)/sor_mpi.c $(PAR_DIR)/laplace_common_mpi.h
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Clean
clean:
	rm -rf $(BIN_DIR)
	rm -rf $(SOL_DIR)/*.dat $(SOL_DIR)/*.png

# Run examples
run-sor: $(BIN_DIR)/sor
	./$(BIN_DIR)/sor 50 1e-6 10000 1
	@mv sor_solution.dat $(SOL_DIR)/ 2>/dev/null || true

run-sor-mpi: $(BIN_DIR)/sor_mpi
	mpirun -np 4 ./$(BIN_DIR)/sor_mpi 100 1e-6 10000
	@mv sor_mpi_solution.dat $(SOL_DIR)/ 2>/dev/null || true

# Run all sequential tests
test-sequential: sequential
	@echo "=== Testing SOR ==="
	./$(BIN_DIR)/sor 30 1e-4 1000 1
	@mv sor_solution.dat $(SOL_DIR)/ 2>/dev/null || true

# Run all parallel tests
test-parallel: parallel
	@echo "=== Testing Parallel SOR Red-Black (4 processes) ==="
	mpirun -np 4 ./$(BIN_DIR)/sor_mpi 50 1e-4 1000
	@mv sor_mpi_solution.dat $(SOL_DIR)/ 2>/dev/null || true

# Visualize solution
visualize: $(SOL_DIR)/visualize.gp
	@cd $(SOL_DIR) && gnuplot visualize.gp
	@echo "Visualization saved to $(SOL_DIR)/heatmap.png and $(SOL_DIR)/contour.png"

# Scalability test
scalability: parallel
	@echo "Scalability test for Parallel SOR (100x100 grid)"
	@echo ""
	@for np in 1 2 4 8; do \
		echo "=== $$np processors ==="; \
		mpirun -np $$np ./$(BIN_DIR)/sor_mpi 100 1e-6 10000; \
		echo ""; \
	done

# Benchmark
benchmark: sequential parallel
	@echo "Benchmark for SOR and MPI SOR"
	@echo ""
	@for i in 50 100 200 400 800; do \
		echo ""; \
		mpirun -np 6 ./$(BIN_DIR)/sor_mpi $$i 1e-6 10000; \
		echo ""; \
		./$(BIN_DIR)/sor $$i 1e-6 10000; \
		echo ""; \
	done


.PHONY: all dirs sequential parallel clean run-sor run-sor-mpi \
        test-sequential test-parallel visualize scalability