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

# Sequential targets
SEQ_SOURCES = $(SEQ_DIR)/jacobi.c $(SEQ_DIR)/sor.c
SEQ_TARGETS = $(BIN_DIR)/jacobi $(BIN_DIR)/sor

# Parallel targets
PAR_SOURCES = $(PAR_DIR)/jacobi_mpi.c $(PAR_DIR)/sor_mpi.c \
              $(PAR_DIR)/jacobi_omp.c $(PAR_DIR)/sor_omp.c
PAR_TARGETS = $(BIN_DIR)/jacobi_mpi $(BIN_DIR)/sor_mpi \
              $(BIN_DIR)/jacobi_omp $(BIN_DIR)/sor_omp

# Default target
all: dirs sequential parallel

# Create directories
dirs:
	@mkdir -p $(BIN_DIR)

# Sequential programs
sequential: dirs $(SEQ_TARGETS)

$(BIN_DIR)/jacobi: $(SEQ_DIR)/jacobi.c $(SEQ_DIR)/laplace_common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/sor: $(SEQ_DIR)/sor.c $(SEQ_DIR)/laplace_common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Parallel programs (require MPI)
parallel: dirs $(PAR_TARGETS)

$(BIN_DIR)/jacobi_mpi: $(PAR_DIR)/jacobi_mpi.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/sor_mpi: $(PAR_DIR)/sor_mpi.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/jacobi_omp: $(PAR_DIR)/jacobi_omp.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDFLAGS)

$(BIN_DIR)/sor_omp: $(PAR_DIR)/sor_omp.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDFLAGS)	

# Clean
clean:
	rm -rf $(BIN_DIR)
	rm -f *.dat

# Run examples
run-jacobi: $(BIN_DIR)/jacobi
	./$(BIN_DIR)/jacobi 50 1e-6 50000

run-sor: $(BIN_DIR)/sor
	./$(BIN_DIR)/sor 50 1e-6 10000

run-jacobi-mpi: $(BIN_DIR)/jacobi_mpi
	mpirun -np 4 ./$(BIN_DIR)/jacobi_mpi 100 1e-6 50000

run-sor-mpi: $(BIN_DIR)/sor_mpi
	mpirun -np 4 ./$(BIN_DIR)/sor_mpi 100 1e-6 10000

# Run all sequential tests
test-sequential: sequential
	@echo "=== Testing Jacobi ==="
	./$(BIN_DIR)/jacobi 30 1e-4 10000
	@echo ""
	@echo "=== Testing SOR ==="
	./$(BIN_DIR)/sor 30 1e-4 1000

# Run all parallel tests
test-parallel: parallel
	@echo "=== Testing Parallel Jacobi (4 processes) ==="
	mpirun -np 4 ./$(BIN_DIR)/jacobi_mpi 50 1e-4 10000
	@echo ""
	@echo "=== Testing Parallel SOR Red-Black (4 processes) ==="
	mpirun -np 4 ./$(BIN_DIR)/sor_mpi 50 1e-4 1000

# Compare methods
compare: sequential
	@echo "Comparing convergence speed on 50x50 grid with tolerance 1e-6"
	@echo ""
	@echo "=== Jacobi ==="
	@./$(BIN_DIR)/jacobi 50 1e-6 100000
	@echo ""
	@echo "=== SOR (optimal omega) ==="
	@./$(BIN_DIR)/sor 50 1e-6 10000

# Scalability test
scalability: parallel
	@echo "Scalability test for Parallel Jacobi (100x100 grid)"
	@echo ""
	@for np in 1 2 4 8; do \
		echo "=== $$np processors ==="; \
		mpirun -np $$np ./$(BIN_DIR)/jacobi_mpi 100 1e-6 10000; \
		echo ""; \
	done

.PHONY: all dirs sequential parallel clean run-jacobi run-sor \
        run-jacobi-mpi run-sor-mpi test-sequential test-parallel compare scalability
