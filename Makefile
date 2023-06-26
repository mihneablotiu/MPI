build:
	mpicc -o tema3 tema3.c functions.c helpers.c -Wall -Wextra

clean:
	rm -rf tema3
