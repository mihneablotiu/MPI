#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include "structures.h"
#include "functions.h"
#include "helpers.h"

int main(int argc, char **argv) {
    // The number of total tasks in the MPI_COMM_WORLD
    int numberOfTasks, rank;

    // The coordinator (used only by the worker tasks)
    int myCoordinator;

    /* Each task is going to have a final topology which is represented 
    by a vector of 4 topologies, one for each coordinator */
    Topology *initialTopology = (Topology *) malloc(4 * sizeof(Topology));
    if (initialTopology == NULL) {
        fprintf(stderr, "%s\n", "Error at memory allocation for topology");
        exit(1);
    }

    /* Each task is going to have a operation struct used for multipling the
    vector by 5 */
    OperationStruct *operationStruct = (OperationStruct *) malloc(sizeof(OperationStruct));
    if (operationStruct == NULL) {
        fprintf(stderr, "%s\n", "Error at memory allocation for operation struct");
        exit(1);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* We make a different operation if the topology is normal, with a half error (tasks 0 and 1
    cannot communicate directly) or with a full error where task 0 is unacessible */
    if (atoi(argv[2]) == 0) {
        noErrorSituation(initialTopology, operationStruct, rank, &myCoordinator, numberOfTasks, argv);
    } else if (atoi(argv[2]) == 1) {
        halfErrorSituation(initialTopology, operationStruct, rank, &myCoordinator, numberOfTasks, argv);
    } else if (atoi(argv[2]) == 2) {
        fullErrorSituation(initialTopology, operationStruct, rank, &myCoordinator, numberOfTasks, argv); 
    }

    // We free the used memory and end the MPI usage
    freeTopologyMemory(&initialTopology);
    freeNumbersMemory(&operationStruct, rank);
    MPI_Finalize();
    fflush(stdout);
    return 0;
}