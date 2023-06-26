#include "mpi.h"
#include "helpers.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


void sendAVectorFromSoruceToDestination(int sender, int receiver, int numberOfElements, int *vector) {
    // Sending the number of elements from the vector
    MPI_Send(&numberOfElements, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    printLogMessage(sender, receiver);

    // Sending the vector itself
    MPI_Send(vector, numberOfElements, MPI_INT, receiver, 0, MPI_COMM_WORLD);
    printLogMessage(sender, receiver);
}

void receiveAVectorToSourceFromDestination(Topology *initialTopology, int sender, int aboutWho) {
    MPI_Status status;

    // We receive the number of elements in the vector and put them in the specific place of the topology
    MPI_Recv(&initialTopology[aboutWho].numberOfWorkerTasks, 1, MPI_INT, sender, 0, MPI_COMM_WORLD, &status);

    // We set the coordinator as being the task which the message is about
    initialTopology[aboutWho].coordinatorTask = aboutWho;

    // We allocate the memory for that task
    initialTopology[aboutWho].workerTasks = (int *) malloc(initialTopology[aboutWho].numberOfWorkerTasks * sizeof(int));
    if (initialTopology[aboutWho].workerTasks == NULL) {
        fprintf(stderr, "%s\n", "Error at memory allocation for worker tasks");
        exit(1);
    }

    // We receive the whole vector and put it in the specific place of the topology
    MPI_Recv(initialTopology[aboutWho].workerTasks, initialTopology[aboutWho].numberOfWorkerTasks,
             MPI_INT, sender, 0, MPI_COMM_WORLD, &status);
}

void generateVector(OperationStruct *operationStruct) {
    // We allocate memory for the vector
    operationStruct->initialVector = (int *) malloc(operationStruct->numberOfElements * sizeof(int));
    if (operationStruct->initialVector == NULL) {
        fprintf(stderr, "%s\n", "Error at memory allocation for the numbers vector");
        exit(1);
    }

    // We populate the vector with numbers respecting the specific formula
    for (int i = 0; i < operationStruct->numberOfElements; i++) {
        operationStruct->initialVector[i] = operationStruct->numberOfElements - i - 1;
    }
}

void readInputFiles(Topology *initialTopology, int rank) {
    // The coordinator for a leader task is the task itself
    initialTopology[rank].coordinatorTask = rank;

    char nameOfTheInputFile[20];
    sprintf(nameOfTheInputFile, "cluster%d.txt", rank);

    FILE *inputFile = fopen(nameOfTheInputFile, "rt");
    
    // We read the number of worker tasks for the current coordinator
    fscanf(inputFile, "%d", &initialTopology[rank].numberOfWorkerTasks);
    initialTopology[rank].workerTasks = (int *) malloc(initialTopology[rank].numberOfWorkerTasks * sizeof(int));
    if (initialTopology[rank].workerTasks == NULL) {
        fprintf(stderr, "%s\n", "Error at memory allocation for worker tasks");
        exit(1);
    }

    // We read the same number of workers corresponding to the current coordinator task
    for (int i = 0; i < initialTopology[rank].numberOfWorkerTasks; i++) {
        fscanf(inputFile, "%d", &initialTopology[rank].workerTasks[i]);
    }

    fclose(inputFile);
}

void showTopology(Topology *initialTopology, int rank) {
    char *printableTopology = (char *) malloc(100 * sizeof(char));
    if (printableTopology == NULL) {
        fprintf(stderr, "%s\n", "Error at memory allocation for printable topology");
        exit(1);
    }

    char partialBuffer[100];

    // We print our own rank (aka the task that is now printing the topology)
    sprintf(partialBuffer, "%d ->", rank);
    strcpy(printableTopology, partialBuffer);

    /* We print the coordinator number and for each of the coordinators we
    print their workers */
    for (int i = 0; i < 4; i++) {
        if (initialTopology[i].numberOfWorkerTasks != 0) {
            sprintf(partialBuffer, " %d:", i);
            strncat(printableTopology, partialBuffer, strlen(partialBuffer));

            for (int j = 0; j < initialTopology[i].numberOfWorkerTasks; j++) {
                if (j != initialTopology[i].numberOfWorkerTasks - 1) {
                    sprintf(partialBuffer, "%d,", initialTopology[i].workerTasks[j]);
                    strncat(printableTopology, partialBuffer, strlen(partialBuffer));
                } else {
                    sprintf(partialBuffer, "%d", initialTopology[i].workerTasks[j]);
                    strncat(printableTopology, partialBuffer, strlen(partialBuffer));
                }
            }
        }
    }

    puts(printableTopology);
    free(printableTopology);
}

void printFinalResult(OperationStruct *operationStructure) {
    printf("%s", "Rezultat: ");

    for (int i = 0; i < operationStructure->numberOfElements; i++) {
        if (i == operationStructure->numberOfElements - 1) {
            printf("%d\n", operationStructure->numbersVector[i]);
        } else {
            printf("%d ", operationStructure->numbersVector[i]);
        }
    }
}

void printLogMessage(int senderRank, int receiverRank) {
    printf("M(%d,%d)\n", senderRank, receiverRank);
    return;
}

void freeNumbersMemory(OperationStruct **operationStruct, int rank) {
    free((*operationStruct)->numbersVector);

    if (rank == 0 || rank == 1 || rank == 2 || rank == 3) {
        free((*operationStruct)->initialVector);
    }
    
    free(*operationStruct);
    return;
}

void freeTopologyMemory(Topology **initialTopology) {
    for (int i = 0; i < 4; i++) {
        free((*initialTopology)[i].workerTasks);
    }

    free(*initialTopology);
    initialTopology = NULL;
}
