#include "mpi.h"
#include "functions.h"
#include "helpers.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define min(a,b) ((a < b) ? a : b)

void noErrorSituation(Topology *initialTopology, OperationStruct *operationStruct,
                      int rank, int *myCoordinator, int numberOfTasks, char **argv) {

    /* If the task is a coordinator, then we read the input files, we send our topology
    between coordinators and then we send the final topology to our workers. If the task
    is a worker we just wait to receive the topology from our coordinators */
    if (rank == 0 || rank == 1 || rank == 2 || rank == 3) {
        readInputFiles(initialTopology, rank);
        sendTopologyInCoordinators(initialTopology, rank);
        sendTopologyToWorkers(initialTopology, rank);
    } else {
        receiveTopologyFromCoordinators(initialTopology, myCoordinator);
    }

    // After everybody got the topology we print it
    showTopology(initialTopology, rank);
    MPI_Barrier(MPI_COMM_WORLD);

    /* If the task is a coordinator we send the initial numbers vector between
    us, then we send it to our workers and then get their final answer back.
    Otherwise we get the vector from the coordinators, we modify it and then
    we send it back */
    if (rank == 0 || rank == 1 || rank == 2 || rank == 3) {
        // The task 0 is also responsible for generating the initial vector
        if (rank == 0) {
            operationStruct->numberOfElements = atoi(argv[1]);
            generateVector(operationStruct);
        }

        sendVectorInCoordinators(operationStruct, rank);
        sendVectorToWorkers(initialTopology, operationStruct, rank);
        receiveUpdatedVectorFromWorkers(initialTopology, operationStruct, rank);
    } else {
        receiveUpdateAndSendVectorBack(initialTopology, operationStruct, *myCoordinator, rank, numberOfTasks - 4, 0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /* If we are coordinators, after we got the modified vector from our workers.
    We send the final results back to task 0 */
    if (rank == 0 || rank == 1 || rank == 2 || rank == 3) {
        sendUpdatedVectorInCoordinators(operationStruct, rank);

        // After task 0 gathers the final vector, he prints it on the screen
        if (rank == 0) {
            printFinalResult(operationStruct);
        }
    }
}

void halfErrorSituation(Topology *initialTopology, OperationStruct *operationStruct,
                        int rank, int *myCoordinator, int numberOfTasks, char **argv) {

    /* If the task is a coordinator, then we read the input files, we send our topology
    between coordinators and then we send the final topology to our workers. If the task
    is a worker we just wait to receive the topology from our coordinators */
    if (rank == 0 || rank == 1 || rank == 2 || rank == 3) {
        readInputFiles(initialTopology, rank);
        sendTopologyInCoordinatorsOneWay(initialTopology, rank);
        sendTopologyInCoordinatorsReverseWay(initialTopology, rank);
        sendTopologyToWorkers(initialTopology, rank);
    } else {
        receiveTopologyFromCoordinators(initialTopology, myCoordinator);
    }

    // After everybody got the topology we print it
    showTopology(initialTopology, rank);
    MPI_Barrier(MPI_COMM_WORLD);

    /* If the task is a coordinator we send the initial numbers vector between
    us, then we send it to our workers and then get their final answer back.
    Otherwise we get the vector from the coordinators, we modify it and then
    we send it back */
    if (rank == 0 || rank == 1 || rank == 2 || rank == 3) {
        if (rank == 0) {
            operationStruct->numberOfElements = atoi(argv[1]);
            generateVector(operationStruct);
        }

        sendVectorInCoordinatorsHalfError(operationStruct, rank);
        sendVectorToWorkers(initialTopology, operationStruct, rank);
        receiveUpdatedVectorFromWorkers(initialTopology, operationStruct, rank);
    } else {
        receiveUpdateAndSendVectorBack(initialTopology, operationStruct, *myCoordinator, rank, numberOfTasks - 4, 0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /* If we are coordinators, after we got the modified vector from our workers.
    We send the final results back to task 0 */
    if (rank == 0 || rank == 1 || rank == 2 || rank == 3) {
        sendUpdatedVectorInCoordinatorsHalfError(operationStruct, rank);

        // After task 0 gathers the final vector, he prints it on the screen
        if (rank == 0) {
            printFinalResult(operationStruct);
        }
    }
}

void fullErrorSituation(Topology *initialTopology, OperationStruct *operationStruct,
                        int rank, int *myCoordinator, int numberOfTasks, char **argv) {
    
    /* If the task is a coordinator, then we read the input files. If we are not task 1, 
    we send our topology between coordinators. We send the final topology to our workers. 
    If the task is a worker we just wait to receive the topology from our coordinators */
    if (rank == 0 || rank == 1 || rank == 2 || rank == 3) {
        readInputFiles(initialTopology, rank);
        if (rank == 0 || rank == 2 || rank == 3) {
            sendTopologyInCoordOneWayFullError(initialTopology, rank);
            sendTopologyInCoordReverseWayFullError(initialTopology, rank);
        }
        sendTopologyToWorkers(initialTopology, rank);
    } else {
        receiveTopologyFromCoordinators(initialTopology, myCoordinator);
    }

    // After everybody got the topology we print it
    showTopology(initialTopology, rank);
    MPI_Barrier(MPI_COMM_WORLD);

    /* If the task is a coordinator but different from 1 we send the initial
    numbers vector between us, then we send it to our workers and then get their
    final answer back. Otherwise if we are not task 1 or a worker of task 1, 
    we get the vector from the coordinators, we modify it and then we send it back */
    if (rank == 0 || rank == 2 || rank == 3) {
        if (rank == 0) {
            operationStruct->numberOfElements = atoi(argv[1]);
            generateVector(operationStruct);
        }

        sendVectorInCoordinatorsFullError(operationStruct, rank);
        sendVectorToWorkers(initialTopology, operationStruct, rank);
        receiveUpdatedVectorFromWorkers(initialTopology, operationStruct, rank);
    } else if (rank != 1) {
        int childOfProcessOne = 1;
        int totalNumberOfAccessibleTasks = 0;
        for (int i = 0; i < 4; i++) {
            if (i != 1) {
                totalNumberOfAccessibleTasks += initialTopology[i].numberOfWorkerTasks;
                for (int j = 0; j < initialTopology[i].numberOfWorkerTasks; j++) {
                    if (rank == initialTopology[i].workerTasks[j]) {
                        childOfProcessOne = -1;
                        break;
                    }
                }
            }
        }

        int taskOneWorkersFullError = numberOfTasks - totalNumberOfAccessibleTasks - 4; 
        if (childOfProcessOne != 1) {
            receiveUpdateAndSendVectorBack(initialTopology, operationStruct, *myCoordinator,
                                           rank, numberOfTasks - 4, taskOneWorkersFullError);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* If we are coordinators, and different from task 1 after we got the modified
    vector from our workers. We send the final results back to task 0 */
    if (rank == 0 || rank == 2 || rank == 3) {
        sendUpdatedVectorInCoordinatorsFullError(operationStruct, rank);

        // After task 0 gathers the final vector, he prints it on the screen
        if (rank == 0) {
            printFinalResult(operationStruct);
        }
    }
}

void sendTopologyInCoordinators(Topology *initialTopology, int rank) {
    MPI_Status status;

    /* Each of the tasks becomes the initiator of the ring communication 
    sending his own part of the topology */
    for (int i = 0; i < 4; i++) {
        int previousTask = rank - 1;
        if (previousTask < 0) {
            previousTask = 3;
        }

        int nextTask = rank + 1;
        if (nextTask > 3) {
            nextTask = 0;
        }

        /* If a task is initiator it sends his topology and waits for the information to come back.
        Otherwise we firstly receive the information, update what we already had and then send it to
        the next task */
        if (rank == i) {
            sendAVectorFromSoruceToDestination(rank, nextTask, initialTopology[rank].numberOfWorkerTasks, initialTopology[rank].workerTasks);

            MPI_Recv(&initialTopology[rank].numberOfWorkerTasks,
                     1, MPI_INT, previousTask, 0, MPI_COMM_WORLD, &status);

            MPI_Recv(initialTopology[rank].workerTasks, initialTopology[rank].numberOfWorkerTasks,
                     MPI_INT, previousTask, 0, MPI_COMM_WORLD, &status);

        } else {        
            receiveAVectorToSourceFromDestination(initialTopology, previousTask, i);
            sendAVectorFromSoruceToDestination(rank, nextTask, initialTopology[i].numberOfWorkerTasks, initialTopology[i].workerTasks);
        }
    }
}

void sendTopologyInCoordinatorsOneWay(Topology *initialTopology, int rank) {
    /* We have three steps in order to send all the topologies from one end
    to the other in one direction */
    for (int i = 1; i < 4; i++) {
        if (i == 1) {
            // At first step, task 2 gets to know 1's topology
            if (rank == 1) {
                sendAVectorFromSoruceToDestination(rank, 2, initialTopology[rank].numberOfWorkerTasks, initialTopology[rank].workerTasks);
            } else if (rank == 2) {
                receiveAVectorToSourceFromDestination(initialTopology, 1, 1);
            }
        } else if (i == 2) {
            // At second step, task 3 gets to know 1's and 2's topology
            if (rank == 2) {
                for (int j = 1; j <= i; j++) {
                    sendAVectorFromSoruceToDestination(rank, 3, initialTopology[j].numberOfWorkerTasks, initialTopology[j].workerTasks);
                }
            } else if (rank == 3) {
                for (int j = 1; j <= i; j++) {
                    receiveAVectorToSourceFromDestination(initialTopology, 2, j);
                }
            }
        } else {
            // At the third step, task 0 gets to know 1's, 2's and 3's topology
            if (rank == 3) {
                for (int j = 1; j <= i; j++) {
                    sendAVectorFromSoruceToDestination(rank, 0, initialTopology[j].numberOfWorkerTasks, initialTopology[j].workerTasks);
                }
            } else if (rank == 0) {
                for (int j = 1; j <= i; j++) {
                    receiveAVectorToSourceFromDestination(initialTopology, 3, j);
                }
            }
        }
    }
}

void sendTopologyInCoordinatorsReverseWay(Topology *initialTopology, int rank) {
    /* We have three steps in order to send all the topologies from one end
    to the other in one direction */
    for (int i = 0; i < 3; i++) {
        if (i == 0) {
            /* At first step, task 3 gets to know 0's topology. He already knew the others
            from the one way send topology */
            if (rank == 0) {
                sendAVectorFromSoruceToDestination(rank, 3, initialTopology[rank].numberOfWorkerTasks, initialTopology[rank].workerTasks);
            } else if (rank == 3) {
                receiveAVectorToSourceFromDestination(initialTopology, 0, 0);
            }
        } else if (i == 1) {
            /* At second step, task 2 gets to know 0's and 3's topology. He already knew the others
            from the one way send topology */
            if (rank == 3) {
                sendAVectorFromSoruceToDestination(rank, 2, initialTopology[0].numberOfWorkerTasks, initialTopology[0].workerTasks);
                sendAVectorFromSoruceToDestination(rank, 2, initialTopology[rank].numberOfWorkerTasks, initialTopology[rank].workerTasks);
            } else if (rank == 2) {
                receiveAVectorToSourceFromDestination(initialTopology, 3, 0);
                receiveAVectorToSourceFromDestination(initialTopology, 3, 3);
            }
        } else {
            /* At third step, task 1 gets to know 0's, 2's and 3's topology. */
            if (rank == 2) {
                sendAVectorFromSoruceToDestination(rank, 1, initialTopology[0].numberOfWorkerTasks, initialTopology[0].workerTasks);
                sendAVectorFromSoruceToDestination(rank, 1, initialTopology[2].numberOfWorkerTasks, initialTopology[2].workerTasks);
                sendAVectorFromSoruceToDestination(rank, 1, initialTopology[3].numberOfWorkerTasks, initialTopology[3].workerTasks);
            } else if (rank == 1) {
                receiveAVectorToSourceFromDestination(initialTopology, 2, 0);
                receiveAVectorToSourceFromDestination(initialTopology, 2, 2);
                receiveAVectorToSourceFromDestination(initialTopology, 2, 3);
            }
        }
    }
}

void sendTopologyInCoordOneWayFullError(Topology *initialTopology, int rank) {
    /* We have two steps in order to send all the topologies from one end
    to the other in one direction */
    for (int i = 1; i < 3; i++) {
        if (i == 1) {
            // At the first step, task 3 finds out about 2's topology
            if (rank == 2) {
                sendAVectorFromSoruceToDestination(rank, 3, initialTopology[rank].numberOfWorkerTasks, initialTopology[rank].workerTasks);
            } else if (rank == 3) {
                receiveAVectorToSourceFromDestination(initialTopology, 2, 2);
            }
        } else {
            // At the second step, task 0 finds out about 2's and 3's topology
            if (rank == 3) {
                for (int j = 2; j <= i + 1; j++) {
                    sendAVectorFromSoruceToDestination(rank, 0, initialTopology[j].numberOfWorkerTasks, initialTopology[j].workerTasks);
                }
            } else if (rank == 0) {
                for (int j = 2; j <= i + 1; j++) {
                    receiveAVectorToSourceFromDestination(initialTopology, 3, j);
                }
            }
        }
    }
}

void sendTopologyInCoordReverseWayFullError(Topology *initialTopology, int rank) {
    /* We have two steps in order to send all the topologies from one end
    to the other in one direction */
    for (int i = 1; i < 3; i++) {
        if (i == 1) {
            // At the first step, task 3 finds out about 0's topology. He already knew 2's topology
            if (rank == 0) {
                sendAVectorFromSoruceToDestination(rank, 3, initialTopology[rank].numberOfWorkerTasks, initialTopology[rank].workerTasks);
            } else if (rank == 3) {
                receiveAVectorToSourceFromDestination(initialTopology, 0, 0);
            }
        } else {
            // At the second step, task 2 finds out about 0's and 3's topology.
            if (rank == 3) {
                sendAVectorFromSoruceToDestination(rank, 2, initialTopology[0].numberOfWorkerTasks, initialTopology[0].workerTasks);
                sendAVectorFromSoruceToDestination(rank, 2, initialTopology[rank].numberOfWorkerTasks, initialTopology[rank].workerTasks);
            } else if (rank == 2) {
                receiveAVectorToSourceFromDestination(initialTopology, 3, 0);
                receiveAVectorToSourceFromDestination(initialTopology, 3, 3);
            }
        }
    }
}

void sendTopologyToWorkers(Topology *initialTopology, int rank) {
    /* For each of our workers, we send them firstly that we are his coordinators
    because they don't know initally who is their coordinator. And then we send
    them the full topology of the system (the coordinator of a group, the number
    of workers in that group and the workers in the group)*/
    for (int i = 0; i < initialTopology[rank].numberOfWorkerTasks; i++) {
        MPI_Send(&initialTopology[rank].coordinatorTask, 1, MPI_INT, initialTopology[rank].workerTasks[i], 0, MPI_COMM_WORLD);
        printLogMessage(rank, initialTopology[rank].workerTasks[i]);

        for (int j = 0; j < 4; j++) {
            MPI_Send(&initialTopology[j].coordinatorTask, 1, MPI_INT, initialTopology[rank].workerTasks[i], 0, MPI_COMM_WORLD);
            printLogMessage(rank, initialTopology[rank].workerTasks[i]);

            MPI_Send(&initialTopology[j].numberOfWorkerTasks, 1, MPI_INT, initialTopology[rank].workerTasks[i], 0, MPI_COMM_WORLD);
            printLogMessage(rank, initialTopology[rank].workerTasks[i]);

            MPI_Send(initialTopology[j].workerTasks, initialTopology[j].numberOfWorkerTasks, MPI_INT, initialTopology[rank].workerTasks[i], 0, MPI_COMM_WORLD);
            printLogMessage(rank, initialTopology[rank].workerTasks[i]);
        }
    }
}

void receiveTopologyFromCoordinators(Topology *initialTopology, int *myCoordinator) {
    MPI_Status status;
    int coordinator;

    // We firstly get to know who is our coordinator to know where to send the messages back
    MPI_Recv(&coordinator, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    *myCoordinator = coordinator;

    /* We receive for each of the clusters, the coordinator of that cluster, the number of
    workers in the cluster and the workers */
    for (int i = 0; i < 4; i++) {
        MPI_Recv(&initialTopology[i].coordinatorTask, 1, MPI_INT, coordinator, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&initialTopology[i].numberOfWorkerTasks, 1, MPI_INT, coordinator, 0, MPI_COMM_WORLD, &status);

        initialTopology[i].workerTasks = (int *) malloc(initialTopology[i].numberOfWorkerTasks * sizeof(int));
        if (initialTopology[i].workerTasks == NULL) {
            fprintf(stderr, "%s\n", "Error at memory allocation for worker tasks");
            exit(1);
        }

        MPI_Recv(initialTopology[i].workerTasks, initialTopology[i].numberOfWorkerTasks, MPI_INT, coordinator, 0, MPI_COMM_WORLD, &status);
    }
}

void sendVectorInCoordinators(OperationStruct *operationStruct, int rank) {
    MPI_Status status;

    int previousTask = rank - 1;
    if (previousTask < 0) {
        previousTask = 3;
    }

    int nextTask = rank + 1;
    if (nextTask > 3) {
        nextTask = 0;
    }

    /* If we are task 0, we just generated the vector so we send it to the next task and wait it to
    come back to me which means that it got to all of the coordinators. Otherwise, we just get the
    vector from the previous task and send it to the next task */
    if (rank == 0) {
        sendAVectorFromSoruceToDestination(rank, nextTask, operationStruct->numberOfElements, operationStruct->initialVector);

        MPI_Recv(&operationStruct->numberOfElements, 1, MPI_INT, previousTask, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(operationStruct->initialVector, operationStruct->numberOfElements,
                 MPI_INT, previousTask, 0, MPI_COMM_WORLD, &status);

    } else {        
        MPI_Recv(&operationStruct->numberOfElements, 1, MPI_INT, previousTask, 0, MPI_COMM_WORLD, &status);

        operationStruct->initialVector = (int *) malloc(operationStruct->numberOfElements * sizeof(int));
        if (operationStruct->initialVector == NULL) {
            fprintf(stderr, "%s\n", "Error at memory allocation for the initial numbers vector");
            exit(1);
        }

        MPI_Recv(operationStruct->initialVector, operationStruct->numberOfElements,
                 MPI_INT, previousTask, 0, MPI_COMM_WORLD, &status);

        sendAVectorFromSoruceToDestination(rank, nextTask, operationStruct->numberOfElements, operationStruct->initialVector);
    }
}

void sendVectorInCoordinatorsHalfError(OperationStruct *operationStruct, int rank) {
    MPI_Status status;

    int previousTask = rank - 1;
    if (previousTask < 0) {
        previousTask = 3;
    }

    int nextTask = rank + 1;
    if (nextTask > 3) {
        nextTask = 0;
    }

    /* If we are task 0 we just send the vector the other way around the coordinator ring and we don't
    wait for the reply because task 1 cannot send it back to task 0. However we know that the vector
    arrived to task 1 because it is waiting for the vector from task 2*/
    if (rank == 0) {
        sendAVectorFromSoruceToDestination(rank, previousTask, operationStruct->numberOfElements, operationStruct->initialVector);
    } else if (rank != 1) {        
        MPI_Recv(&operationStruct->numberOfElements, 1, MPI_INT, nextTask, 0, MPI_COMM_WORLD, &status);

        operationStruct->initialVector = (int *) malloc(operationStruct->numberOfElements * sizeof(int));
        if (operationStruct->initialVector == NULL) {
            fprintf(stderr, "%s\n", "Error at memory allocation for the initial numbers vector");
            exit(1);
        }

        MPI_Recv(operationStruct->initialVector, operationStruct->numberOfElements,
                 MPI_INT, nextTask, 0, MPI_COMM_WORLD, &status);

        sendAVectorFromSoruceToDestination(rank, previousTask, operationStruct->numberOfElements, operationStruct->initialVector);
    } else {
        MPI_Recv(&operationStruct->numberOfElements, 1, MPI_INT, nextTask, 0, MPI_COMM_WORLD, &status);

        operationStruct->initialVector = (int *) malloc(operationStruct->numberOfElements * sizeof(int));
        if (operationStruct->initialVector == NULL) {
            fprintf(stderr, "%s\n", "Error at memory allocation for the initial numbers vector");
            exit(1);
        }

        MPI_Recv(operationStruct->initialVector, operationStruct->numberOfElements,
                 MPI_INT, nextTask, 0, MPI_COMM_WORLD, &status);
    }
}

void sendVectorInCoordinatorsFullError(OperationStruct *operationStruct, int rank) {
    MPI_Status status;

    int previousTask = rank - 1;
    if (previousTask < 0) {
        previousTask = 3;
    }

    int nextTask = rank + 1;
    if (nextTask > 3) {
        nextTask = 0;
    }

    /* If we are task 0, it means that we generated the vector so we just send it forward. Otherwise,
    if we are not the last task in the communication (aka task 2) we receive the vector from the previous
    task and send it to the next task and if we are task 2 we just receive the vector from the previous
    task and we dont send it forward */
    if (rank == 0) {
        sendAVectorFromSoruceToDestination(rank, previousTask, operationStruct->numberOfElements, operationStruct->initialVector);
    } else if (rank != 2) {        
        MPI_Recv(&operationStruct->numberOfElements, 1, MPI_INT, nextTask, 0, MPI_COMM_WORLD, &status);

        operationStruct->initialVector = (int *) malloc(operationStruct->numberOfElements * sizeof(int));
        if (operationStruct->initialVector == NULL) {
            fprintf(stderr, "%s\n", "Error at memory allocation for the initial numbers vector");
            exit(1);
        }

        MPI_Recv(operationStruct->initialVector, operationStruct->numberOfElements,
                 MPI_INT, nextTask, 0, MPI_COMM_WORLD, &status);

        sendAVectorFromSoruceToDestination(rank, previousTask, operationStruct->numberOfElements, operationStruct->initialVector);
    } else {
        MPI_Recv(&operationStruct->numberOfElements, 1, MPI_INT, nextTask, 0, MPI_COMM_WORLD, &status);

        operationStruct->initialVector = (int *) malloc(operationStruct->numberOfElements * sizeof(int));
        if (operationStruct->initialVector == NULL) {
            fprintf(stderr, "%s\n", "Error at memory allocation for the initial numbers vector");
            exit(1);
        }

        MPI_Recv(operationStruct->initialVector, operationStruct->numberOfElements,
                 MPI_INT, nextTask, 0, MPI_COMM_WORLD, &status);
    }
}

void sendVectorToWorkers(Topology *initialTopology, OperationStruct *operationStruct, int rank) {
    for (int i = 0; i < initialTopology[rank].numberOfWorkerTasks; i++) {
        sendAVectorFromSoruceToDestination(rank, initialTopology[rank].workerTasks[i], operationStruct->numberOfElements, operationStruct->initialVector);
    }
}

void receiveUpdateAndSendVectorBack(Topology *initialTopology, OperationStruct *operationStruct,
                                    int myCoordinator, int rank, int numberOfWorkers,
                                    int taskOneWorkersFullError) {
    MPI_Status status;
    
    MPI_Recv(&operationStruct->numberOfElements, 1, MPI_INT, myCoordinator, 0, MPI_COMM_WORLD, &status);

    operationStruct->numbersVector = (int *) malloc(operationStruct->numberOfElements * sizeof(int));
    if (operationStruct->numbersVector == NULL) {
        fprintf(stderr, "%s\n", "Error at memory allocation for the numbers vector");
        exit(1);
    }

    // We receive the full vector from the coordinator
    MPI_Recv(operationStruct->numbersVector, operationStruct->numberOfElements,
             MPI_INT, myCoordinator, 0, MPI_COMM_WORLD, &status);

    int numberOfPossibleTasks = 0;
    int startPosition, endPosition;

    /* If there is a full error on the communication system it means that we cannot use the workers
    of task's one for multipling so we will just consider the others. Otherwise we just set the positions
    between which we have to make the multipling with the standard formula from parallel programming */
    if (taskOneWorkersFullError != 0) {
        int *workersVector = (int *) malloc(10000 * sizeof(int));
        for (int i = 0; i < 4; i++) {
            if (i != 1) {
                for (int j = 0; j < initialTopology[i].numberOfWorkerTasks; j++) {
                    workersVector[numberOfPossibleTasks++] = initialTopology[i].workerTasks[j];
                }
            }
        }

        for (int i = 0; i < numberOfPossibleTasks; i++) {
            if (workersVector[i] == rank) {
                startPosition = i * (double) operationStruct->numberOfElements / numberOfPossibleTasks;
                endPosition = min((i + 1) * (double) operationStruct->numberOfElements / numberOfPossibleTasks,
                                  operationStruct->numberOfElements);
                break;
            }
        }

        free(workersVector);
    } else {
        startPosition = (rank - 4) * (double) operationStruct->numberOfElements / numberOfWorkers;
        endPosition = min((rank - 4 + 1) * (double) operationStruct->numberOfElements / numberOfWorkers,
                          operationStruct->numberOfElements);
    }

    /* After we update the vector between our positions, we send it back to our coordinator */
    for (int i = startPosition; i < endPosition; i++) {
        operationStruct->numbersVector[i] *= 5;
    }

    MPI_Send(operationStruct->numbersVector, operationStruct->numberOfElements,
             MPI_INT, myCoordinator, 0, MPI_COMM_WORLD);
    printLogMessage(rank, myCoordinator);
}

void receiveUpdatedVectorFromWorkers(Topology *initialTopology, OperationStruct *operationStruct,
                                     int rank) {
    MPI_Status status;

    int *newVector = (int *) malloc(operationStruct->numberOfElements * sizeof(int));
    if (newVector == NULL) {
        fprintf(stderr, "%s\n", "Error at memory alloc for numbers vector");
        exit(1);
    }

    operationStruct->numbersVector = (int *) malloc(operationStruct->numberOfElements * sizeof(int));
    if (operationStruct->numbersVector == NULL) {
        fprintf(stderr, "%s\n", "Error at memory alloc for numbers vector");
        exit(1);
    }

    for (int i = 0; i < operationStruct->numberOfElements; i++) {
        operationStruct->numbersVector[i] = operationStruct->initialVector[i];
    }

    /* We receive the updated vector from each of our workers and if in that vector
    there are any elements different than the same element in the original vector
    it means that it has been updated by the worker so we have to update it in our
    final vector */
    for (int i = 0; i < initialTopology[rank].numberOfWorkerTasks; i++) {
        MPI_Recv(newVector, operationStruct->numberOfElements,
                 MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

        for (int j = 0; j < operationStruct->numberOfElements; j++) {
            if (newVector[j] != operationStruct->initialVector[j]) {
                operationStruct->numbersVector[j] = newVector[j];
            }
        }
    }

    free(newVector);
}

void sendUpdatedVectorInCoordinators(OperationStruct *operationStruct, int rank) {
    MPI_Status status;

    int previousTask = rank - 1;
    if (previousTask < 0) {
        previousTask = 3;
    }

    int nextTask = rank + 1;
    if (nextTask > 3) {
        nextTask = 0;
    }

    /* If we are task 0, we just send the final vector to the next task and wait it to
    come back updated to me which means that it got to all of the coordinators. 
    Otherwise, we just get the vector from the previous task compare and update our vector
    and then send it to the next task */
    if (rank == 0) {
        MPI_Send(operationStruct->numbersVector, operationStruct->numberOfElements,
                 MPI_INT, nextTask, 0, MPI_COMM_WORLD);
        printLogMessage(rank, nextTask);

        MPI_Recv(operationStruct->numbersVector, operationStruct->numberOfElements,
                 MPI_INT, previousTask, 0, MPI_COMM_WORLD, &status);

    } else {       
        int *recvVect = (int *) malloc(operationStruct->numberOfElements * sizeof(int));
        if (recvVect == NULL) {
            fprintf(stderr, "%s\n", "Error at memory allocation for the numbers vector");
            exit(1);
        }

        MPI_Recv(recvVect, operationStruct->numberOfElements,
                 MPI_INT, previousTask, 0, MPI_COMM_WORLD, &status);
        
        for (int i = 0; i < operationStruct->numberOfElements; i++) {
            if (recvVect[i] != operationStruct->initialVector[i]) {
                operationStruct->numbersVector[i] = recvVect[i];
            }
        } 

        MPI_Send(operationStruct->numbersVector, operationStruct->numberOfElements,
                 MPI_INT, nextTask, 0, MPI_COMM_WORLD);
        printLogMessage(rank, nextTask);

        free(recvVect);
    }
}

void sendUpdatedVectorInCoordinatorsHalfError(OperationStruct *operationStruct, int rank) {
    MPI_Status status;

    int previousTask = rank - 1;
    if (previousTask < 0) {
        previousTask = 3;
    }

    int nextTask = rank + 1;
    if (nextTask > 3) {
        nextTask = 0;
    }

    /* If we are task 1 we just send our vector to the next task. If we are not task 0 (aka the
    last task in the ring) we get the vector from the previous task, we update our own vector
    and then send the updated vector to the next task. If we are task 0, we just receive the
    final vector from the previous task */
    if (rank == 1) {
        MPI_Send(operationStruct->numbersVector, operationStruct->numberOfElements,
                 MPI_INT, nextTask, 0, MPI_COMM_WORLD);
        printLogMessage(rank, nextTask);
    } else if (rank != 0) {       
        int *recvVect = (int *) malloc(operationStruct->numberOfElements * sizeof(int));
        if (recvVect == NULL) {
            fprintf(stderr, "%s\n", "Error at memory allocation for the numbers vector");
            exit(1);
        }

        MPI_Recv(recvVect, operationStruct->numberOfElements,
                 MPI_INT, previousTask, 0, MPI_COMM_WORLD, &status);
        
        for (int i = 0; i < operationStruct->numberOfElements; i++) {
            if (recvVect[i] != operationStruct->initialVector[i]) {
                operationStruct->numbersVector[i] = recvVect[i];
            }
        } 

        MPI_Send(operationStruct->numbersVector, operationStruct->numberOfElements,
                 MPI_INT, nextTask, 0, MPI_COMM_WORLD);
        printLogMessage(rank, nextTask);

        free(recvVect);
    } else {
        int *recvVect = (int *) malloc(operationStruct->numberOfElements * sizeof(int));
        if (recvVect == NULL) {
            fprintf(stderr, "%s\n", "Error at memory allocation for the numbers vector");
            exit(1);
        }

        MPI_Recv(recvVect, operationStruct->numberOfElements,
                 MPI_INT, previousTask, 0, MPI_COMM_WORLD, &status);
        
        for (int i = 0; i < operationStruct->numberOfElements; i++) {
            if (recvVect[i] != operationStruct->initialVector[i]) {
                operationStruct->numbersVector[i] = recvVect[i];
            }
        }
    }
}

void sendUpdatedVectorInCoordinatorsFullError(OperationStruct *operationStruct, int rank) {
    MPI_Status status;

    int previousTask = rank - 1;
    if (previousTask < 0) {
        previousTask = 3;
    }

    int nextTask = rank + 1;
    if (nextTask > 3) {
        nextTask = 0;
    }

    /* If we are task 2, it means that we are the initiator of the communication because at the end, the vector
    has to arrive at the task 0. Otherwise, if we are not the last task in the communication (aka task 0) we receive
    the vector from the previous task update our own and send it to the next task. If we are task 0 we just receive
    the vector from the previous task and we dont send it forward */
    if (rank == 2) {
        MPI_Send(operationStruct->numbersVector, operationStruct->numberOfElements,
                 MPI_INT, nextTask, 0, MPI_COMM_WORLD);
        printLogMessage(rank, nextTask);
    } else if (rank != 0) {       
        int *recvVect = (int *) malloc(operationStruct->numberOfElements * sizeof(int));
        if (recvVect == NULL) {
            fprintf(stderr, "%s\n", "Error at memory allocation for the numbers vector");
            exit(1);
        }

        MPI_Recv(recvVect, operationStruct->numberOfElements,
                 MPI_INT, previousTask, 0, MPI_COMM_WORLD, &status);
        
        for (int i = 0; i < operationStruct->numberOfElements; i++) {
            if (recvVect[i] != operationStruct->initialVector[i]) {
                operationStruct->numbersVector[i] = recvVect[i];
            }
        } 

        MPI_Send(operationStruct->numbersVector, operationStruct->numberOfElements,
                 MPI_INT, nextTask, 0, MPI_COMM_WORLD);
        printLogMessage(rank, nextTask);

        free(recvVect);
    } else {
        int *recvVect = (int *) malloc(operationStruct->numberOfElements * sizeof(int));
        if (recvVect == NULL) {
            fprintf(stderr, "%s\n", "Error at memory allocation for the numbers vector");
            exit(1);
        }

        MPI_Recv(recvVect, operationStruct->numberOfElements,
                 MPI_INT, previousTask, 0, MPI_COMM_WORLD, &status);
        
        for (int i = 0; i < operationStruct->numberOfElements; i++) {
            if (recvVect[i] != operationStruct->initialVector[i]) {
                operationStruct->numbersVector[i] = recvVect[i];
            }
        }
    }
}