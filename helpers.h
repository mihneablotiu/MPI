#ifndef __HELPERS_H_
#define __HELPERS_H_

#include "structures.h"

/* The function that sends a vector from one task to another. Firstly sends the number of
elements in a vector and then the vector itself
    @sender -> the rank of the sender task 
    @receiver -> the rank of the receiver task
    @numberOfElements -> the number of elements in the vector
    @vector -> the start address of the vector */
void sendAVectorFromSoruceToDestination(int sender, int receiver, int numberOfElements, int *vector);

/* The function that receives a topology vector from a sender about a specific task
    @initialTopology -> the current topology of the task that receives the vector
    @sender -> the rank of the sender task 
    @aboutWho -> the rank of the task that the recieved vector is about */
void receiveAVectorToSourceFromDestination(Topology *initialTopology, int sender, int aboutWho);

/* The function that generates the initial numbers vector (called only by task 0)
    @operationStruct -> the struct used when working with the numbers vector */
void generateVector(OperationStruct *operationStruct);

/* The function used by the coordinators tasks to read the input files 
    @initialTopology -> their initial topology in which they read the information
    @rank -> the rank of the task */
void readInputFiles(Topology *initialTopology, int rank);

/* The function used by all the tasks to print the final topology after they all
found out about it
    @initialTopolgy -> their initial topology in which they have all the system topology 
    @rank -> the rank of the current task */
void showTopology(Topology *initialTopology, int rank);

/* The function used just by the task 0 to print the final vector after every worker 
multiplied their part with 5 and sent the result back 
    @operationStructure -> the final structure of the vector */
void printFinalResult(OperationStruct *operationStructure);

/* The function that is used every time we send a message in order to print the log
message between which task was the message sent
    @senderRank -> the rank of the sender
    @receiverRank -> the rank of the receiver */
void printLogMessage(int senderRank, int receiverRank);

/* The function used by all the tasks to free the the vector of numbers 
used for the multipling operation
    @operationStruct -> the freed structure
    @rank -> the rank of the current task that is using free on the structure */
void freeNumbersMemory(OperationStruct **operationStruct, int rank);

/* The function used by all the tasks to free the memory occupied by the
full topology of the system
    @initialTopology -> the vector of topologies that is going to be freed */
void freeTopologyMemory(Topology **initialTopology);

#endif /* __HELPERS_H_ */