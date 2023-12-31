Bloțiu Mihnea-Andrei - 333CA - Algoritmi paraleli și distribuiți - Tema 3
Calcule colaborative in sisteme distribuite - 12.01.2023

The overall point of this homework was to implement a distributed program
in MPI that does an operation (multiplying a vector by 5) with a given
topology made of 4 clusters, each of them having one coordinator and a
random number of workers. The same operation had to be done in two different
situation of errors, one in which two processes could not communicate directly
and the other one in which one cluster was not accesible at all.

This was done accordingly to the indications we had and that being said, the
project was structured as follows:
    - tema3.c the file where the logic of the program happens;
    - functions.c the extra functions specific to MPI communication;
    - functions.h the headers of those functions mentioned above;
    - helpers.c the extra functions that have nothing to do with MPI;
    - helpers.h the headers of those functions mentioned above;
    - structures.h the structures used in this project.
    - Makefile
    - README - the documentation of the project

The whole flow of the project is described in the tema3.c file in the main function
as follows:
    - Firstly, no matter rank of the task, each of those is going to have to know
    the topology and to work with the vector so we are going to allocate those
    structures for all of the tasks;
    - Afterwards, we start the MPI program and depending on the normal, half error
    or full error situation we are going to do the specific operations;
    - After the operations are finished, we dealocate the structures used and end
    the MPI program.

That being said, I will explain now the no error situation approach and afterwards,
I will mark the differences between this approach and the ones I used for the errors.

* noErrorSituation:
    - Firstly at the beginning of the program, if a task is a coordinator one,
    it is going to read it's own input file. That being said, after each
    coordinator read it's input file, he is going to know just the topology
    about his own cluster so the next step is to make the other coordinators
    know about all the clusters;
    - We did that by sending the topology in a ring in four steps. In each of
    the steps, one of the coordinator tasks is going to be the initiator and
    is going to send his own topology around the ring. After the his topology
    is going to come back to him it means that all the coordinators found out
    about his topology and the next coordinator can be the initiator with his
    topology and so on;
    - After all the coordinators know the full topology, it is time for each
    of the coordinators to send the topology to their workers. Firstly they
    send their own id (because at the beginning, the workers don't know who
    is their coordinator and they need this information to communicate back
    to their leader) and afterwards, in 4 steps (because there are 4 clusters),
    the whole topology (the coordinator of the current cluster, the number of
    workers in the current cluster and the workers);
    - If a task is a worker, the first action that is going to do in the MPI
    program is the fact that it is going to wait for the coordinators to share
    the topology and he is going to wait for his coordinator to send him the
    topology;
    - After a coordinator got his topology and sent it to it's workers or
    after a worker got the whole topology, all the tasks are going to print
    it on the screen. After each of the tasks finished printing the topology,
    we can go forward;
    - The next step is for task 0 to generate a vector and then send it to
    all of the coordinators. This is done in the same was as the topology with
    the only difference that now, only task 0 is going to be the initiator
    because he is the only one that knows the vector at the beginning.
    - After each of the coordinators got the vector from task 0, they just
    send the entire vector to each of their workers;
    - The workers are going to receive the entire vector, determine their
    portion of the vector where they have to make changes using the classic
    formula from parallel programming, based on their rank. They will multiply
    the numbers by 5 between their start and end positions and then they are
    going to send the updated vector back to their coordinator;
    - The coordinator is going to wait for the updated vectors to come back.
    They will compare each of the updated vectors to their original vector
    and if any values are different, they will update their original vector.
    At the end, the vector of each coordinator is going to be the result of
    merging all the individual vectors of their own workers;
    - After all the coordinators have their updated vector, they are going to
    send it back to task 0, each of them receiving the updated vector from
    the previous task, updating its own with the new information and sending
    the new vector forward;
    - When the vector arrives back to task 0, it means that each of the
    coordinators put it's own updates in the final vector and task 0 is ready
    to print the final result on the screen;

* halfErrorSituation:
    - The logic of the program is the same as in the no error situation with
    the only difference being the communication between coordinators;
    - For communicating the topology at the beginning of the program, now it
    is going to be done in two steps. In the first step, we will start from 
    task 1 and communicate in the only direction possible 1's topology. Task
    two is going to find out about the topology of task 1 and is going to send
    forward to task 3 its own topology and the topology that found out about
    from task 1 and so on. In the next step, the same procedure is going to
    be done but starting from task 0 and in the reverse way. In this way
    after the two steps, all the tasks are going to know the whole topology.
    - For communicating the initial vector, we are going to do the same as
    before with the difference that task 0 is not going to wait for his own
    vector to come back and task 1 is not going to send the vector back to
    task 0;
    - And, for communicating the final updated vector, the same logic as
    before is applied but now task 1 is going to be the initiator and the
    information is going to be sent in the opposite way because all the
    information has to be gathered to task 0. Afterwards, task 0 is going
    to print the final result;

*fullErrorSituation:
    - The logic of the full error situation is identical with the half error
    situation with the only difference being the communication between
    coordinators and the fact that the workers from task 1 are not included
    for operations;
    - So, when sending the topology between coordinators, just tasks 0, 2 and
    3 are going to participate. Then, each of the tasks are going to send
    just the part of the topology that they know about;
    - When sending the vector between coordinators, task 1 will not participate
    and when the workers update the vector, we have to also update the start
    postion and end position formula for each of them because the workers of
    task 1 cannot participate;
    - Task 1 is not going to take part in the process of sending the updated 
    vector back to task 0.
