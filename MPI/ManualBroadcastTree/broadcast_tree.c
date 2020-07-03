#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#define SIZE 30

/* Returns Current CPU Time  */
double getWallTime()
{
    struct timeval time;
    if(gettimeofday(&time, NULL)) return 0;
    double wallTime = (double)time.tv_sec + (double)time.tv_usec * 0.000001;
    return wallTime;
}

int main(int argc, char* argv[])
{
    int myrank, nproc;
    char test[SIZE];
    double startTime, endTime;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // Broadcasting Manually
    if(myrank == 0) {
        strcpy(test, "TEST_MESSAGE_TREE");
        startTime = getWallTime();
    }

    int iterations = (int)ceil(log2(nproc));

    for(int i = 0; i < iterations; i++)
    {
        for(int source = 0; source < (1<<i); source++)
        {
            int destination = source + (1<<(i));
            if (source < nproc && destination < nproc) {
                if(myrank == source) {
                    MPI_Send(test, SIZE, MPI_BYTE, destination, 0, MPI_COMM_WORLD);
                }
                else if(myrank == destination) {
                    MPI_Recv(test, SIZE, MPI_BYTE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == 0) {
        endTime = getWallTime();
        printf("Time Elapsed Manual Tree: %lf\n", endTime-startTime);
    }

    // Broadcasting Automatically
    if (myrank == 0) {
        strcpy(test, "TEST_MESSAGE_BCAST");
        startTime = getWallTime();
    }

    MPI_Bcast(test, SIZE, MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == 0) {
        endTime = getWallTime();
        printf("Time Elapsed BroadCast: %lf\n", endTime-startTime);
    }

    MPI_Finalize();

    return 0;
}