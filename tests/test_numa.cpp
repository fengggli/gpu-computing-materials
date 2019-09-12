// code is from:
//  https://software.intel.com/en-us/mkl-linux-developer-guide-managing-multi-core-performance       
#define _GNU_SOURCE //for using the GNU CPU affinity
// (works with the appropriate kernel and glibc)
// Set affinity mask
#include <sched.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>
int main(void) {
        int NCPUs = sysconf(_SC_NPROCESSORS_CONF);
        printf("Using thread affinity on %i NCPUs\n", NCPUs);
#pragma omp parallel default(shared)
        {
                cpu_set_t new_mask;
                cpu_set_t was_mask;
                int tid = omp_get_thread_num();
                
                CPU_ZERO(&new_mask);
                
                // 2 packages x 2 cores/pkg x 1 threads/core (4 total cores)
                CPU_SET(tid==0 ? 0 : 2, &new_mask);
                
                if (sched_getaffinity(0, sizeof(was_mask), &was_mask) == -1) {
                        printf("Error: sched_getaffinity(%d, sizeof(was_mask), &was_mask)\n", tid);
                }
                if (sched_setaffinity(0, sizeof(new_mask), &new_mask) == -1) {
                        printf("Error: sched_setaffinity(%d, sizeof(new_mask), &new_mask)\n", tid);
                }
                printf("tid=%d new_mask=%08X was_mask=%08X\n", tid,
                                                *(unsigned int*)(&new_mask), *(unsigned int*)(&was_mask));
        }
        // Call Intel MKL FFT function
        return 0;
}
 
        

