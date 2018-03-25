#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <arm_neon.h>
#include <sys/mman.h>
#include <fcntl.h>


#define SIZE 1000000
#define THREAD_SIZE 250000
#define FILE_NAME "input.txt"
struct args {
    unsigned int offset;
};

float * begin;
float source[SIZE] ;
float weight[SIZE] ;
long neon_iteration;

pthread_t pid1, pid2, pid3, pid4;
float32_t part_result[4] = {0.0, 0.0, 0.0, 0.0};

//float32_t mul(float *source,float *weight);
void * mul(void * argument);

void create_input(int size);

static long diff_in_us(struct timespec t1, struct timespec t2)
{
    struct timespec diff;
    if (t2.tv_nsec-t1.tv_nsec < 0) {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec - 1;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec + 1000000000;
    } else {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec;
    }
    return (diff.tv_sec * 1000000.0 + diff.tv_nsec / 1000.0);
}


int main(int argc, char *argv[])
{

    struct timespec start,end;
    //clock_gettime(CLOCK_REALTIME,&start);
    FILE *fptr;
    create_input(SIZE);
    fptr = fopen(FILE_NAME,"rb");
    int fd;
    fd = fileno(fptr);
    struct stat st;
    int r = fstat(fd, &st);
    begin = (float *)mmap(NULL, st.st_size, PROT_READ,MAP_SHARED, fd, 0);
    int rc, rc1, rc2, rc3, rc4;
    struct args args_1;
    args_1.offset = 0;
    struct args args_2;
    args_2.offset = 1;
    struct args args_3;
    args_3.offset = 2;
    struct args args_4;
    args_4.offset = 3;
    clock_gettime(CLOCK_REALTIME,&start);
    rc1 = pthread_create( &pid1, NULL, &mul, (void *) &args_1 );
    rc2 = pthread_create( &pid2, NULL, &mul, (void *) &args_2 );
    rc3 = pthread_create( &pid3, NULL, &mul, (void *) &args_3 );
    rc4 = pthread_create( &pid4, NULL, &mul, (void *) &args_4 );

    /*for(int i = 0; i < SIZE; i++) {
        source[i] = begin[i * 2];
        weight[i] = begin[i * 2 + 1];
    }*/

    /*for(int i = 0; i<SIZE; i++) {
        //fscanf(fptr,"%f%f",&source[i],&weight[i]);
        fread(&source[i], sizeof(source[0]), 1, fptr);
        fread(&weight[i], sizeof(source[0]), 1, fptr);
    }
    fclose(fptr);*/

    //printf("output:  %f\n",mul(source,weight));
    rc = pthread_join(pid1,NULL);
    rc = pthread_join(pid2,NULL);
    rc = pthread_join(pid3,NULL);
    rc = pthread_join(pid4,NULL);
    printf("output: %f\n", part_result[0] + part_result[1] + part_result[2] + part_result[3]);
    fclose(fptr);
    munmap(begin, st.st_size);
    clock_gettime(CLOCK_REALTIME,&end);
    printf("spend:  %ld us\n",diff_in_us(start,end));

    return 0;
}

void * mul(void * argument)
{
    struct args *arg = (struct args *)argument;
    unsigned int offset = arg->offset;
    unsigned int base_addr = offset * THREAD_SIZE;
    for(unsigned int i = 0; i < THREAD_SIZE; i++) {
        source[i + base_addr] = begin[(i + base_addr) * 2];
        weight[i + base_addr] = begin[(i + base_addr) * 2 + 1];
    }

    float32x4_t in1_128,in2_128,sum1,sum2,prod;
    float32_t result[4];
    //float32_t output = 0.0;

    prod = vmovq_n_f32(0.0f);
    for (unsigned int i=0; i < THREAD_SIZE; i+=4) {
        in1_128 = vld1q_f32(&source[i + base_addr]);
        in2_128 = vld1q_f32(&weight[i + base_addr]);
#ifdef FLUSH
        prod = vmulq_f32(in1_128, in2_128);
        sum1 = vaddq_f32(prod, vrev64q_f32(prod));
        sum2 = vaddq_f32(sum1, vcombine_f32(vget_high_f32(sum1), vget_low_f32(sum1)));
        vst1q_f32((float32_t *)result, sum2);
        //output += result[0];
        part_result[offset] += result[0];
#endif
#ifdef NON_FLUSH
        prod = vmlaq_f32(prod, in1_128, in2_128);
#endif
    }
#ifdef NON_FLUSH
    sum1 = vaddq_f32(prod, vrev64q_f32(prod));
    sum2 = vaddq_f32(sum1, vcombine_f32(vget_high_f32(sum1), vget_low_f32(sum1)));
    vst1q_f32((float32_t *)result, sum2);
    part_result[offset] = result[0];
    //output = result[0];
#endif
    pthread_exit(NULL);
    //return output;
}

/*float32_t mul(float *source,float *weights)
{
    float32x4_t in1_128,in2_128,sum1,sum2,prod;
    float32_t result[4];
    float32_t output = 0.0;
    int i;
    prod = vmovq_n_f32(0.0f);
    for (i=0; i<SIZE; i+=4) {
        in1_128 = vld1q_f32(&source[i]);
        in2_128 = vld1q_f32(&weights[i]);
#ifdef FLUSH
        prod = vmulq_f32(in1_128, in2_128);
        sum1 = vaddq_f32(prod, vrev64q_f32(prod));
        sum2 = vaddq_f32(sum1, vcombine_f32(vget_high_f32(sum1), vget_low_f32(sum1)));
        vst1q_f32((float32_t *)result, sum2);
        output += result[0];
#endif
#ifdef NON_FLUSH
        prod = vmlaq_f32(prod, in1_128, in2_128);
#endif
    }
#ifdef NON_FLUSH
    sum1 = vaddq_f32(prod, vrev64q_f32(prod));
    sum2 = vaddq_f32(sum1, vcombine_f32(vget_high_f32(sum1), vget_low_f32(sum1)));
    vst1q_f32((float32_t *)result, sum2);
    output = result[0];
#endif

    return output;
}*/

void create_input(int size)
{
    FILE *fptr;
    if( access( FILE_NAME, F_OK ) != -1 ) {
        return;
    }

    fptr = fopen(FILE_NAME,"wb");
    srand(time(NULL));
    for(int i = 0; i<SIZE; i++) {
        //fprintf(fptr,"%f%f",(float32_t)(rand()%4)+1,(float32_t)(rand()%4)+1);
        float f1, f2;
        f1 =  (float32_t)(rand()%4)+1;
        f2 =  (float32_t)(rand()%4)+1;
        fwrite(&f1, sizeof(f1), 1, fptr);
        fwrite(&f1, sizeof(f2), 1, fptr);
    }
    fclose(fptr);
}

