#ifndef __COMMONFUNC_H__
#define __COMMONFUNC_H__
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//net.h と activates.h 両方で定義を参照する必要があったので、ここに定義しています。
#define     _MAX_UNITS      128         //128   units each layer max



static void _Assert( bool a, const char*comment ,...){
    //toku toriaezu
    if(!a){
        printf("=== ASSERT [%s] ========> PROGRAM ETREMINATED\n" , comment);
        exit(-1);
    }
}
//static
static float rand_uniform(float a, float b) {
    float x = ((float)rand() + 1.0)/((float)RAND_MAX + 2.0);
    return (b - a) * x + a;
}
static float rand_normal(float mu, float sigma) {
    float z = sqrtf(- 2.0 * logf(rand_uniform(0.0f, 1.0f))) * sinf(2.0 * M_PI * rand_uniform(0.0f, 1.0f));
    return mu + sigma * z;
}

#endif  //