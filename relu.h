#ifndef __RELU_H__
#define __RELU_H__
//#include "net.h"
#include "activates.h"

class ReLU : public activates
{
public:
    virtual float act(float a){
//        printf("relu:%f->" , a);
        if(a < 0.0f) {  return 0.0f; }
 //       printf("%f\n",a);
        return a;
    }
    virtual float d_act(float a){
        if(a < 0.0f) {  return 0.0f;   }
        return  1.0f;
    }
};

#endif  //