#ifndef __SIGMOID_H__
#define __SIGMOID_H__

#include "activates.h"
#include <math.h>

class sigmoid : public activates
{
public:
    virtual float act(float a){
        return  1.0f / (1.0f + expf(- a));
    }
    virtual float d_act(float a){
        float z = this->act(a);
        return  z * (1.0f - z);
    }
};
#endif  //