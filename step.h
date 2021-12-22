#ifndef __STEP_H__
#define __STEP_H__
#include "activates.h"
class step: public activates
{
public:
    virtual float act(float a){
        if(a < 0.0f) {  return 0.0f; }
        return 1.0f;
    }
    virtual float d_act(float a){
        return  0.0f;
    }
};
#endif  //