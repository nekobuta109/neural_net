#ifndef __SOFTMAX_H__
#define __SOFTMAX_H__
#include <stdio.h>
#include <math.h>
#include "activates.h"

class softmax : public activates
{
public:
    virtual float act(float a){
       { printf("softmax::act() is not avalilabe  :: act(float a)(program terminated\n"); exit(-1);  }
    }
    virtual float d_act(float a){
       { printf("softmax is not avalilabe  :: d_act(float a)(program terminated\n"); exit(-1);  }
    }
    virtual void array_act(int n, float in[] , float out[]){
        float sum_exp=0.0;
        //max_a‹‚ß‚Ü‚·B
        float max_a=0.0;
        {for(int i = 0 ; i < n ; ++i ){  if(in[i] > max_a ){max_a = in[i];}  }}
#if 0  //debug
max_a =0.0;

#endif
        for(int i=0; i < n; ++i) {
            sum_exp += ( out[i] = (expf(in[i] - max_a)) );
        }
        for(int i=0 ; i< n ; ++i){
            out[i] /= sum_exp;
        }
    }
    virtual void array_d_act(int n , float in[] , float d_da[][_MAX_UNITS]){        
        float y[n];
        this->array_act(n,in,y);    //`Šˆ«‰»ŠÖ”‚ğÀs‚µ‚Ü‚·B
        //d_da[][]‚Ù‚©‚Ì‚Æˆá‚¤‚©‚½‚¿‚¶‚á
        for(int i=0 ; i < n ; ++i) {
            for(int k = 0; k < i ; ++k ) {
                d_da[i][k] = d_da[k][i] = -1 * y[i] * y[k];
            }
            d_da[i][i] = y[i] * (1.0f - y[i]); 
        }
    }
};
#endif      //