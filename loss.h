#ifndef __LOSS_H__
#define __LOSS_H__
#include <math.h>
#include "common.h"

//
class loss
{
public:
    virtual float E(float y,float t)=0;
    virtual float dE_dy(float y,float t)=0;
    //配列対応：基本クラスでは、単純な足し算をしておきます。
    virtual float array_E(int n , float y[],float t[]){
        float e=0.0f;
        for(int i=0 ; i < n ; ++i){
              e += E( y[i] , t[i] );   
//            printf("loss::array_E[%d] y:%f t:%f e=%f\n" ,i, y[i] , t[i] , e );
        }
        return e;
    };
    virtual void array_dE_dy(int n , float y[],float t[] , float de_dy[] ){
        for(int i=0 ; i < n ;++i ) {    de_dy[i]  = dE_dy( y[i] , t[i] );   }
    };
};

// 二乗誤差
class LOSS_mean_squared_error : public loss
{
public:
    float E(float y, float t)       {  return (t-y)*(t-y);      }
    float dE_dy(float y,float t)    {   return  2.0f * (t-y);   }
    float array_E(int n,float y[],float t[]){
        _Assert(n>0 , "mean_squared_error ::arrayE() n > 0 must");
        return loss::array_E(n,y,t)/n;
    }
    virtual void array_dE_dy(int n , float y[],float t[] , float de_dy[] ){
        _Assert(n>0 , "mean_squared_error ::array_dE_dy() n > 0 must.");
        loss::array_dE_dy(n , y, t , de_dy);
        for(int i=0; i < n ; ++i ){
            de_dy[i]/=n;        //全要素をnで割るらしい。
        }
    }
};

//交差エントロピー
#define EPS (1e-12)
//const float EPS = 1e-12;
class LOSS_cross_entropy : public loss
{
public:
    float E(float y, float t)       {  return  -1 *  t * logf(y + EPS); }
    float dE_dy(float y,float t)    {   return -1 *  (t / (y + EPS));   }
};
#endif