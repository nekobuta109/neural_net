#ifndef __ACTIVATES_H__
#define __ACTIVATES_H__
#include "common.h"

class activates
{
public:
    virtual float act(float a)=0;           //activate
    virtual float d_act(float a)=0;         //微分した関数
    //array
    virtual void array_act(int n , float in[] , float out[]){
//         printf("activates::array_Act() in(out)\n" );
        for(int i=0 ;  i < n ; ++i){
            out[i]=act(in[i]);
        }
    };
    virtual void array_d_act(int n , float in[] , float d_da[][_MAX_UNITS]){
        //ソフトマックス関数以外はスカラーをとりますのでこの形です。
        //二次元配列をとるのは、ソフトマックスに合わせたため。
        //ソフトマックス以外の関数は d_da[i][i]をアクセスする。
        //全部の成分をゼロクリア
        for(int i=0;i<n;++i)for(int j=0; j<n ;++j) {d_da[i][j]=0.0;}
        //一次元の配列として、便宜上二次元配列の対角成分だけ値を入れる。
        for(int i=0 ; i<n ; ++i){
            d_da[i][i]=d_act(in[i]);
        }
    };
};
#endif