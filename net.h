#ifndef __MODEL_H__
#define __MODEL_H__

#include "common.h"
#include "activates.h"
#include "loss.h"

#define     _MAX_LAYERS     128         //128   layers max(最終段も含める)


//activate types
enum{
    AC_STEP,
    AC_RELU,
    AC_SIGMOID,
    AC_SOFTMAX,
} ACTIVATE_TYPES;

//損失関数の一覧です。
enum{
    LOSS_MEAN_SQUARE,
    LOSS_ENTROPY
} LOSS_TYPES;

#define     INPUT_LAYER     (0)
#define     MIDDLE_LAYER    (1)
#define     OUTPUT_LAYER    (L-1)

class net {
protected:
    int L;                                                  //number of hidden layers
    int n_units[_MAX_LAYERS];                              //number of units
    activates* activator[_MAX_LAYERS];                   // activation function
    loss *loss_func;                                             //loss function

    //foward
    float   w[_MAX_LAYERS][_MAX_UNITS][_MAX_UNITS];           //重み付けです。[レイヤ][前段][後段]分のパラメータがあります。
    float   z[_MAX_LAYERS][_MAX_UNITS];                       //アクティベーション後の値です。
    float   a[_MAX_LAYERS][_MAX_UNITS];                       //入力値です。前段のzの総和だと
    float   y[_MAX_UNITS];                                      //このニューラルネットの答えです。
    float   t[_MAX_UNITS];                                      //正解です。

    //back propergation用変数。
    float   dE_dy[_MAX_UNITS];                      //損失関数の微分    格納用
    float   dz_da[_MAX_UNITS][_MAX_UNITS];          //活性化関数の微分  格納用
    float   dE_da[_MAX_LAYERS][_MAX_UNITS];         //後段で算出された dE/da.
    //
    float   dE_dw[_MAX_LAYERS][_MAX_UNITS][_MAX_UNITS];                 //最終的に算出された dE/dw`
    float   dE_dw_total[_MAX_LAYERS][_MAX_UNITS][_MAX_UNITS];           //バッチ処理で
    int     dE_dw_total_n_update;                       //dE_dwの更新回数。
    //コンストラクタですね。
    void    _dump_2D(const char *caption , float array[][_MAX_UNITS]               , int layer);
    void    _dump_3D(const char *caption, float array[][_MAX_UNITS][_MAX_UNITS]    , int layer);
    void    _w_update(float learning_rate);
public:
    net(){;}//一応あとからでも変更できるように
    net(int _n_inputs,int _n_hidden_layers , int _n_hidden_layer_units,int n_outputs , int hidden_layer_activate_type ,int output_layer_activate_type,int _loss_type )
    {
        build(_n_inputs,_n_hidden_layers ,_n_hidden_layer_units,n_outputs ,hidden_layer_activate_type ,output_layer_activate_type,_loss_type );
    }
    void build(int _n_inputs,int _n_hidden_layers , int _n_hidden_layer_units,int n_outputs , int hidden_layer_activate_type ,int output_layer_activate_type,int _loss_type );
    const float *forward(const float input[]);      //順伝播
    void backward(float learning_rate);              //逆伝播
    void reset_w_update_parameter(void);        //
    //外部から損失関数を利用する場合のインターフェイスを用意します。
    float loss(void);       //直近のfowardの結果をもとに計算します。
    //
    int most_active_in_output_layer(void) const
    {
        int i_max=0;float max=0.0;
        for(int i=0 ; i < n_units[OUTPUT_LAYER]+1 ; ++i) {
            if( z[OUTPUT_LAYER][i] > max)   { i_max=i;max=z[OUTPUT_LAYER][i];  }
//printf("Out_L[%d] : %f\n " , i , z[OUTPUT_LAYER][i] );
        }
//printf("max=%d\n" , i_max);
        return i_max;
    }
    //
    int n_output(void)  {   return n_units[OUTPUT_LAYER];   }
    int n_input(void)   {   return n_units[INPUT_LAYER];    }

    //t(回答)をセットします。
    void set_t(int t);      //整数n →  最終段のクラス分けとしてtを作ります、

    //debug dump
    void dump_layer_a(int l)    {_dump_2D( "a" , a , l );   }
    void dump_layer_z(int l)    {_dump_2D( "z" , z , l );   }
    void dump_w(int l)          {_dump_3D( "w" , w , l);    }
};
#endif  //__MODEL_H__