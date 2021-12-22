#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "common.h"
#include "net.h"
#include "activates.h"
#include "loss.h"

//==================================================================================
//  static functions
//==================================================================================
static activates*       _get_act(int type);
static loss     *       _get_lossfunc(int type);
//==================================================================================
//      ネットワーク構成
//==================================================================================
void net::build(int n_inputs,int n_hidden_layers , int n_hidden_layer_units,int n_outputs , int hidden_layer_activate_type ,int output_layer_activate_type,int loss_type )
{
    L = 1 + n_hidden_layers + 1; //入力・出力段を一緒にします。
                            //入力段        ：[0]
                            //隠れ段階      ：[1-L-2]
                            //出力段階      : [L-1]
                            //として運用してみる。分かりにくければあとで変更。
                            //考え方として同じ配列として入力・中間レイヤ・出力レイヤも扱うという考え。

    _Assert( L < _MAX_LAYERS , "net::build() layer overflow");
    _Assert( n_inputs +1 < _MAX_UNITS , "net::builde() input layer overflow");
    _Assert( n_outputs +1 < _MAX_UNITS , "net::builde() output layer overflow");
    _Assert( n_hidden_layer_units + 1 < _MAX_UNITS , "net::build() hiddenl layase unit overflow");

     //----------------------------------------------------
    //各レイヤのニューロン数を決めます。ここは決め打ちのようです。
    //----------------------------------------------------
    n_units[0] = n_inputs;     //初段
    for(int i = 1; i < L-1; ++i) {  n_units[i] = n_hidden_layer_units;     }  //隠しレイヤ
    n_units[L-1] = n_outputs;
    //----------------------------------------------------
    //  各レイヤの活性化関数の設定です。ここはReLU限定のようです。
    //----------------------------------------------------
    for(int i = 0; i < L-1; ++i) {   activator[i] = _get_act(hidden_layer_activate_type);    }
    activator[L-1] = _get_act(output_layer_activate_type);
    //一応ヌルポインタがないようにしておくか？
    
    loss_func = _get_lossfunc(loss_type);    //損失関数です.

    // w 初期値をセットアップしていきます。
    //w は、便宜上、バイアス成分（足す重み）を、[0]とするようにします。
    //そのため、配列上 [レイヤ][0:bias , 1 ? ユニット数]が対応する数となる。
    //ややこしいけど、そうするようです。
    {
        srand(111);
        for(int l = 0; l < L; ++l )    {
            float sigma = 2.0f / sqrtf((float)n_units[l]);
            for(int i = 0; i < n_units[l-1]+1 ; ++i ) {
                for(int j = 0; j < n_units[l]+1 ; ++j) {
                    w[l][i][j] = rand_normal(0.0f, sigma);
                }
            }
        }
    }
    //z これはアクティベーション後の値を確保するためかしら。初期値いるか?0に初期化です。
    memset( (void*)&z[0][0] , 0 , sizeof(z) );
    //a をセットアップしていきます。
    memset( (void*)&a[0][0] , 0 , sizeof(a) );
     //y をセットアップします。 y は 最終段の答えです。
    memset( (void*)&y[0] , 0 , sizeof(y));    
    memset( (void*)&t[0] , 0 , sizeof(t));   
    //dE/dw 格納用。これはでかいとおもうな。
    memset( (void*)&dE_dw[0][0][0]      , 0 , sizeof(dE_dw) );
    memset( (void*)&dE_dw_total[0][0][0] , 0 , sizeof(dE_dw_total) );
    //dE/da 格納用。
    memset( (void*)&dE_da[0][0]      , 0 , sizeof(dE_da) );
}
//w_updateに関する操作。
void net::reset_w_update_parameter(void)
{
    memset((void*)&dE_dw_total[0][0][0] , 0 , sizeof(dE_dw_total) );
//  float dE_dw_total[_MAX_LAYERS][_MAX_UNITS][_MAX_UNITS];          //dE/dw total? 平均らしい。
    dE_dw_total_n_update = 0 ;
}
//wの更新作業です。total_dE_dw_totalを使用して行います。
void net::_w_update(float learning_rate)
{
    _Assert(dE_dw_total_n_update > 0 , "net::w_update() no back_propagation");
    //======================================================
    //  w[l][i][j]  の更新を行います。
    //======================================================
    for(int l = 0; l < OUTPUT_LAYER ; ++l ) {
        for(int i = 1; i < n_units[l]+1 ; ++i) {
            for(int j = 0; j < n_units[l+1]+1 ; ++j ) {
                float _backup_w = w[l][i][j];   //(debug)
                w[l][i][j] -= learning_rate * dE_dw[l][i][j];
                //wの更新にtotalを使い、バッチサイズで平均をとったもの）として計算するようだ。・・
                w[l][i][j] -= learning_rate * (dE_dw_total[l][i][j] / (float)dE_dw_total_n_update);
            }
        }
    }
}
//直近のfowardの結果をもとに計算します。(配列の損失関数を呼びます。)
float net::loss(void)
{
    _Assert(loss_func!=0 , "net::loss() loss_func is null");
    float E = loss_func->array_E( n_units[OUTPUT_LAYER] + 1 , y, t );
    return E;
}

//debug bdump
void    net::_dump_2D(const char *caption , float array[][_MAX_UNITS] , int layer)
{
    _Assert( layer < L+1 , "");
    printf(" ======[%s] LAYER [%d] / [%d] \n" , caption , layer ,  L );
    for(int i=0 ; i < n_units[layer]+1; ++i ){
        printf("%s[%d][%d]\t=\t%f\n" ,caption , layer , i , array[layer][i] );
    }
}
void    net::_dump_3D(const char *caption, float array[][_MAX_UNITS][_MAX_UNITS]    , int layer)
{
    _Assert( layer < L+1 , "_dump_3D :layer over ");
    _Assert( layer > 0 , "dump3D : layer muse > 0");
    printf(" ~~~~~~[dump3D] [%s] LAYER [%d] / [%d] ~~~~~~~~~~\n" , caption , layer ,  L);
    for(int i= 0 ; i < n_units[layer-1]+1 ; ++i){
        for(int j=0 ;  j < n_units[layer]+1 ; ++j ){
            printf("%s[%d][%d][%d] : %f\n" ,caption, layer , i , j , array[layer][i][j] );
        }
    }
}
void net::set_t(const int _t )
{
    _Assert( _t < n_units[OUTPUT_LAYER] + 1 , "set_t  overflow" );   //+! は１オリジンでも大丈夫なように
    memset( (void*)t , 0 ,sizeof(t) );
    t[_t] = 1.0f;        //該当するクラスに１をセットします。
}

//==================================================================================
//      活性化関数をここに置いておきます。
//==================================================================================
#include    "step.h"
#include    "relu.h"
#include    "sigmoid.h"
#include    "softmax.h"
static  step    _step;
static  ReLU    _ReLU;
static  sigmoid _sigmoid;
static  softmax _softmax;
//所望の活性化関数のポインタを返すようにします。
static activates* _get_act(int type)
{
    switch(type)
    {
        case AC_STEP:           return  (activates*)&_step;
        case AC_RELU:           return  (activates*)&_ReLU;
        case AC_SIGMOID:        return  (activates*)&_sigmoid;
        case AC_SOFTMAX:        return  (activates*)&_softmax;
        default:
        _Assert(0,"get_act(): illegal type");
    }
    return (activates*)0;
}


//==================================================================================
//      損失関数です。ここに置いておきます。
//==================================================================================
LOSS_mean_squared_error     _mean_squared_error;
LOSS_cross_entropy          _cross_entropy;
static loss     *       _get_lossfunc(int type)
{
    switch(type){
        case LOSS_MEAN_SQUARE:  return &_mean_squared_error;
        case LOSS_ENTROPY:      return &_cross_entropy;
        default:
        _Assert(0,"get_lossfunc(): illegal type");
    }
    return (loss*)0;
}
