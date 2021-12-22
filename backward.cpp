#include "net.h"

void net::backward(float learning_rate)
{
    //toku この時点でt()はセットされている前提です。

    //関数ポインタヌルチェックしておきます。
    _Assert(loss_func!=0 , "net::backward() lossfunc is null");
    for(int l=0 ; l <= OUTPUT_LAYER ; ++l){   _Assert( activator[l]!=0 , "net:backward() activator is null");    }

    //======================================================
    //  出力層
    //   dE        dE         dy
    //  ----  =  ------  *  ------
    //   da        dy         da
    //======================================================
    // dE/dy 出力層の　出力と、解答の差　損失関数の微分を求める。
    loss_func->array_dE_dy(n_units[OUTPUT_LAYER]+1 ,y , t , dE_dy);
    //出力層の  出力値の活性化関数の微分　 dy / da  ( 変数は　dz_da を使用。) です。
    activator[OUTPUT_LAYER]->array_d_act( n_units[OUTPUT_LAYER]+1 ,  a[OUTPUT_LAYER], dz_da);
    //
    for(int i = 1; i < n_units[OUTPUT_LAYER]+1 ; ++ i ) {
        dE_da[OUTPUT_LAYER][i] = 0.0f;
        for(int _k = 1; _k < n_units[OUTPUT_LAYER] + 1 ; ++_k ) {
            dE_da[OUTPUT_LAYER][i]  +=  dE_dy[_k] * dz_da[_k][i];
//printf("OUT_LAYER : dE_da[%d][%d] = %f\n" , OUTPUT_LAYER , i  , dE_da[OUTPUT_LAYER][i] );
        }
    }
    //======================================================
    //  隠れ層
    //      dE                dE
    //  ----------   =   ------------  *  z (l)
    //    dw(l)             da(l+1) 
    //
    //      dE                             |       dE                   |
    //  ----------   =   h'[l]( ai(l) )) * | Σ --------------  * w(l)ij |
    //     da(l)                           |     da j (l+1)             |
    //======================================================
    for(int l = OUTPUT_LAYER - 1 ; l > INPUT_LAYER ; --l) 
    {
//        printf("    back  ==== layer %d ===== \n" , l);
        //  dE/dw を求めます。　後段の dE / da を使用 dE/dw は [l][0][0] も更新する・・・
        {
            for(int i = 0; i < n_units[l]+1 ; ++i) {
                for(int j = 0; j < n_units[l+1] + 1 ; ++j ) {
                    dE_dw[l][i][j] = z[l][i] * dE_da[l+1][j]; //toku dE/da は、前段で 1 - しか計算してない。（０はない) 
                    dE_dw_total[l][i][j] += dE_dw[l][i][j];     //toku totalとして足し算の結果を
//                    printf("dE_dw[%d][%d][%d] = %f  total=%f\n " , l , i , j , dE_dw[l][i][j] , dE_dw_total[l][i][j]  ) ;
                }
            }
        }
        //このレイヤの dE / da を求めます。
        {
            // まずこのレイヤの　dz / da を求めます。
            activator[l]->array_d_act(n_units[l]+1 , a[l] , dz_da );
            //dE_da を計算していきます。やはりなぜか 1 からしか更新しない。。。。
            for(int i = 1; i < n_units[l] + 1 ; ++i ) {
                float tmp = 0.0;
                for(int j = 1; j < n_units[l+1]+1 ; ++j ) {
                    tmp += w[l][i][j] * dE_da[l+1][j];  //ΣdE/da*w 後段の
                }
//                printf("     (tmp=%f) --> " , tmp);
                dE_da[l][i] = dz_da[i][i] * tmp;    //このレイヤのdz/da = h'(a[l])
//               printf("dE_da[%d][%d] = %f\n" , l , i  , dE_da[l][i]  ) ;
            }
        }
    }   

    //======================================================
    //  入力段
    //         dE                 dE
    //  ---------------  =   -------------- * z(l) i 
    //      dw ij (0)            da j(1)      ^^^^^^^ (入力値)
    //======================================================
    //toku ここは隠れ層の、 dE_dw を求める計算とおなじです。（後段がないので、 dE_da[][]を求める必要がない。
    {
        for(int i = 0; i < n_units[INPUT_LAYER] + 1 ; ++i) {
            for(int j = 0; j < n_units[INPUT_LAYER + 1 ] + 1 ; ++j ) {
                dE_dw[INPUT_LAYER][i][j] = z[INPUT_LAYER][i] * dE_da[INPUT_LAYER+1][j];
                dE_dw_total[INPUT_LAYER][i][j] += dE_dw[0][i][j];
            }
        }
    }
    //dE_dw_totalを何度更新したかを数えておきます。これは w_update()で使用します。
    //バッチで学習するために、dE_dw_totalをバッチ回数で平均したものを使います。
    dE_dw_total_n_update ++ ;
    _w_update(learning_rate);//auto update
}