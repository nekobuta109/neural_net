#include "net.h"

void net::backward(float learning_rate)
{
    //toku ���̎��_��t()�̓Z�b�g����Ă���O��ł��B

    //�֐��|�C���^�k���`�F�b�N���Ă����܂��B
    _Assert(loss_func!=0 , "net::backward() lossfunc is null");
    for(int l=0 ; l <= OUTPUT_LAYER ; ++l){   _Assert( activator[l]!=0 , "net:backward() activator is null");    }

    //======================================================
    //  �o�͑w
    //   dE        dE         dy
    //  ----  =  ------  *  ------
    //   da        dy         da
    //======================================================
    // dE/dy �o�͑w�́@�o�͂ƁA�𓚂̍��@�����֐��̔��������߂�B
    loss_func->array_dE_dy(n_units[OUTPUT_LAYER]+1 ,y , t , dE_dy);
    //�o�͑w��  �o�͒l�̊������֐��̔����@ dy / da  ( �ϐ��́@dz_da ���g�p�B) �ł��B
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
    //  �B��w
    //      dE                dE
    //  ----------   =   ------------  *  z (l)
    //    dw(l)             da(l+1) 
    //
    //      dE                             |       dE                   |
    //  ----------   =   h'[l]( ai(l) )) * | �� --------------  * w(l)ij |
    //     da(l)                           |     da j (l+1)             |
    //======================================================
    for(int l = OUTPUT_LAYER - 1 ; l > INPUT_LAYER ; --l) 
    {
//        printf("    back  ==== layer %d ===== \n" , l);
        //  dE/dw �����߂܂��B�@��i�� dE / da ���g�p dE/dw �� [l][0][0] ���X�V����E�E�E
        {
            for(int i = 0; i < n_units[l]+1 ; ++i) {
                for(int j = 0; j < n_units[l+1] + 1 ; ++j ) {
                    dE_dw[l][i][j] = z[l][i] * dE_da[l+1][j]; //toku dE/da �́A�O�i�� 1 - �����v�Z���ĂȂ��B�i�O�͂Ȃ�) 
                    dE_dw_total[l][i][j] += dE_dw[l][i][j];     //toku total�Ƃ��đ����Z�̌��ʂ�
//                    printf("dE_dw[%d][%d][%d] = %f  total=%f\n " , l , i , j , dE_dw[l][i][j] , dE_dw_total[l][i][j]  ) ;
                }
            }
        }
        //���̃��C���� dE / da �����߂܂��B
        {
            // �܂����̃��C���́@dz / da �����߂܂��B
            activator[l]->array_d_act(n_units[l]+1 , a[l] , dz_da );
            //dE_da ���v�Z���Ă����܂��B��͂�Ȃ��� 1 ���炵���X�V���Ȃ��B�B�B�B
            for(int i = 1; i < n_units[l] + 1 ; ++i ) {
                float tmp = 0.0;
                for(int j = 1; j < n_units[l+1]+1 ; ++j ) {
                    tmp += w[l][i][j] * dE_da[l+1][j];  //��dE/da*w ��i��
                }
//                printf("     (tmp=%f) --> " , tmp);
                dE_da[l][i] = dz_da[i][i] * tmp;    //���̃��C����dz/da = h'(a[l])
//               printf("dE_da[%d][%d] = %f\n" , l , i  , dE_da[l][i]  ) ;
            }
        }
    }   

    //======================================================
    //  ���͒i
    //         dE                 dE
    //  ---------------  =   -------------- * z(l) i 
    //      dw ij (0)            da j(1)      ^^^^^^^ (���͒l)
    //======================================================
    //toku �����͉B��w�́A dE_dw �����߂�v�Z�Ƃ��Ȃ��ł��B�i��i���Ȃ��̂ŁA dE_da[][]�����߂�K�v���Ȃ��B
    {
        for(int i = 0; i < n_units[INPUT_LAYER] + 1 ; ++i) {
            for(int j = 0; j < n_units[INPUT_LAYER + 1 ] + 1 ; ++j ) {
                dE_dw[INPUT_LAYER][i][j] = z[INPUT_LAYER][i] * dE_da[INPUT_LAYER+1][j];
                dE_dw_total[INPUT_LAYER][i][j] += dE_dw[0][i][j];
            }
        }
    }
    //dE_dw_total�����x�X�V�������𐔂��Ă����܂��B����� w_update()�Ŏg�p���܂��B
    //�o�b�`�Ŋw�K���邽�߂ɁAdE_dw_total���o�b�`�񐔂ŕ��ς������̂��g���܂��B
    dE_dw_total_n_update ++ ;
    _w_update(learning_rate);//auto update
}