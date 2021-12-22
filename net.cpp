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
//      �l�b�g���[�N�\��
//==================================================================================
void net::build(int n_inputs,int n_hidden_layers , int n_hidden_layer_units,int n_outputs , int hidden_layer_activate_type ,int output_layer_activate_type,int loss_type )
{
    L = 1 + n_hidden_layers + 1; //���́E�o�͒i���ꏏ�ɂ��܂��B
                            //���͒i        �F[0]
                            //�B��i�K      �F[1-L-2]
                            //�o�͒i�K      : [L-1]
                            //�Ƃ��ĉ^�p���Ă݂�B������ɂ�����΂��ƂŕύX�B
                            //�l�����Ƃ��ē����z��Ƃ��ē��́E���ԃ��C���E�o�̓��C���������Ƃ����l���B

    _Assert( L < _MAX_LAYERS , "net::build() layer overflow");
    _Assert( n_inputs +1 < _MAX_UNITS , "net::builde() input layer overflow");
    _Assert( n_outputs +1 < _MAX_UNITS , "net::builde() output layer overflow");
    _Assert( n_hidden_layer_units + 1 < _MAX_UNITS , "net::build() hiddenl layase unit overflow");

     //----------------------------------------------------
    //�e���C���̃j���[�����������߂܂��B�����͌��ߑł��̂悤�ł��B
    //----------------------------------------------------
    n_units[0] = n_inputs;     //���i
    for(int i = 1; i < L-1; ++i) {  n_units[i] = n_hidden_layer_units;     }  //�B�����C��
    n_units[L-1] = n_outputs;
    //----------------------------------------------------
    //  �e���C���̊������֐��̐ݒ�ł��B������ReLU����̂悤�ł��B
    //----------------------------------------------------
    for(int i = 0; i < L-1; ++i) {   activator[i] = _get_act(hidden_layer_activate_type);    }
    activator[L-1] = _get_act(output_layer_activate_type);
    //�ꉞ�k���|�C���^���Ȃ��悤�ɂ��Ă������H
    
    loss_func = _get_lossfunc(loss_type);    //�����֐��ł�.

    // w �����l���Z�b�g�A�b�v���Ă����܂��B
    //w �́A�֋X��A�o�C�A�X�����i�����d�݁j���A[0]�Ƃ���悤�ɂ��܂��B
    //���̂��߁A�z��� [���C��][0:bias , 1 ? ���j�b�g��]���Ή����鐔�ƂȂ�B
    //��₱�������ǁA��������悤�ł��B
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
    //z ����̓A�N�e�B�x�[�V������̒l���m�ۂ��邽�߂�����B�����l���邩?0�ɏ������ł��B
    memset( (void*)&z[0][0] , 0 , sizeof(z) );
    //a ���Z�b�g�A�b�v���Ă����܂��B
    memset( (void*)&a[0][0] , 0 , sizeof(a) );
     //y ���Z�b�g�A�b�v���܂��B y �� �ŏI�i�̓����ł��B
    memset( (void*)&y[0] , 0 , sizeof(y));    
    memset( (void*)&t[0] , 0 , sizeof(t));   
    //dE/dw �i�[�p�B����͂ł����Ƃ������ȁB
    memset( (void*)&dE_dw[0][0][0]      , 0 , sizeof(dE_dw) );
    memset( (void*)&dE_dw_total[0][0][0] , 0 , sizeof(dE_dw_total) );
    //dE/da �i�[�p�B
    memset( (void*)&dE_da[0][0]      , 0 , sizeof(dE_da) );
}
//w_update�Ɋւ��鑀��B
void net::reset_w_update_parameter(void)
{
    memset((void*)&dE_dw_total[0][0][0] , 0 , sizeof(dE_dw_total) );
//  float dE_dw_total[_MAX_LAYERS][_MAX_UNITS][_MAX_UNITS];          //dE/dw total? ���ς炵���B
    dE_dw_total_n_update = 0 ;
}
//w�̍X�V��Ƃł��Btotal_dE_dw_total���g�p���čs���܂��B
void net::_w_update(float learning_rate)
{
    _Assert(dE_dw_total_n_update > 0 , "net::w_update() no back_propagation");
    //======================================================
    //  w[l][i][j]  �̍X�V���s���܂��B
    //======================================================
    for(int l = 0; l < OUTPUT_LAYER ; ++l ) {
        for(int i = 1; i < n_units[l]+1 ; ++i) {
            for(int j = 0; j < n_units[l+1]+1 ; ++j ) {
                float _backup_w = w[l][i][j];   //(debug)
                w[l][i][j] -= learning_rate * dE_dw[l][i][j];
                //w�̍X�V��total���g���A�o�b�`�T�C�Y�ŕ��ς��Ƃ������́j�Ƃ��Čv�Z����悤���B�E�E
                w[l][i][j] -= learning_rate * (dE_dw_total[l][i][j] / (float)dE_dw_total_n_update);
            }
        }
    }
}
//���߂�foward�̌��ʂ����ƂɌv�Z���܂��B(�z��̑����֐����Ăт܂��B)
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
    _Assert( _t < n_units[OUTPUT_LAYER] + 1 , "set_t  overflow" );   //+! �͂P�I���W���ł����v�Ȃ悤��
    memset( (void*)t , 0 ,sizeof(t) );
    t[_t] = 1.0f;        //�Y������N���X�ɂP���Z�b�g���܂��B
}

//==================================================================================
//      �������֐��������ɒu���Ă����܂��B
//==================================================================================
#include    "step.h"
#include    "relu.h"
#include    "sigmoid.h"
#include    "softmax.h"
static  step    _step;
static  ReLU    _ReLU;
static  sigmoid _sigmoid;
static  softmax _softmax;
//���]�̊������֐��̃|�C���^��Ԃ��悤�ɂ��܂��B
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
//      �����֐��ł��B�����ɒu���Ă����܂��B
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
