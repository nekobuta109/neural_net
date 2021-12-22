#ifndef __MODEL_H__
#define __MODEL_H__

#include "common.h"
#include "activates.h"
#include "loss.h"

#define     _MAX_LAYERS     128         //128   layers max(�ŏI�i���܂߂�)


//activate types
enum{
    AC_STEP,
    AC_RELU,
    AC_SIGMOID,
    AC_SOFTMAX,
} ACTIVATE_TYPES;

//�����֐��̈ꗗ�ł��B
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
    float   w[_MAX_LAYERS][_MAX_UNITS][_MAX_UNITS];           //�d�ݕt���ł��B[���C��][�O�i][��i]���̃p�����[�^������܂��B
    float   z[_MAX_LAYERS][_MAX_UNITS];                       //�A�N�e�B�x�[�V������̒l�ł��B
    float   a[_MAX_LAYERS][_MAX_UNITS];                       //���͒l�ł��B�O�i��z�̑��a����
    float   y[_MAX_UNITS];                                      //���̃j���[�����l�b�g�̓����ł��B
    float   t[_MAX_UNITS];                                      //�����ł��B

    //back propergation�p�ϐ��B
    float   dE_dy[_MAX_UNITS];                      //�����֐��̔���    �i�[�p
    float   dz_da[_MAX_UNITS][_MAX_UNITS];          //�������֐��̔���  �i�[�p
    float   dE_da[_MAX_LAYERS][_MAX_UNITS];         //��i�ŎZ�o���ꂽ dE/da.
    //
    float   dE_dw[_MAX_LAYERS][_MAX_UNITS][_MAX_UNITS];                 //�ŏI�I�ɎZ�o���ꂽ dE/dw`
    float   dE_dw_total[_MAX_LAYERS][_MAX_UNITS][_MAX_UNITS];           //�o�b�`������
    int     dE_dw_total_n_update;                       //dE_dw�̍X�V�񐔁B
    //�R���X�g���N�^�ł��ˁB
    void    _dump_2D(const char *caption , float array[][_MAX_UNITS]               , int layer);
    void    _dump_3D(const char *caption, float array[][_MAX_UNITS][_MAX_UNITS]    , int layer);
    void    _w_update(float learning_rate);
public:
    net(){;}//�ꉞ���Ƃ���ł��ύX�ł���悤��
    net(int _n_inputs,int _n_hidden_layers , int _n_hidden_layer_units,int n_outputs , int hidden_layer_activate_type ,int output_layer_activate_type,int _loss_type )
    {
        build(_n_inputs,_n_hidden_layers ,_n_hidden_layer_units,n_outputs ,hidden_layer_activate_type ,output_layer_activate_type,_loss_type );
    }
    void build(int _n_inputs,int _n_hidden_layers , int _n_hidden_layer_units,int n_outputs , int hidden_layer_activate_type ,int output_layer_activate_type,int _loss_type );
    const float *forward(const float input[]);      //���`�d
    void backward(float learning_rate);              //�t�`�d
    void reset_w_update_parameter(void);        //
    //�O�����瑹���֐��𗘗p����ꍇ�̃C���^�[�t�F�C�X��p�ӂ��܂��B
    float loss(void);       //���߂�foward�̌��ʂ����ƂɌv�Z���܂��B
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

    //t(��)���Z�b�g���܂��B
    void set_t(int t);      //����n ��  �ŏI�i�̃N���X�����Ƃ���t�����܂��A

    //debug dump
    void dump_layer_a(int l)    {_dump_2D( "a" , a , l );   }
    void dump_layer_z(int l)    {_dump_2D( "z" , z , l );   }
    void dump_w(int l)          {_dump_3D( "w" , w , l);    }
};
#endif  //__MODEL_H__