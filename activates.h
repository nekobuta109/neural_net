#ifndef __ACTIVATES_H__
#define __ACTIVATES_H__
#include "common.h"

class activates
{
public:
    virtual float act(float a)=0;           //activate
    virtual float d_act(float a)=0;         //���������֐�
    //array
    virtual void array_act(int n , float in[] , float out[]){
//         printf("activates::array_Act() in(out)\n" );
        for(int i=0 ;  i < n ; ++i){
            out[i]=act(in[i]);
        }
    };
    virtual void array_d_act(int n , float in[] , float d_da[][_MAX_UNITS]){
        //�\�t�g�}�b�N�X�֐��ȊO�̓X�J���[���Ƃ�܂��̂ł��̌`�ł��B
        //�񎟌��z����Ƃ�̂́A�\�t�g�}�b�N�X�ɍ��킹�����߁B
        //�\�t�g�}�b�N�X�ȊO�̊֐��� d_da[i][i]���A�N�Z�X����B
        //�S���̐������[���N���A
        for(int i=0;i<n;++i)for(int j=0; j<n ;++j) {d_da[i][j]=0.0;}
        //�ꎟ���̔z��Ƃ��āA�֋X��񎟌��z��̑Ίp���������l������B
        for(int i=0 ; i<n ; ++i){
            d_da[i][i]=d_act(in[i]);
        }
    };
};
#endif