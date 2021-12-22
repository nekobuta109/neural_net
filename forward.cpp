#include "net.h"
const float * net::forward(const float input[])
{
	//==================================================
    //		input layer		[0]
	//==================================================
	//���͒l�́Aa[],z[]�Ƃ��ɁA 1 �I���W���Ƃ��܂��Bw�̓Y��0���o�C�A�X�Ƃ��Ďg���Ă���ȏ�A
	//�S������ɏK��Ȃ��Ƃ����܂���B
	//���͒l input[] �́A0 �I���W���Ƃ��܂��B�������邵���Ȃ��̂��Ȃ��E�E
	//a[0],z[0]�͕s��ł��B�ꉞ�O����Ă����܂���
/*for( int i=0 ; i< n_units[0]+1 ; ++i){
	printf("in[%d] %f\n" , i , input[i]);
}*/

	{
		a[0][0] = z[0][0]=0.0;	//�g���܂���
		for(int i=1 ; i < n_units[0]+1 ; ++ i ){
	    	a[0][i]=z[0][i] = input[i-1];
		}
	}
//	dump_layer_a(0);

	//==================================
    //		hidden layer	[1]-[L-2]
	//	�ƁAoutput layer [L-1]�̏��`�d�ł��B
	//==================================
	for(int l = 1; l < L; l++) {
//	printf(" -------------------------------  layer[%d] ---------------------------------\n" , l);
//dump_w(l);		
		//�O�i�̓����� z[l-1] �ł��B�ŏ��̃��C���̂Ƃ��́Az[0][]�@�����͂̒l�ł��B

		//�e���͂̒l�́A
		//����̃��C���̓��͒l�����߂Ă����B
		for(int j = 1; j < n_units[l]+1 ; ++j)	{		//���ꂪ�ΏۂƂȂ�j���[�������j�b�g�ł��Bi->j
//			a[l][j]  = w[l-1][0][j];				//bias�́A�O�i��[0]�Ԗڂ��o�C�A�X�v�f�̂悤�ł��B
			a[l][j]=0.0;
													//���ꂪ����Ă��邩���B������1�Ƃ��̌Œ�ł����̂��H
			for(int i = 1 ; i < n_units[l-1]+1 ; ++i) {	//�O�i�̃��j�b�g�����A�d�݂������Ȃ��瑫���Ă����܂��B
				a[l][j]	+=	w[l-1][i][j] * z[l-1][i];
			}
        }
		//z���A����̊������֐�����������̏����ł��B
//printf("f 3 layer[%d]/[%d]\n"  , l , L-1 );
//dump_layer_a(l);
		activator[l]->array_act(n_units[l] , a[l], z[l]);
//dump_layer_z(l);
	}
	//����ŁA�ŏI�i z[L-1]���A�j���[�����l�b�g�̓����ƂȂ�܂��B
	//y[]�ɃR�s�[���ĕԂ��܂��B
	for(int i=0 ; i< n_units[OUTPUT_LAYER]+1 ; ++i){
		y[i] = z[OUTPUT_LAYER][i];
//		printf("fw : y[%d] = %f\n" , i , y[i] );
	}
	return (const float*)y;
}
