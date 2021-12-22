#include <stdio.h>
#include "mini_batch.h"
//
void mini_batch::_run( bool train , float learning_rate ,const char*exam_name)
{
   _Assert ( _data.size() > 0 , "minibatch::_run() : data not loaded");
    _E_total = _E_true_count = 0 ;
    _net.reset_w_update_parameter();

#if 1
    for(int n = 0; n < _data.size(); ++n) {
#else       //debug
    for(int n = 0; n < 10; ++n) {
#endif
//printf("[%d]1\n" , n);
        _net.set_t( _data[n].t() );         //�񓚂��Z�b�g���܂��B
        _net.forward(_data[n].x());         //���`�d���܂��B
       //toku ������t[]�@��y[] ���ł��Ă���K�v������܂��B
        _E_total += _net.loss();   //�������v�Z���܂��B
//printf("[%d] t=%d  _E_total=%f\n"  ,  n  , _data[n].t()  , _E_total );
//        printf("[%d:%d]:" , _net.most_active_in_output_layer() , _data[n].t());
        if( _net.most_active_in_output_layer() == _data[n].t()){
                   _E_true_count++;  /*printf("O "); */  }
        else {
          /*  printf("X");*/
        }
        if(train){
//            printf(" ==== backward ====== t=%d \n" , _data[n].t() );
          _net.backward(learning_rate);
//            _net.w_update( learning_rate );
        }
//        printf("\n");
    }
    //���s���ʂ�\������悤�ɂ��܂��B
    printf("\t%s\tloss: %f,\taccuracy: %f\tsuccess(\t%d\t/ %lu\t)\n", (exam_name ? exam_name : "---") , E_average() , E_accuracy() , _E_true_count , _data.size() );
}
void mini_batch::load(const char* filename )
{
    FILE *fp;    int n_batch;    //�����Ńf�[�^���ƍ����Ă邩���m�F������x�ɂ��܂��B
    printf("load():[%s]\n" , filename);
    if((fp = fopen(filename, "r")) == NULL) {_Assert( 0, "batch::load() : open error");}
    fscanf(fp, "%d", &n_batch);       //�f�[�^���B
    //�f�[�^�������̐��l�̐��͂Ȃ����Œ�B�E�E�E
    printf("opened : n[%d]\n" , n_batch);
    for(int i = 0; i < n_batch ; ++i ) {
        train_data d(_net.n_input());
        d.reset();
        {int t ;     fscanf(fp, "%d", &t);  d.set_t(t);}                                    //answer
        for(int j = 0; j < d.size() ; ++j ) { float v;  fscanf(fp, "%f", &v);  d.set(j,v);   }  //data
        _data.push_back(d);
//        delete d;
    }
    fclose(fp);
}