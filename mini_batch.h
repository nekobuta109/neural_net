#ifndef __MINI_BATCH_H__
#define __MINI_BATCH_H__

#include "net.h"
#include "common.h"
#include <string.h>
#include <vector>

#define _CLASS_MAX      128
#define _DATA_MAX       32
//toku これが1回分の読み込みデータです。
class data_array {
protected:
    //toku とりあえず難しいので、固定長にしてみます。
    float _x[_DATA_MAX];   // input
    int _size;  //
public:
    data_array()   {    _Assert(0,"input data size (constractor) must.");}
    data_array(int data_size) :  _size(data_size) {
        //動的な領域確保をするとvectorへのpush_backでエラーになるので差し当たり固定長、サイズチェックだけします
        _Assert( _size < _DATA_MAX ,  "data_array: data size overflow");
    }
    ~data_array(){;}
    void reset(void){   memset((void*)_x , 0 , _size*sizeof(float) );    }
    const float *x(void)    {   return (const float*)_x;     }

    int         size(void){return _size;}
    void        set(int idx,float val)     {  _Assert(idx < _size, "set()overflow"); _x[idx]=val;   }
};
class train_data   :public data_array
{
protected:
    int _t;      // 答えのデータです。

public:
    train_data(int data_size)  : data_array(data_size)  {;}  
    int t(void)const    {return _t;}
    void set_t(int t){  _t=t;   }
};


//バッチ処理用のクラスを作ります。
class mini_batch
{
protected:
    net     &_net;
    void _run( bool train , float learning_rate=0.0 ,const char*exam_name=0);
    //-------------------------------------------------------
    //  datas
    //-------------------------------------------------------
    std::vector<train_data>     _data;          //これがデータ配列となります。

    //実行結果格納用
    float   _E_total;           ///
    int   _E_true_count;       //
public:
    //-------------------------------------------------------
    //  methods:
    //-------------------------------------------------------
//    mini_batch() : _net(new net) { _Assert( 0 , "mini_batch cant allow default constructor" ); delete &_net;    }
    mini_batch(net &_n , const char*train_data_name): _net(_n)     { load(train_data_name);  } 
    void    load(const char* filename );
    void    do_train(float learning_rate,const char*exam_name=0)    {   return  _run( true ,learning_rate , exam_name);    }
    void    do_evalation(const char*exam_name=0)                    {   return  _run(false ,0.0,exam_name);                }

    //netupdate_parameters(model_parameter, learning_rate, N_train);
    float   E_average(void) const     {   if(_data.size())    {   return _E_total / _data.size();     }   else return 0.0;    } 
    float   E_accuracy(void) const    {   if(_data.size())    {   return (float)_E_true_count/_data.size();    }   else return 0.0;    }
};
#endif  //__MINI_BATCH_H__
