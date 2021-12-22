#include <stdio.h>
#include <stdlib.h>

#include "step.h"
#include "relu.h"
#include "sigmoid.h"
#include "softmax.h"

#include "loss.h"


#include "net.h"
#include "mini_batch.h"
//==================================================================================
//      main
//==================================================================================
static void _dump(float a[] , float z[] , int d , const char*caption)
{
    printf("---------------------\n");
    printf("%s\n" , caption);
    printf("---------------------\n");
    for(int i = 0; i < d ; ++i){
        printf("%f : %f \n" , a[i] , z[i]);
    }
}
//-------------------------------------------------------------------------------------------------
//      トレーニングのコードです。
//-------------------------------------------------------------------------------------------------
void train(const int EPOCH , const char* train_data_name , const char* validate_data_name , const  char* test_data_name )
{
    const int n_inputs = 5*5;
//    const int EPOCH = 10;
    const float LEARNING_RATE = 0.01f;
    //ネットワークを作ります。
    net *_net = new net( n_inputs , 2  , 16 ,  4  , AC_RELU , AC_SOFTMAX , LOSS_ENTROPY );  //make net
    //
    mini_batch * train_batch = new mini_batch(*_net , train_data_name);
    mini_batch * validate_batch = new mini_batch(*_net , validate_data_name);
    mini_batch * test_batch = new mini_batch(*_net , test_data_name);
 
    for(int epoch = 1; epoch <= EPOCH; epoch++) {
//        printf(" ========= Epoch: %d / %d\n<train> ======== \n", epoch, EPOCH);
        train_batch->do_train(LEARNING_RATE , "<trainning>");
        validate_batch->do_evalation("<validate>");
    }
    test_batch->do_evalation("<test>");    

    {
        delete _net;
        delete train_batch;
        delete validate_batch;
        delete test_batch;
    }
}
int main(int argc,char**argv)
{
//    void train(const char* train_data_name , const char* validate_data_name , const  char* test_data_name )

//    train(".\\data\\train_data.txt", ".\\data\\validation_data.txt.txt" ,".\\data\\test_data.txt");
    train(10 , "./data/train_data.txt", "./data/validation_data.txt" ,"./data/test_data.txt");

    /*
    const int D = 10;   //demension 10
    float a[D] = { -3.0 , -2.0 , -1.0 , 0.0 , 1.0  , 2.0  , 3.0 , 4.0 , 5.0 , 6.0 };
    float z[D] ;    //結果格納用
    step s;
    ReLU relu;
    sigmoid sigm;
    softmax soft;
    //activate test
    s.array_act(D,a,z);         _dump(a,z,D,"STEP");
    relu.array_act(D,a,z);      _dump(a,z,D,"ReLU");
    sigm.array_act(D,a,z);      _dump(a,z,D,"sigmoid");
    soft.array_act(D,a,z);      _dump(a,z,D,"softmax");
    //loss test
    {
        LOSS_mean_squared_error loss_sq;
        LOSS_cross_entropy      loss_et;
        float y =10 ,t=13;
        float E,D;
        E = loss_sq.E(y,t); loss_sq.dE_dy(y,t);
               printf("E = %f D=%f\n" , E , D);
        E = loss_et.E(y,t); loss_et.dE_dy(y,t);
               printf("E = %f D=%f\n" , E , D);
    }
    */
    //

}