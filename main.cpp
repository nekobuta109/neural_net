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
    const float LEARNING_RATE = 0.01f;
    //ネットワークを作ります。
    net *_net = new net( n_inputs , 2  , 16 ,  4  , AC_RELU , AC_SOFTMAX , LOSS_ENTROPY );  //make net
    mini_batch * train_batch = new mini_batch(*_net , train_data_name);
    mini_batch * validate_batch = new mini_batch(*_net , validate_data_name);
    mini_batch * test_batch = new mini_batch(*_net , test_data_name);
 
    for(int epoch = 1; epoch <= EPOCH; epoch++) {
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
    train(10 , "./data/train_data.txt", "./data/validation_data.txt" ,"./data/test_data.txt");

}