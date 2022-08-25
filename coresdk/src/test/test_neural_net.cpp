#include "neural_net.h"
#include <iostream>

using namespace std;
using namespace splashkit_lib;

void run_neural_net_test()
{
    // Create layers
    Linear l1 = Linear(2, 1, false);
    Sigmoid l2 = Sigmoid(1);
    Linear l3 = Linear(1, 2, false);
    Sigmoid l4 = Sigmoid(2);

    // Connect layers
    Sequential seq = Sequential();
    seq.add_layer<Linear>(l1);
    seq.add_layer<Sigmoid>(l2);
    seq.add_layer<Linear>(l3);
    seq.add_layer<Sigmoid>(l4);

    // Create an input
    vector<float> x;
    x.push_back(.59);
    x.push_back(.1);

    // Expected output
    vector<float> y;
    y.push_back(1);
    y.push_back(0);

    // Train network
    for (int epoch = 0; epoch < 100000; epoch++)
    {
        vector<float> yhat = seq.forward(x);
        MSELoss loss_fn = MSELoss(y, yhat);
        seq.backward(loss_fn, 0.05);
        if (epoch % 1000 == 0)
            cout << loss_fn.loss() << endl;
    }

    // Display final network state
    seq.display();
}