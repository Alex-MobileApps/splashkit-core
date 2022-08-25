#include <math.h>
#include <stdexcept>
#include <iostream>

#include "neural_net.h"

using namespace std;

namespace splashkit_lib
{
    Layer::Layer(int n_inputs, int n_outputs, bool inc_bias, string name)
    {
        this->n_inputs = n_inputs;
        this->n_outputs = n_outputs;
        this->inc_bias = inc_bias;
        this->name = name;

        // Add input node weights (edges handled by layer subclasses e.g. dense, activation)
        for (int i = 0; i < n_inputs + inc_bias; i++)
            this->node_weights.push_back(0);
    }

    void Layer::set_node_weights(vector<float> node_weights)
    {
        // Validate input size
        if (node_weights.size() != this->n_inputs)
            throw invalid_argument("Invalid argument");

        // Set weights
        for (int i = 0; i < this->n_inputs; i++)
            this->node_weights[i] = node_weights[i];
    }

    vector<float> Layer::get_node_weights()
    {
        return this->node_weights;
    }

    DenseLayer::DenseLayer(int n_inputs, int n_outputs, bool inc_bias, string name) : Layer(n_inputs, n_outputs, inc_bias, name)
    {
        // Create fully connected edges
        for (int i = 0; i < this->n_inputs + this->inc_bias; i++)
        {
            vector<float> tmp_edge_weights;
            for (int j = 0; j < this->n_outputs; j++)
                tmp_edge_weights.push_back(2 * (float)rand() / RAND_MAX - 1);
            this->edge_weights.push_back(tmp_edge_weights);
        }
    }

    void DenseLayer::set_edge_weights(vector<vector<float> > edge_weights)
    {
        // Validate input size
        if (edge_weights.size() != this->edge_weights.size())
            throw invalid_argument("Invalid argument");

        // Set weights
        for (int i = 0; i < this->n_inputs; i++)
            for (int j = 0; j < this->n_outputs; j++)
                this->edge_weights[i][j] = edge_weights[i][j];
    }

    vector<std::vector<float> > DenseLayer::get_edge_weights()
    {
        return this->edge_weights;
    }

    void DenseLayer::display()
    {
        cout << this->name << " (" << this->n_inputs << ',' << this->n_outputs << ')' << endl;
        for (int i = 0; i < this->n_inputs; i++)
        {
            for (int j = 0; j < this->n_outputs; j++)
            {
                cout << "  Edge (" << i << ',' << j << "): " << this->edge_weights[i][j] << endl;
            }
        }
    }

    ActivationLayer::ActivationLayer(int n_inputs, string name) : Layer(n_inputs, n_inputs, false, name) {}

    void ActivationLayer::set_edge_weights(vector<vector<float> > edge_weights) {}

    vector<std::vector<float> > ActivationLayer::get_edge_weights()
    {
        return this->edge_weights;
    }

    void ActivationLayer::display()
    {
        cout << this->name << " (" << this->n_inputs << ',' << this->n_outputs << ')' << endl;
        cout << "  Activation" << endl;
    }

    Linear::Linear(int n_inputs, int n_outputs, bool inc_bias) : DenseLayer(n_inputs, n_outputs, inc_bias, "Linear") {}

    vector<float> Linear::forward()
    {
        vector<float> result;
        for (int i = 0; i < this->n_outputs; i++)
        {
            float tmp_result = 0;
            for (int j = 0; j < this->n_inputs + this->inc_bias; j++)
                tmp_result += this->node_weights[j] * this->edge_weights[j][i];
            result.push_back(tmp_result);
        }

        return result;
    }

    vector<float> Linear::backward(float lr, vector<float> &delta)
    {
        // Validate input size
        if (delta.size() != this->n_outputs)
            throw invalid_argument("Invalid argument");

        // Update edge weights and calculate running error
        vector<float> result;
        for (int i = 0; i < this->n_inputs; i++)
        {
            float tmp_result = 0;
            for (int j = 0; j < this->n_outputs; j++)
            {
                tmp_result += delta[j] * this->edge_weights[i][j];
                this->edge_weights[i][j] -= lr * delta[j] * this->node_weights[i];
            }
            result.push_back(tmp_result);
        }

        return result;
    }

    Sigmoid::Sigmoid(int n_inputs) : ActivationLayer(n_inputs, "Sigmoid") {}

    vector<float> Sigmoid::forward()
    {
        vector<float> result;
        for (int i = 0; i < this->n_inputs; i++)
            result.push_back(1.0 / (1 + exp(-this->node_weights[i])));
        return result;
    }

    vector<float> Sigmoid::backward(float lr, vector<float> &delta)
    {
        // Validate input size
        if (delta.size() != this->n_outputs)
            throw invalid_argument("Invalid argument");

        // Calculate running error
        vector<float> result = this->forward(); // sigmoid is reused in gradient function sig * (1 - sig)
        for (int i = 0; i < this->n_inputs; i++)
            result[i] *= delta[i] * (1 - result[i]);

        return result;
    }

    LossFunction::LossFunction(vector<float> &y, vector<float> &yhat)
    {
        // Validate input sizes
        if (y.size() != yhat.size())
            throw invalid_argument("Invalid argument");

        // Set state
        this->y = y;
        this->yhat = yhat;
    }

    MSELoss::MSELoss(vector<float> &y, vector<float> &yhat) : LossFunction(y, yhat) {}

    vector<float> MSELoss::backward()
    {
        vector<float> result;
        for (int i = 0; i < this->y.size(); i++)
            result.push_back(-(this->y[i] - this->yhat[i]));

        return result;
    }

    float MSELoss::loss()
    {
        float result = 0;
        for (int i = 0; i < this->y.size(); i++)
            result += pow(this->y[i] - this->yhat[i], 2);
        return result / 2;
    }

    template <typename T> void Sequential::add_layer(T &layer)
    {
        this->layers.push_back(make_shared<T>(layer));
    }
    template void Sequential::add_layer<Linear>(Linear &layer);
    template void Sequential::add_layer<Sigmoid>(Sigmoid &layer);

    vector<float> Sequential::forward(vector<float> &x)
    {
        this->layers[0]->set_node_weights(x);
        for (int i = 1; i < this->layers.size(); i++)
            this->layers[i]->set_node_weights(this->layers[i-1]->forward());
        return this->layers[this->layers.size()-1]->forward();
    }

    void Sequential::backward(LossFunction &loss_fn, float lr)
    {
        vector<float> delta = loss_fn.backward();
        for (int i = this->layers.size() - 1; i >= 0; i--)
            delta = this->layers[i]->backward(lr, delta);
    }

    void Sequential::display()
    {
        for (int i = 0; i < this->layers.size(); i++)
            this->layers[i]->display();
    }
}
