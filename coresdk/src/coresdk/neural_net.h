#ifndef neural_net_h
#define neural_net_h

#include <vector>

namespace splashkit_lib
{
    struct Layer
    {
        protected:
            int n_inputs;
            int n_outputs;
            bool inc_bias;
            std::string name;
            std::vector<float> node_weights;
            std::vector<std::vector<float> > edge_weights;

        public:
            Layer(int n_inputs, int n_outputs, bool inc_bias, bool activation, std::string name);
            void set_node_weights(std::vector<float> node_weights);
            void set_edge_weights(std::vector<std::vector<float> > edge_weights);
            std::vector<float> get_node_weights();
            std::vector<std::vector<float> > get_edge_weights();
            void display();
            virtual std::vector<float> forward() = 0;
            virtual std::vector<float> backward(float lr, std::vector<float> &delta) = 0;
    };

    struct Linear : Layer
    {
        public:
            Linear(int n_inputs, int n_outputs, bool inc_bias);
            std::vector<float> forward();
            std::vector<float> backward(float lr, std::vector<float> &delta);
    };

    struct Sigmoid : Layer
    {
        public:
            Sigmoid(int n_inputs);
            std::vector<float> forward();
            std::vector<float> backward(float lr, std::vector<float> &delta);
    };

    struct LossFunction
    {
        protected:
            std::vector<float> y;
            std::vector<float> yhat;

        public:
            LossFunction(std::vector<float> &y, std::vector<float> &yhat);
            virtual std::vector<float> backward() = 0;
            virtual float loss() = 0;
    };

    struct MSELoss : LossFunction
    {
        public:
            MSELoss(std::vector<float> &y, std::vector<float> &yhat);
            std::vector<float> backward();
            float loss();
    };

    struct Sequential
    {
        private:
            std::vector<std::shared_ptr<Layer> > layers;

        public:
            template <typename T> void add_layer(T &layer);
            std::vector<float> forward(std::vector<float> &x);
            void backward(LossFunction &loss_fn, float lr);
            void display();
    };
}

#endif
