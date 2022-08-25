#ifndef neural_net_h
#define neural_net_h

#include <vector>
#include <string>

namespace splashkit_lib
{
    /**
     * A Layer is a base struct used to implement various types of neural network layers
     */
    struct Layer
    {
        protected:

            /**
             * Number of input nodes in the neural network layer
             */
            int n_inputs;

            /**
             * Number of output nodes in the neural network layer
             */
            int n_outputs;

            /**
             * Whether or not a bias unit should be included in the layer
             */
            bool inc_bias;

            /**
             * A name that identifies the type of layer
             */
            std::string name;

            /**
             * Values outputted from the previous layer during the most recent forward pass
             * or the input node values if the first layer
             */
            std::vector<float> node_weights;

            /**
             * Weights for each edge in the layer
             * The implementation of this variable varies depending on the type of layer created (e.g. dense, activation, etc.)
             */
            std::vector<std::vector<float> > edge_weights;

        public:

            /**
             * Construct a new Layer object
             *
             * @param n_inputs      Number of input nodes in the layer
             * @param n_outputs     Number of output nodes in the layer
             * @param inc_bias      Whether or not a bias unit should be included in the layer
             * @param name          A name that identifies the type of layer
             */
            Layer(int n_inputs, int n_outputs, bool inc_bias, std::string name);

            /**
             * Set the node weights object
             *
             * @param node_weights      Node weights to replace the existing node weights with
             */
            void set_node_weights(std::vector<float> node_weights);

            /**
             * Get the node weights object
             *
             * @return std::vector<float>   Current node weights in the layer
             */
            std::vector<float> get_node_weights();

            /**
             * Set the edge weights object
             *
             * @param edge_weights      Edge weights to replace the existing edge weights with
             */
            virtual void set_edge_weights(std::vector<std::vector<float> > edge_weights) = 0;

            /**
             * Get the edge weights object
             *
             * @return std::vector<std::vector<float> >     Current edge weights in the layer
             */
            virtual std::vector<std::vector<float> > get_edge_weights() = 0;

            /**
             * Displays a summary of the layer's state in the console
             */
            virtual void display() = 0;

            /**
             * Performs and returns the output of a forward pass on the layer using the existing node weights
             *
             * @return std::vector<float>   Output values for the forward pass on the layer
             */
            virtual std::vector<float> forward() = 0;

            /**
             * Performs a backwards pass on the layer, updates the layers weights, and
             * returns the new accumulated error for each of the input nodes
             *
             * @param lr                        Learning rate
             * @param delta                     Accumulated error for each output node in the layer
             * @return std::vector<float>       New accumulated error for each input node in the layer
             */
            virtual std::vector<float> backward(float lr, std::vector<float> &delta) = 0;
    };

    /**
     * A fully connected neural network layer
     */
    struct DenseLayer : Layer
    {
        public:
            DenseLayer(int n_inputs, int n_outputs, bool inc_bias, std::string name);
            void display();
            void set_edge_weights(std::vector<std::vector<float> > edge_weights);
            std::vector<std::vector<float> > get_edge_weights();
    };

    /**
     * An activation only neural network layer without edges
     */
    struct ActivationLayer : Layer
    {
        public:
            ActivationLayer(int n_inputs, std::string name);
            void display();
            void set_edge_weights(std::vector<std::vector<float> > edge_weights);
            std::vector<std::vector<float> > get_edge_weights();
    };

    /**
     * A dense neural network layer that performs a weighted linear combination of the input nodes
     * to each output node
     */
    struct Linear : DenseLayer
    {
        public:
            Linear(int n_inputs, int n_outputs, bool inc_bias);
            std::vector<float> forward();
            std::vector<float> backward(float lr, std::vector<float> &delta);
    };

    /**
     * A neural network layer that performs sigmoid activation of the input nodes
     */
    struct Sigmoid : ActivationLayer
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
