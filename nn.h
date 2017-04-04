/*
* @Author: kmrocki
* @Date:   2016-02-24 15:28:10
* @Last Modified by:   kmrocki
* @Last Modified time: 2016-03-11 10:14:25
*/

/**************

TODO:
- Gradient check
- Serialization

**************/

#ifndef __NN_H__
#define __NN_H__

#include <layers.h>
#include <iomanip>

class NN {

	public:

		std::deque<Layer*> layers;

		const size_t batch_size;

		void forward(Matrix& input_data, bool skip_dropout = false) {

			//copy inputs to the lowest point in the network
			layers[0]->x = input_data;

			//compute forward activations
			for (size_t i = 0; i < layers.size(); i++) {

				//y = f(x)
				layers[i]->forward(skip_dropout);

				//x(next layer) = y(current layer)
				if (i + 1 < layers.size())
					layers[i + 1]->x = layers[i]->y;
			}

		}

		void backward(Matrix& t) {

			//set targets at the top
			layers[layers.size() - 1]->dy = t;

			//propagate error backward
			for (int i = layers.size() - 1; i >= 0; i--) {

				layers[i]->resetGrads();
				layers[i]->backward();

				//dy(previous layer) = dx(current layer)
				if (i > 0) {

					layers[i - 1]->dy = layers[i]->dx;

				}

			}

		}

		void update(float alpha, float decay) {

			//update all layers according to gradients
			for (size_t i = 0; i < layers.size(); i++) {

				layers[i]->applyGrads(alpha, decay);

			}

		}

		void train(std::deque<datapoint>& data, float alpha, float decay, size_t iterations, size_t classes) {

			//get random examples of size batch_size from data
			Eigen::VectorXi random_numbers(batch_size);
			Matrix batch, targets;
			float loss, smooth_loss;
			float damping_factor = 0.99;

			/* TODO: change it to an external param */

			size_t correct;

			for (size_t ii = 0; ii < iterations; ii++) {

				correct = 0;

				randi(random_numbers, 0, data.size() - 1);

				// [784 x batch_size]
				batch = make_batch(data, random_numbers);
				targets = make_targets(data, random_numbers, classes);

				//forward activations
				forward(batch);

				correct += count_correct_predictions(layers[layers.size() - 1]->y, targets);

				loss = cross_entropy(layers[layers.size() - 1]->y, targets);

				smooth_loss = ii == 0 ? loss : damping_factor * smooth_loss + (1 - damping_factor) * loss;
				std::cout << std::setprecision(1) << std::setw(5) << "[" << std::setw(3) << ii + 1 << "/"
						  << std::setw(3) << iterations
						  << "] Loss = " << std::setw(5) << loss << ", Smooth loss = "
						  << std::setw(5) << smooth_loss << ", " << std::setw(5)
						  << std::fixed << 100.0 * (float)correct / (float)batch_size
						  << "% batch correct" << "      \r" << std::flush;

				//backprogagation
				backward(targets);

				//apply changes
				update(alpha, decay);
			}

		}


		/* TDOO: Eigen::MatrixXf compute_numerical_gradients(void) { } */

		float test(std::deque<datapoint>& data, size_t classes) {

			Eigen::VectorXi numbers(batch_size);
			Matrix batch, targets;

			size_t correct = 0;

			for (size_t ii = 0; ii < data.size(); ii += batch_size) {

				linspace(numbers, ii, ii + batch_size);

				batch = make_batch(data, numbers);
				targets = make_targets(data, numbers, classes);

				//skip dropout layers at test time
				bool skip_dropout = true;

				forward(batch, skip_dropout);

				correct += count_correct_predictions(layers[layers.size() - 1]->y, targets);

			}

			return (float)correct / (float)(data.size());

		}

		/* TODO: serialization */

		void save_to_files(std::string prefix) {

			for (size_t i = 0; i < layers.size(); i++) {

				std::ostringstream ss;
				ss << std::setw(3) << std::setfill('0') << i + 1;
				layers[i]->save_to_files(prefix + "_" + ss.str() + "_" + layers[i]->name);
			}

		}

		NN(size_t minibatch_size) : batch_size(minibatch_size) { }

		~NN() {

			for (size_t i = 0; i < layers.size(); i++) {

				delete(layers[i]);
			}

		}
}; /*class NN */

#endif