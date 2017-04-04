/*
* @Author: kmrocki
* @Date:   2016-02-24 09:43:05
* @Last Modified by:   kmrocki
* @Last Modified time: 2016-03-01 09:23:07
*/

#include <importer.h>
#include <utils.h>
#include <layers.h>
#include <nn.h>
#include <iostream>


int main() {

	size_t epochs = 500;
	size_t batch_size = 250;
	double learning_rate = 1e-2;
	double weight_decay = 0;

	NN nn(batch_size);

	// a softmax-only version, no hidden layers (~91.6% after 10, ~92.4% after 50)
	// nn.layers.push_back(new Linear(28 * 28, 10, batch_size));
	// nn.layers.push_back(new Softmax(10, 10, batch_size));

	// 28 x 28 -> 400 ReLU -> 10 (97.2% after 10, 98% after 50 epochs)
	// nn.layers.push_back(new Linear(28 * 28, 400, batch_size));
	// nn.layers.push_back(new ReLU(400, 400, batch_size));
	// nn.layers.push_back(new Linear(400, 10, batch_size));
	// nn.layers.push_back(new Softmax(10, 10, batch_size));

	// 28 x 28 -> 400 -> 100 -> 10 (97.4% after 10, 98% after 50 epochs, train 99.9%)
	// nn.layers.push_back(new Linear(28 * 28, 400, batch_size));
	// nn.layers.push_back(new ReLU(400, 400, batch_size));
	// nn.layers.push_back(new Linear(400, 100, batch_size));
	// nn.layers.push_back(new ReLU(100, 100, batch_size));
	// nn.layers.push_back(new Linear(100, 10, batch_size));
	// nn.layers.push_back(new Softmax(10, 10, batch_size));

	// 28 x 28 -> 784 -> 400 -> 10 (~98.1% after 10, 98.3% after 50, train 100%)
	// nn.layers.push_back(new Linear(28 * 28, 784, batch_size));
	// nn.layers.push_back(new ReLU(784, 784, batch_size));
	// nn.layers.push_back(new Linear(784, 400, batch_size));
	// nn.layers.push_back(new ReLU(400, 400, batch_size));
	// nn.layers.push_back(new Linear(400, 10, batch_size));
	// nn.layers.push_back(new Softmax(10, 10, batch_size));

	// 3 hidden
	// 28 x 28 -> 784 -> 400 -> 256 -> 10 (~98.1% after 10, 98.3% after 50, train 100%)
	// nn.layers.push_back(new Linear(28 * 28, 784, batch_size));
	// nn.layers.push_back(new ReLU(784, 784, batch_size));
	// nn.layers.push_back(new Linear(784, 400, batch_size));
	// nn.layers.push_back(new ReLU(400, 400, batch_size));
	// nn.layers.push_back(new Linear(400, 256, batch_size));
	// nn.layers.push_back(new ReLU(256, 256, batch_size));
	// nn.layers.push_back(new Linear(256, 10, batch_size));
	// nn.layers.push_back(new Softmax(10, 10, batch_size));

	//[60000, 784]
	// std::deque<datapoint> train_data =
	// 	MNISTImporter::importFromFile("data/mnist/train-images-idx3-ubyte",
	// 								  "data/mnist/train-labels-idx1-ubyte");
	// //[10000, 784]
	// std::deque<datapoint> test_data =
	// 	MNISTImporter::importFromFile("data/mnist/t10k-images-idx3-ubyte",
	// 								  "data/mnist/t10k-labels-idx1-ubyte");

	//[60000, 784]
	std::deque<datapoint> train_data, partial_data;
	partial_data = CIFAR10Importer::importFromFile("data/cifar-10-batches-bin/data_batch_1.bin");
	train_data.insert(train_data.end(), partial_data.begin(), partial_data.end());
	partial_data = CIFAR10Importer::importFromFile("data/cifar-10-batches-bin/data_batch_2.bin");
	train_data.insert(train_data.end(), partial_data.begin(), partial_data.end());
	partial_data = CIFAR10Importer::importFromFile("data/cifar-10-batches-bin/data_batch_3.bin");
	train_data.insert(train_data.end(), partial_data.begin(), partial_data.end());
	partial_data = CIFAR10Importer::importFromFile("data/cifar-10-batches-bin/data_batch_4.bin");
	train_data.insert(train_data.end(), partial_data.begin(), partial_data.end());
	partial_data = CIFAR10Importer::importFromFile("data/cifar-10-batches-bin/data_batch_5.bin");
	train_data.insert(train_data.end(), partial_data.begin(), partial_data.end());

	//[10000, 784]
	std::deque<datapoint> test_data =
		CIFAR10Importer::importFromFile("data/cifar-10-batches-bin/test_batch.bin");

	//500 epochs
	//Train : 74.2 %
	//Test  : 51.6 %
	nn.layers.push_back(new Linear(train_data[0].x.size(), 1000, batch_size));
	nn.layers.push_back(new ReLU(1000, 1000, batch_size));
	nn.layers.push_back(new Linear(1000, 1000, batch_size));
	nn.layers.push_back(new ReLU(1000, 1000, batch_size));
	nn.layers.push_back(new Linear(1000, 100, batch_size));
	nn.layers.push_back(new ReLU(100, 100, batch_size));
	nn.layers.push_back(new Linear(100, 10, batch_size));
	nn.layers.push_back(new Softmax(10, 10, batch_size));

	for (size_t e = 0; e < epochs; e++) {

		std::cout << "Epoch " << e + 1 << std::endl << std::endl;
		nn.train(train_data, learning_rate, weight_decay, train_data.size() / batch_size);

		double train_acc = nn.test(train_data);
		double test_acc = nn.test(test_data);

		std::cout << std::endl;
		std::cout << "Train : " << 100.0 * train_acc << " %" <<  std::endl;
		std::cout << "Test  : " << 100.0 * test_acc << " %" << std::endl;
	}

}
