/*
* @Author: kmrocki
* @Date:   2016-02-24 09:43:05
* @Last Modified by:   kmrocki
* @Last Modified time: 2016-03-17 09:29:14
*/

#include <importer.h>
#include <utils.h>
#include <layers.h>
#include <nn.h>
#include <iostream>
#include <utils.h>
#include <cblas.h>

int main() {

	//using OpenMP - limit OpenBLAS to 1 core
	openblas_set_num_threads(1);

	// load CIFAR-10 data
	std::deque<datapoint> train_data, partial_data;
	partial_data = CIFARImporter::importFromFile("data/cifar-10-batches-bin/data_batch_1.bin", 1);
	train_data.insert(train_data.end(), partial_data.begin(), partial_data.end());
	partial_data = CIFARImporter::importFromFile("data/cifar-10-batches-bin/data_batch_2.bin", 1);
	train_data.insert(train_data.end(), partial_data.begin(), partial_data.end());
	partial_data = CIFARImporter::importFromFile("data/cifar-10-batches-bin/data_batch_3.bin", 1);
	train_data.insert(train_data.end(), partial_data.begin(), partial_data.end());
	partial_data = CIFARImporter::importFromFile("data/cifar-10-batches-bin/data_batch_4.bin", 1);
	train_data.insert(train_data.end(), partial_data.begin(), partial_data.end());
	partial_data = CIFARImporter::importFromFile("data/cifar-10-batches-bin/data_batch_5.bin", 1);
	train_data.insert(train_data.end(), partial_data.begin(), partial_data.end());

	std::deque<datapoint> test_data =
		CIFARImporter::importFromFile("data/cifar-10-batches-bin/test_batch.bin", 1);

	//raw pixel -> softmax: ~39% (10 epochs)
	// no padding, no decay, no data aug, just dropout
	// Train : 93.96 %
	// Test  : 81.55 %
	// 100 epochs
	/*
		size_t 	classes = 10;
		size_t 	input_channels = 3;
		size_t 	filter_size[] = {3, 3, 3, 3, 2};
		size_t 	filters[] = {32, 32, 128, 128, 512};
		size_t  fully_connected_size = 1024;

		size_t 	pooling_window = 2;
		float 	dropout = 0.5f;

		size_t 	epochs = 500;
		size_t 	batch_size = 16;
		double 	learning_rate = 1e-2;
		double 	weight_decay = 0;

		NN nn(batch_size);

		//CONV 3x3 -> CONV 3x3 -> POOL 2x
		nn.layers.push_back(new Convolution(train_data[0].x.rows() / input_channels, input_channels, filter_size[0], filters[0], batch_size));
		nn.layers.push_back(new ReLU(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size));
		nn.layers.push_back(new Convolution(nn.layers.back()->y.rows() / filters[0], filters[0], filter_size[1], filters[1], batch_size));
		nn.layers.push_back(new ReLU(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size));
		nn.layers.push_back(new Pooling(nn.layers.back()->y.rows(), pooling_window, filters[1], batch_size));
		nn.layers.push_back(new Dropout(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size, dropout));

		//CONV 3x3 -> CONV 3x3 -> POOL 2x
		nn.layers.push_back(new Convolution(nn.layers.back()->y.rows() / filters[1], filters[1], filter_size[2], filters[2], batch_size));
		nn.layers.push_back(new ReLU(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size));
		nn.layers.push_back(new Convolution(nn.layers.back()->y.rows() / filters[2], filters[2], filter_size[3], filters[3], batch_size));
		nn.layers.push_back(new ReLU(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size));
		nn.layers.push_back(new Pooling(nn.layers.back()->y.rows(), pooling_window, filters[3], batch_size));
		nn.layers.push_back(new Dropout(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size, dropout));

		//FULLY CONNECTED
		nn.layers.push_back(new Linear(nn.layers.back()->y.rows(), fully_connected_size, batch_size));
		nn.layers.push_back(new ReLU(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size));
		nn.layers.push_back(new Dropout(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size, dropout));

		//SOFTMAX
		nn.layers.push_back(new Linear(nn.layers.back()->y.rows(), classes, batch_size));
		nn.layers.push_back(new Softmax(classes, classes, batch_size));
	*/

	/* with padding (1) */
	/*
		no dropout, 3x3x32 -> 3x3x32 -> Pool -> 3x3x128 -> 3x3x128 -> Pool -> 3x3x128 -> 3x3x128 -> Pool > fc256 -> softmax
		Train : 98.46 %
		Test  : 77.90 %

	*/

	/* with padding (1) */
	/* no fc

	Train : 97.68 %
	Test  : 84.60 %
	500 epochs

	*/

	size_t 	classes = 10;
	size_t 	input_channels = 3;
	size_t 	filter_size[] = {3, 3, 3, 3, 3, 3};
	size_t 	filters[] = {32, 32, 64, 64, 192, 192};
	size_t  fully_connected_size = 1024;

	size_t 	pooling_window = 2;
	float 	dropout = 0.5f;

	size_t 	epochs = 500;
	size_t 	batch_size = 16;
	double 	learning_rate = 1e-2;
	double 	weight_decay = 0;

	NN nn(batch_size);

	//CONV 3x3 -> CONV 3x3 -> POOL 2x
	nn.layers.push_back(new Padding(train_data[0].x.rows(), input_channels, batch_size, 1));
	nn.layers.push_back(new Convolution(nn.layers.back()->y.rows() / input_channels, input_channels, filter_size[0], filters[0], batch_size));
	nn.layers.push_back(new ReLU(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size));

	nn.layers.push_back(new Padding(nn.layers.back()->y.rows(), filters[0], batch_size, 1));
	nn.layers.push_back(new Convolution(nn.layers.back()->y.rows() / filters[0], filters[0], filter_size[1], filters[1], batch_size));
	nn.layers.push_back(new ReLU(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size));

	nn.layers.push_back(new Pooling(nn.layers.back()->y.rows(), pooling_window, filters[1], batch_size));
	nn.layers.push_back(new Dropout(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size, dropout));

	//CONV 3x3 -> CONV 3x3 -> POOL 2x
	nn.layers.push_back(new Padding(nn.layers.back()->y.rows(), filters[1], batch_size, 1));
	nn.layers.push_back(new Convolution(nn.layers.back()->y.rows() / filters[1], filters[1], filter_size[2], filters[2], batch_size));
	nn.layers.push_back(new ReLU(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size));

	nn.layers.push_back(new Padding(nn.layers.back()->y.rows(), filters[2], batch_size, 1));
	nn.layers.push_back(new Convolution(nn.layers.back()->y.rows() / filters[2], filters[2], filter_size[3], filters[3], batch_size));
	nn.layers.push_back(new ReLU(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size));

	nn.layers.push_back(new Pooling(nn.layers.back()->y.rows(), pooling_window, filters[3], batch_size));
	nn.layers.push_back(new Dropout(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size, dropout));

	//CONV 3x3 -> CONV 3x3 -> POOL 2x
	nn.layers.push_back(new Padding(nn.layers.back()->y.rows(), filters[3], batch_size, 1));
	nn.layers.push_back(new Convolution(nn.layers.back()->y.rows() / filters[3], filters[3], filter_size[4], filters[4], batch_size));
	nn.layers.push_back(new ReLU(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size));

	nn.layers.push_back(new Padding(nn.layers.back()->y.rows(), filters[4], batch_size, 1));
	nn.layers.push_back(new Convolution(nn.layers.back()->y.rows() / filters[4], filters[4], filter_size[5], filters[5], batch_size));
	nn.layers.push_back(new ReLU(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size));

	nn.layers.push_back(new Pooling(nn.layers.back()->y.rows(), pooling_window, filters[5], batch_size));
	nn.layers.push_back(new Pooling(nn.layers.back()->y.rows(), pooling_window, filters[5], batch_size));

	nn.layers.push_back(new Dropout(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size, dropout));

	//FULLY CONNECTED
	//nn.layers.push_back(new Linear(nn.layers.back()->y.rows(), fully_connected_size, batch_size));
	//nn.layers.push_back(new ReLU(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size));
	//nn.layers.push_back(new Dropout(nn.layers.back()->y.rows(), nn.layers.back()->y.rows(), batch_size, dropout));

	//SOFTMAX
	nn.layers.push_back(new Linear(nn.layers.back()->y.rows(), classes, batch_size));
	nn.layers.push_back(new Softmax(classes, classes, batch_size));


	for (size_t e = 0; e < epochs; e++) {

		std::cout << "Epoch " << e + 1 << std::endl << std::endl;
		nn.train(train_data, learning_rate, weight_decay, train_data.size() / batch_size, classes);

		nn.save_to_files("out/c10_conv_big_fc");

		double train_acc = nn.test(train_data, classes);
		double test_acc = nn.test(test_data, classes);

		// std::cout << std::endl;
		std::cout << std::setprecision(2) << "Train : " << 100.0 * train_acc << " %" <<  std::endl;
		std::cout << std::setprecision(2) << "Test  : " << 100.0 * test_acc << " %" << std::endl;

	}

}
