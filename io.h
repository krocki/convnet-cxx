/*
* @Author: kmrocki
* @Date:   2016-02-26 16:38:27
* @Last Modified by:   kmrocki
* @Last Modified time: 2016-03-08 16:03:45
*/

#include <iostream>
#include <string>

#ifndef __IO_H__
#define __IO_H__

void save_matrix_to_file(Eigen::MatrixXf& m, std::string filename) {

	filename = filename + "_" + std::to_string(m.rows()) + "x" + std::to_string(m.cols()) + ".txt";
	std::cout << "Saving a matrix to " << filename << "... " << std::endl;
	std::ofstream file(filename.c_str());

	if (file.is_open()) {

		file << m;
		file.close();

	} else {

		std::cout << "file save error: (" << filename << ")" << std::endl;

	}

}

void save_matrix_to_file(Eigen::VectorXf& m, std::string filename) {

	filename = filename + "_" + std::to_string(m.rows()) + "x" + std::to_string(m.cols()) + ".txt";
	std::cout << "Saving a matrix to " << filename << "... " << std::endl;

	std::ofstream file(filename.c_str());

	if (file.is_open()) {

		file << m;
		file.close();

	} else {

		std::cout << "file save error: (" << filename << ")" << std::endl;

	}

}

#endif