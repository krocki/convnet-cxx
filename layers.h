/*
* @Author: kmrocki
* @Date:   2016-02-24 15:26:54
* @Last Modified by:   kmrocki
* @Last Modified time: 2016-03-14 16:27:11
*/

#ifndef __LAYERS_H__
#define __LAYERS_H__

#include <utils.h>
#include <io.h>
#include <string>

//abstract
class Layer {

	public:
	
		const std::string name;
		
		//used in forward pass
		Matrix x; //inputs
		Matrix y; //outputs
		
		//grads, used in backward pass
		Matrix dx;
		Matrix dy;
		
		Layer ( size_t inputs, size_t outputs, size_t batch_size,
				std::string _label = "layer" ) : name ( _label ) {
				
			x = Matrix ( inputs, batch_size );
			y = Matrix ( outputs, batch_size );
			dx = Matrix ( inputs, batch_size );
			dy = Matrix ( outputs, batch_size );
			
		};
		
		//need to override these
		
		/*		TODO: allow 2 versions of forward (with 0 and 1 params),
				currently you must override version with 1 				*/
		
		virtual void forward ( bool test = false ) = 0;
		virtual void backward() = 0;
		
		//this is mainly for debugging - TODO: proper serialization of layers and object NN
		virtual void save_to_files ( std::string prefix ) {
		
			save_matrix_to_file ( x, prefix + "_x" );
			save_matrix_to_file ( y, prefix + "_y" );
			save_matrix_to_file ( dx, prefix + "_dx" );
			save_matrix_to_file ( dy, prefix + "_dy" );
			
		};
		
		virtual void resetGrads() {};
		virtual void applyGrads ( double alpha, double decay ) {};
		
		virtual ~Layer() {};
};


class Padding : public Layer {

	public:
	
		size_t channels;
		size_t padding;
		
		void forward ( bool test = false ) {
		
			pad ( x, y, channels, padding );
			
		}
		
		void backward() {
		
			unpad ( dx, dy, channels, padding );
			
		}
		
		Padding ( size_t inputs, size_t _channels, size_t batch_size, size_t _padding ) :
			Layer ( inputs, _channels * ( sqrt ( inputs / _channels ) + _padding * 2 ) * ( sqrt (
						inputs / _channels ) + _padding * 2 ), batch_size, "padding" ),
			padding ( _padding ),  channels ( _channels ) {};
		~Padding() {};
		
};

class Dropout : public Layer {

	public:
	
		const float keep_ratio;
		Matrix dropout_mask;
		
		void forward ( bool test = false ) {
		
			if ( test ) // skip at test time
			
				y = x;
				
			else {
			
				Matrix rands = Matrix::Zero ( y.rows(), y.cols() );
				rand ( rands, 0.0f, 1.0f );
				
				//dropout mask - 1s - preserved elements
				dropout_mask = ( rands.array() < keep_ratio ).cast <float> ();
				
				// y = y .* dropout_mask, discard elements where mask is 0
				y.array() = x.array() * dropout_mask.array();
				
				// normalize, so that we don't have to do anything at test time
				y /= keep_ratio;
				
			}
		}
		
		void backward() {
		
			dx.array() = dy.array() * dropout_mask.array();
		}
		
		Dropout ( size_t inputs, size_t outputs, size_t batch_size, float _ratio ) :
			Layer ( inputs, outputs, batch_size, "dropout" ),  keep_ratio ( _ratio ) {};
		~Dropout() {};
		
};

class Pooling : public Layer {

	public:
	
		void forward ( bool test = false ) {
		
			pooling_forward ( channels, x, y, cache, window_size );
			
		}
		
		void backward() {
		
			pooling_backward ( channels, dx, dy, cache, window_size );
			
		}
		
		//this is mainly for debugging - TODO: proper serialization of layers
		
		Pooling ( size_t inputs, size_t _window_size, size_t _channels, size_t batch_size ) :
			Layer ( inputs,
					_channels * ( sqrt ( inputs / _channels ) / _window_size ) * ( sqrt ( inputs / _channels ) / _window_size ),
					batch_size, "pool" ),
			channels ( _channels ), window_size ( _window_size ) {
			
			cache = Matrix::Zero ( x.rows(), x.cols() );
			
		};
		
		~Pooling() {};
		
		Matrix cache;
		
		const size_t channels;
		const size_t window_size;
};

class Convolution : public Layer {

	public:
	
		Matrix W;
		Vector b;
		
		Matrix dW;
		Matrix db;
		
		Matrix mW;
		Matrix mb;
		
		const size_t input_channels;
		const size_t output_channels;
		const size_t kernel_size;
		const size_t output_map_size;
		
		void forward ( bool test = false ) {
		
			//pad(x, x_padded, kernel_size, input_channels);
			convolution_forward_gemm ( input_channels, y, W, x, b );
			
		}
		
		void backward() {
		
			dx.setZero();
			//dW
			convolution_backward_gemm ( input_channels, dy, dW, x, db );
			//dx
			convolution_backward_full_gemm ( input_channels, dy, W, dx );
		}
		
		//this is mainly for debugging - TODO: proper serialization of layers
		void save_to_files ( std::string prefix ) {
		
			save_matrix_to_file ( W, prefix + "_W" );
			save_matrix_to_file ( b, prefix + "_b" );
			Layer::save_to_files ( prefix );
			
		}
		
		Convolution ( size_t inputs, size_t channels, size_t filter_size, size_t filters, size_t batch_size ) :
			Layer ( inputs * channels, filters * ( sqrt ( inputs ) - filter_size + 1 ) * ( sqrt ( inputs ) - filter_size + 1 ),
					batch_size, "conv" ),
			input_channels ( channels ), kernel_size ( filter_size ), output_channels ( filters ),
			output_map_size ( ( sqrt ( inputs ) - filter_size + 1 ) * ( sqrt ( inputs ) - filter_size + 1 ) ) {
			
			
			W = Matrix ( filters, filter_size * filter_size * input_channels );
			b = Vector::Zero ( filters );
			
			//W << 0.1, 0, 0, 0, 0, 0, 0, 0, 0;
			size_t fan_in = channels * filter_size * filter_size;
			size_t fan_out = filters * filter_size * filter_size;
			double range = sqrt ( 6.0 / double ( fan_in + fan_out ) );
			
			rand ( W, -range, range );
			
			mW = Matrix::Zero ( W.rows(), W.cols() );
			mb = Vector::Zero ( b.rows() );
			
		};
		
		void resetGrads() {
		
			dW = Matrix::Zero ( W.rows(), W.cols() );
			db = Vector::Zero ( b.rows() );;
			
		}
		
		void applyGrads ( double alpha, double decay = 0 ) {
		
			//adagrad
			mW += dW.cwiseProduct ( dW );
			mb += db.cwiseProduct ( db );
			
			W = ( 1 - decay ) * W + alpha * dW.cwiseQuotient ( mW.unaryExpr ( std::ptr_fun ( sqrt_eps ) ) );
			b += alpha * db.cwiseQuotient ( mb.unaryExpr ( std::ptr_fun ( sqrt_eps ) ) );
			
		}
		
		~Convolution() {};
		
};

class Linear : public Layer {

	public:
	
		Matrix W;
		Vector b;
		
		Matrix dW;
		Matrix db;
		
		Matrix mW;
		Matrix mb;
		
		void forward ( bool test = false ) {
		
			y = b.replicate ( 1, x.cols() );
			BLAS_mmul ( y, W, x );
			
		}
		
		void backward() {
		
			dW.setZero();
			BLAS_mmul ( dW, dy, x, false, true );
			//dW = dy * x.transpose();
			db = dy.rowwise().sum();
			dx.setZero();
			BLAS_mmul ( dx, W, dy, true, false );
			//dx = W.transpose() * dy;
			
		}
		
		Linear ( size_t inputs, size_t outputs, size_t batch_size ) :
			Layer ( inputs, outputs, batch_size, "fc" ) {
			
			W = Matrix ( outputs, inputs );
			b = Vector::Zero ( outputs );
			double range = sqrt ( 6.0 / double ( inputs + outputs ) );
			
			rand ( W, -range, range );
			
			mW = Matrix::Zero ( W.rows(), W.cols() );
			mb = Vector::Zero ( b.rows() );
			
		};
		
		void resetGrads() {
		
			dW = Matrix::Zero ( W.rows(), W.cols() );
			db = Vector::Zero ( b.rows() );
		}
		
		void applyGrads ( double alpha, double decay = 0 ) {
		
			//adagrad
			mW += dW.cwiseProduct ( dW );
			mb += db.cwiseProduct ( db );
			
			W = ( 1 - decay ) * W + alpha * dW.cwiseQuotient ( mW.unaryExpr ( std::ptr_fun ( sqrt_eps ) ) );
			b += alpha * db.cwiseQuotient ( mb.unaryExpr ( std::ptr_fun ( sqrt_eps ) ) );
			
			// 'plain' fixed learning rate update
			// b += alpha * db;
			// W += alpha * dW;
			
		}
		
		void save_to_files ( std::string prefix ) {
		
			save_matrix_to_file ( W, prefix + "_W" );
			save_matrix_to_file ( b, prefix + "_b" );
			Layer::save_to_files ( prefix );
			
		}
		
		~Linear() {};
		
};

class Sigmoid : public Layer {

	public:
	
		void forward ( bool test = false ) {
		
			y = logistic ( x );
			
		}
		
		void backward() {
		
			dx.array() = dy.array() * y.array() * ( 1.0 - y.array() ).array();
			
		}
		
		Sigmoid ( size_t inputs, size_t outputs, size_t batch_size ) :
			Layer ( inputs, outputs, batch_size, "sigmoid" ) {};
		~Sigmoid() {};
		
};

class Identity : public Layer {

	public:
	
		void forward ( bool test = false ) {
		
			y = x;
			
		}
		
		void backward() {
		
			dx = dy;
			
		}
		
		Identity ( size_t inputs, size_t outputs, size_t batch_size ) :
			Layer ( inputs, outputs, batch_size, "noop" ) {};
		~Identity() {};
		
};

class ReLU : public Layer {

	public:
	
		void forward ( bool test = false ) {
		
			y = rectify ( x );
			
		}
		
		void backward() {
		
			dx.array() = derivative_ReLU ( y ).array() * dy.array();
			
		}
		
		ReLU ( size_t inputs, size_t outputs, size_t batch_size ) :
			Layer ( inputs, outputs, batch_size, "relu" ) {};
		~ReLU() {};
		
};

// Exponential Linear Unit
// http://arxiv.org/pdf/1511.07289v5.pdf

class ELU : public Layer {

	public:
	
		void forward ( bool test = false ) {
		
			y = activation_ELU ( x );
			
		}
		
		void backward() {
		
			dx.array() = derivative_ELU ( y ).array() * dy.array();
			
		}
		
		ELU ( size_t inputs, size_t outputs, size_t batch_size ) :
			Layer ( inputs, outputs, batch_size, "elu" ) {};
		~ELU() {};
		
};

class Softmax : public Layer {

	public:
	
		void forward ( bool test = false ) {
		
			y = softmax ( x );
			
		}
		
		void backward() {
		
			dx = dy - y;
		}
		
		
		Softmax ( size_t inputs, size_t outputs, size_t batch_size ) :
			Layer ( inputs, outputs, batch_size, "softmax" ) {};
		~Softmax() {};
		
};

#endif
