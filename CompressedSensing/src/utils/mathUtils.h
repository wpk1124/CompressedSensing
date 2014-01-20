#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#define BOOST_UBLAS_NDEBUG

#include <opencv2/opencv.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <vector>

using namespace boost::numeric;

namespace CS {
namespace math {

	class MathUtils {
	public:
		static void randperm(int n, int* vec);
		static std::vector<std::vector<float>> matToStdMatrix(const cv::Mat& A);
		static ublas::matrix<float> matToBoostMatrix(const cv::Mat& A);
		static void normalizeImage(cv::Mat& input);
		std::vector<int> findIndt(cv::Mat& dqt, cv::Mat& tsols);
	};

}} //namespace end

#endif /*MATH_UTILS_H*/