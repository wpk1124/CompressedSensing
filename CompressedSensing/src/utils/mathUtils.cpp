#include "src/utils/mathUtils.h"
#include "src/utils/log.h"
#include <cstdlib>

void CS::math::MathUtils::randperm(int n, int* perm) {
	int i, j, t;

	for(i=0; i<n; i++)
		perm[i] = i;
	for(i=0; i<n; i++) {
		j = rand()%(n-i)+i;
		t = perm[j];
		perm[j] = perm[i];
		perm[i] = t;
	}
}

std::vector<std::vector<float>> CS::math::MathUtils::matToStdMatrix(const cv::Mat& A) {
    int height = A.rows;
    int width = A.cols;
	std::vector<std::vector<float>> stdMatrix(height);
    for(int i = 0; i < height; i++) {
		stdMatrix[i].resize(width);
        //tried iterators, got exceptions :(
        for(int j = 0; j < width; j++) {
			stdMatrix[i][j] = A.at<float>(i,j);
		}
	}

	return stdMatrix;
}

boost::numeric::ublas::matrix<float> CS::math::MathUtils::matToBoostMatrix(const cv::Mat& A) {
	using namespace boost::numeric::ublas;

	unbounded_array<float> storage(A.size().height * A.size().width);
	std::copy(A.begin<float>(), A.end<float>(), storage.begin());

	return matrix<float>(A.size().height, A.size().width, storage);
}

void CS::math::MathUtils::normalizeImage(cv::Mat& input) {
	input -= cv::norm(input, cv::NORM_L2);
	input -= cv::mean(input);
}

std::vector<int> CS::math::MathUtils::findIndt(cv::Mat& dqt, cv::Mat& tsols) {
	std::vector<int> output;
	for(int i = 0; i < dqt.rows; i++) {
		for(int j = 0; j < dqt.cols; j++) {
			if(dqt.at<float>(i,j) && (tsols.at<float>(i,j) > 0))
				output.push_back(i * dqt.rows + j);
		}
   }
   return output;
}