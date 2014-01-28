
#include "gtest/gtest.h"
#include "test/testutils/TestUtils.h"
#include "src/algorithm/TvqcAlgorithm.h"

namespace CS {
namespace test {
	void TestTvqc() {
		//gathering Matlab sample data
		cv::Mat A = cv::imread("A.bmp", CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat b = cv::imread("b.bmp", CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat x0 = cv::imread("x0.bmp", CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat xp = cv::imread("xp.bmp", CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat measurementMatrix = cv::Mat(A.rows, A.cols, CV_32FC1);
		cv::Mat observations = cv::Mat(b.rows, b.cols, CV_32FC1);
		cv::Mat startingSolution = cv::Mat(x0.rows, x0.cols, CV_32FC1);
		cv::Mat MatlabResult = cv::Mat(xp.rows, xp.cols, CV_32FC1);
		A.convertTo(measurementMatrix, measurementMatrix.type(), 1/255.0);
		b.convertTo(observations, observations.type(), 1/255.0);
		x0.convertTo(startingSolution, startingSolution.type(), 1/255.0);
		xp.convertTo(MatlabResult, MatlabResult.type(), 1/255.0);

		//running TVQC on sample data
		cv::Mat TestResult = cv::Mat(xp.rows, xp.cols, CV_32FC1);
		CS::algorithm::TvqcAlgorithm TestAlgorithm;
		TestResult = TestAlgorithm.recoverImage(measurementMatrix, observations, startingSolution);

		//displaying image difference
		cv::Mat error = cv::abs(MatlabResult - TestResult);
		cv::Mat draw;
		double maxValue, minValue;
		cv::minMaxLoc(error, &minValue, &maxValue);
		error.convertTo(draw, CV_8U, 255.0/(maxValue - minValue), -minValue *255.0/(maxValue - minValue));
		cv::namedWindow("debug-window", CV_WINDOW_AUTOSIZE);
		cv::imshow("debug-window", draw);
	}
}}

