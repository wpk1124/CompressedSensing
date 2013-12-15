#include "src/camera/TestCamera.h"
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <iostream>

using namespace CS::camera;

TestCamera::TestCamera(int imageWidth, int imageHeight, double measurementRatio) {BOOST_LOG_TRIVIAL(info) << "TestCamera construction\n";}
TestCamera::~TestCamera() {BOOST_LOG_TRIVIAL(info) << "TestCamera de-construction\n";}

cv::Mat& TestCamera::gatherMeasurements() {
	outFile = cv::Mat::ones(3,3, CV_8UC1);
	return outFile;
}