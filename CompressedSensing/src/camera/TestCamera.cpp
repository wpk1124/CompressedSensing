#include "src/camera/TestCamera.h"
#include "src/utils/log.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>
#include <exception>

using namespace CS::camera;

TestCamera::TestCamera(int imageWidth, int imageHeight) : isGrabbing(false) {
	LOG_DEBUG("");
	this->imageWidth = imageWidth;
	this->imageHeight = imageHeight;
	loadImage();
}
TestCamera::~TestCamera() {
	LOG_DEBUG("joinable query: displayThread = "<<displayThread.joinable()<<" processing thread = "<<processingThread.joinable());
	if(displayThread.joinable()) {displayThread.join();}
	if(processingThread.joinable()) {processingThread.join();}
	LOG_DEBUG("");
}


void TestCamera::grab() {
	isGrabbing = true;
	displayThread = std::thread(&TestCamera::displayImage, this);
	processingThread = std::thread(&TestCamera::processImage, this);
	LOG_DEBUG("Started grabbing and displaying");
}

void TestCamera::stop() {
	isGrabbing = false;
	displayThread.join();
	processingThread.join();
	LOG_DEBUG("stopped gracefully");
}

void TestCamera::registerCallback(std::function<void (Frame& frame)> function) {
	processingFunction = function;
}

// private methods
void TestCamera::loadImage() {
	cv::Mat inputPic = cv::imread("D:/Programming/Git/CompressedSensing/CompressedSensing/pics/rihanna.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if(inputPic.data) {
		LOG_DEBUG("inputPic loaded");
	}else {
		throw std::runtime_error("Could not load test camera picture. Check if this pic exists in pics directory and if you call application from correct context");
	}
	internalPic = cv::Mat(imageHeight, imageWidth, CV_8UC1);
	LOG_DEBUG("input type = "<<inputPic.type() << "and internal = "<<internalPic.type());
	cv::resize(inputPic, internalPic, cv::Size(imageWidth, imageHeight));
}

void TestCamera::displayImage() {
	cv::namedWindow("display", CV_WINDOW_AUTOSIZE);
	while(isGrabbing) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		m.lock();
		cv::imshow("display", internalPic);
		m.unlock();

		cv::waitKey(5);
		m.lock();
		loadImage();
		m.unlock();
		isNewDataReady = true;
	}
	cv::destroyWindow("display");
	LOG_DEBUG("finished");
}

void TestCamera::processImage() {
	while(isGrabbing) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		int hash = std::this_thread::get_id().hash();
		LOG_DEBUG("Processing thread hash = "<<hash);
		if(isNewDataReady) {
			m.lock();
			Frame frame = matToFrame(internalPic);
			m.unlock();

			//simulate callback which has to sync itself with acquiring thread
			processingFunction(frame);
			isNewDataReady = false;
		}
	}

	LOG_DEBUG("finished");
}

Frame TestCamera::matToFrame(cv::Mat& image) {
	unsigned long long simulatedTimestamp = boost::posix_time::microsec_clock::local_time().time_of_day().total_milliseconds();
	return Frame(simulatedTimestamp, image.size().width, image.size().height, image.data);
}