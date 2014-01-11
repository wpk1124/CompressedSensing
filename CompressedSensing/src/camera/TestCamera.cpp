#include "src/camera/TestCamera.h"
#include "src/utils/log.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>
#include <exception>

using namespace CS::camera;

TestCamera::TestCamera(int imageWidth, int imageHeight) : isGrabbing(false), fps(100) {
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
	isGrabbing.store(true);
	displayThread = std::thread(&TestCamera::displayImage, this);
	processingThread = std::thread(&TestCamera::processImage, this);
	LOG_DEBUG("Started grabbing and displaying");
}

void TestCamera::stop() {
	isGrabbing.store(false);
	displayThread.join();
	processingThread.join();
	LOG_DEBUG("stopped gracefully");
}

void TestCamera::registerCallback(std::function<void (const Frame& frame)> function) {
	processingFunction = function;
}

// private methods
void TestCamera::loadImage() {
	//"D:/Programming/Git/CompressedSensing/CompressedSensing/pics/rihanna.jpg"
	cv::Mat inputPic = cv::imread("C:/GitHub/CompressedSensing/CompressedSensing/pics/rihanna.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if(inputPic.data) {
		LOG_DEBUG("inputPic loaded");
	}else {
		throw std::runtime_error("Could not load test camera picture. Check if this pic exists in pics directory and if you call application from correct context");
	}
	internalPic = cv::Mat(imageHeight, imageWidth, CV_32FC1);
	LOG_DEBUG("input type = "<<inputPic.type() << "and internal = "<<internalPic.type());
	cv::resize(inputPic, inputPic, cv::Size(imageWidth, imageHeight));
	inputPic.convertTo(internalPic, internalPic.type(), 1/255.0);
}

void TestCamera::displayImage() {
	cv::namedWindow("display", CV_WINDOW_AUTOSIZE);
	while(isGrabbing.load()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1000/fps));
		m.lock();
		cv::imshow("display", internalPic);
		m.unlock();

		cv::waitKey(1);
		m.lock();
		loadImage();
		m.unlock();

		notifyNewFrameReady();
	}
	cv::destroyWindow("display");
	LOG_DEBUG("finished");
}

void TestCamera::processImage() {
	while(isGrabbing) {
		waitForNewFrame();
		
		m.lock();
		long simulatedTimestamp = (long)boost::posix_time::microsec_clock::local_time().time_of_day().total_milliseconds();
		Frame frame(internalPic, simulatedTimestamp);
		m.unlock();

		//simulate callback which has to sync itself with acquiring thread
		processingFunction(frame);
		resetNewFrameReady();
	}

	LOG_DEBUG("Processing thread hash = "<<(int)std::this_thread::get_id().hash()<<" finished");
}

void TestCamera::waitForNewFrame() {
	{
		std::unique_lock<std::mutex> lock(dataReadyMutex);
		dataReadyCV.wait(lock, [this]{return isNewDataReady;});
	}
	LOG_DEBUG("New frame arrived");
}

void TestCamera::notifyNewFrameReady() {
	{
		std::unique_lock<std::mutex> lock(dataReadyMutex);
		isNewDataReady = true;
	}
	dataReadyCV.notify_one();
	LOG_DEBUG("New frame dispatched");
}

void TestCamera::resetNewFrameReady() {
	{
		std::unique_lock<std::mutex> lock(dataReadyMutex);
		isNewDataReady = false;
	}
}