#include "src/experiment/ExperimentHandler.h"
#include <Windows.h>
#include <chrono>
#include "src/utils/utils.h"
#include "src/utils/log.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace CS::experiment;
using namespace CS::camera;

ExperimentHandler::ExperimentHandler(std::shared_ptr<ICamera> _camera, ExperimentParameters& params): framesProcessed(0),
	framesToProcess((int)ceil(params.measurementRatio * params.imageHeight * params.imageWidth)), isMeasurementEnded(false), parameters(params.measurementRatio, params.imageWidth, params.imageHeight) {
	this->parameters = params;
	singlePixelCameraOutput = cv::Mat(framesToProcess, 1, CV_32S);
	camera = std::move(_camera);
	camera->registerCallback(std::bind(&ExperimentHandler::simulateSinglePixelCamera, this, std::placeholders::_1));
}

ExperimentHandler::~ExperimentHandler() {
	LOG_DEBUG("");
}

void ExperimentHandler::handleExperiment() {
	auto cameraOutput = gatherMeasurements();
	cv::Mat x0 = computeStartingSolution(std::get<0>(cameraOutput), std::get<1>(cameraOutput));
	debugImageShow(x0);
}

//private methods

void ExperimentHandler::debugImageShow(const cv::Mat& image) {
	cv::Mat imShow = image.clone();
	LOG_DEBUG("image size = ("<<image.size().height<<","<<image.size().width<<")");
	cv::resize(imShow, imShow, cv::Size(parameters.imageHeight, parameters.imageWidth), 0, 0, cv::INTER_CUBIC);
	LOG_DEBUG("resized image size = ("<<imShow.size().height<<","<<imShow.size().width<<")");
	cv::namedWindow("starting solution", cv::WINDOW_AUTOSIZE);
	cv::imshow("starting solution", imShow);
	LOG_WARNING("Close window to continue execution!");
	cv::waitKey();
}

std::tuple<cv::Mat&, cv::Mat&> ExperimentHandler::gatherMeasurements() {
	measurementMatrix = cv::Mat(framesToProcess, parameters.imageHeight * parameters.imageWidth, CV_32FC1);
	gpuSolver.createBinaryMeasurementMatrix(framesToProcess, parameters.imageHeight * parameters.imageWidth, &measurementMatrix);
	
	int hash = (int)std::this_thread::get_id().hash();
	LOG_DEBUG("this_thread_hash = "<<hash);
	camera->grab();
	waitUntilGrabbingFinished();
	camera->stop();

	return std::make_tuple(std::ref(measurementMatrix), std::ref(singlePixelCameraOutput));
}

cv::Mat ExperimentHandler::computeStartingSolution(cv::Mat& measurementMatrix, cv::Mat& cameraOutput) {
	//starting solution = min energy
	return gpuSolver.transProduct(measurementMatrix, cameraOutput);
}

void ExperimentHandler::waitUntilGrabbingFinished() {
	{
		std::unique_lock<std::mutex> lock(measurementMutex);
		cv.wait(lock, [this] {return isMeasurementEnded;});
	}
	LOG_DEBUG("Grabbing finished!");
}

void ExperimentHandler::simulateSinglePixelCamera(const Frame& frame) {
	if(framesProcessed < framesToProcess) {
		image = frame.image.reshape(1, 1);
		
		LOG_DEBUG("frame.image size"<<frame.image.size()<<" and internal size " <<image.size()<<"types = "<<frame.image.type()<<" and "<<image.type());
		
		singlePixelCameraOutput.at<float>(framesProcessed,0) = (float)measurementMatrix.row(framesProcessed).dot(image);
		framesProcessed++;

		LOG_DEBUG("hash = "<<(int)std::this_thread::get_id().hash());
		LOG_DEBUG("size measMat.row("<<framesProcessed-1<<") = "<<measurementMatrix.row(framesProcessed-1).size() << "imageTaken size = "<<image.size().height * image.size().width);
		LOG_DEBUG("Frames processed = "<<framesProcessed<<"output["<<framesProcessed-1<<"] = "<<singlePixelCameraOutput.at<float>(framesProcessed-1,0));
		if(framesProcessed >= framesToProcess) {
			notifyMeasurementEnded();
			LOG_DEBUG("Experiment ended. Frames processed = "<<framesProcessed<<" toProcess = "<<framesToProcess);
		}
	}
}

void ExperimentHandler::notifyMeasurementEnded() {
	{
		std::unique_lock<std::mutex> lock(measurementMutex);
		isMeasurementEnded = true;
	}
	cv.notify_one();
}