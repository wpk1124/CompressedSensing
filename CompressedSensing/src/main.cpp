#define __CL_ENABLE_EXCEPTIONS

#include <thread>
#include <memory>
#include <iostream>
#include <CL/cl.hpp>
#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>

#include "src/utils/log.h"
#include "src/utils/utils.h"
#include "src/exceptions/Exceptions.h"
#include "src/camera/CameraFactory.h"
#include "src/algorithm/AlgorithmFactory.h"
#include "src/gpu/GPUSolver.h"
#include "src/experiment/ExperimentHandler.h"

#include "src/gpu/GPUSolver.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace CS;
using namespace CS::camera;
using namespace CS::algorithm;
using namespace CS::experiment;
namespace opts = boost::program_options;

#ifdef _DEBUG
#pragma comment (lib, "opencv_highgui244d.lib")
#pragma comment (lib, "opencv_core244d.lib")
#pragma comment (lib, "opencv_imgproc244d.lib")
#else
#pragma comment (lib, "opencv_highgui244.lib")
#pragma comment (lib, "opencv_core244.lib")
#pragma comment (lib, "opencv_imgproc244d.lib")
#endif

void tinyTest() {
	CS::gpu::GPUSolver solver;
	float data[256];
	float dataY[16];

	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 16; j++) {
			data[i*16 + j] = 1;
			if(i == j)
				data[i*16 + j] = 10;
		}
		dataY[i] = 10*(i+1);
	}
	cv::Mat A(16,16, CV_32FC1, &data);
	cv::Mat y(16,1, CV_32FC1, &dataY);

	cv::Mat output = solver.linsolve(A, y);
}

int main(int argc, char **argv) {
	double measurementRatio = 0.0;
	bool verbose = false;
	string cameraName, algorithmName;

	try {
		LOG_INFO("Application start");
		LOG_DEBUG("main_thread_hash = "<<(int)std::this_thread::get_id().hash());
		opts::options_description desc("Allowed options");
		opts::variables_map vm = utils::setAndRunCommandLineArgs(argc, argv, &measurementRatio, &verbose, &cameraName, &algorithmName, desc);

		if(vm.count("help")) {
			cout << desc << "\n";
			return 0;
		}

		tinyTest();

		utils::setLoggingParameters(verbose);

		int imageWidth = vm["image-width"].as<int>();
		int imageHeight = vm["image-height"].as<int>();

		utils::validateInputArguments(imageWidth, imageHeight, measurementRatio);
		std::shared_ptr<ICamera> pCamera(CameraFactory::getInstance(cameraName, imageWidth, imageHeight));
		std::shared_ptr<ICSAlgorithm> pAlgorithm(AlgorithmFactory::getInstance(algorithmName));
		ExperimentParameters params(measurementRatio, imageWidth, imageHeight);

		ExperimentHandler handler(pCamera, pAlgorithm, params);
		handler.handleExperiment();

		LOG_INFO("Application successful exit");
	}catch(std::bad_alloc& e) {
		cerr << "Probably not enough memory: "<<e.what() << std::endl;
	}catch(CS::exception::JaiCameraException& e) {
		cerr << "Exception in JaiCamera: " << e.what() << std::endl;
	}catch(CS::exception::UnknownTypeException& e) {
		cerr << "type_error: "<<e.what() << std::endl;
	}catch(std::range_error& e) {
		cerr << "range_error: "<<e.what() << std::endl;
	}catch(std::runtime_error& e) {
		cerr << "runtime_error: "<<e.what() << std::endl;
	}catch(std::exception& e) {
		cerr << "error: "<<e.what() << std::endl;
	}catch(...) {
		cerr << "Exception of unknown type!\n";
	}

	return 0;
}