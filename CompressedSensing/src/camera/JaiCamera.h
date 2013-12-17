#ifndef JAI_CAMERA_H
#define JAI_CAMERA_H

#include "src/camera/ICamera.h"
#include <string>
#include <stdint.h>
#include <Jai_Factory.h>

class cv::Mat;

namespace CS {
namespace camera {

class JaiCamera : public ICamera {
public:
	JaiCamera(int, int);
	~JaiCamera();

	cv::Mat& gatherMeasurements();
	void grab();
	void stop();

	void setCallback(void (*callbackFunction)(void *context));
private:
	void (*callbackFunction)(J_tIMAGE_INFO *pAqImageInfo);
	bool measurementComplete;
	cv::Mat measurementMatrix;
	int currentRow;
	int nRows, nCols;

	CAM_HANDLE camHandle;
	FACTORY_HANDLE factoryHandle;
	VIEW_HANDLE viewHandle;
	THRD_HANDLE hThread;

	int64_t getParameter(std::string paramName);
	int64_t getSizeOfBuffer();

	void validator(const char *, J_STATUS_TYPE *);
	void openLiveViewStream();
	void openCameraOfIndex(int index);
	void openStream();
	void openMeasurementStream();
	void waitUntilMeasurementFinished();
	void streamCBFunc(J_tIMAGE_INFO *pAqImageInfo);
	void getMeasurementMatrixCBFunc(J_tIMAGE_INFO *pAqImageInfo);
	bool openFactoryAndCamera();
	void closeFactoryAndCamera();
};

}} //namespace brackets

#endif /*JAI_CAMERA_H*/