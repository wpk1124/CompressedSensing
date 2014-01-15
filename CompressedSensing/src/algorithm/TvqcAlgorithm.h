#pragma once
#include "src/algorithm/ICSAlgorithm.h"

namespace CS {
namespace algorithm {

class TvqcAlgorithm : public ICSAlgorithm {
public:
	TvqcAlgorithm();
	~TvqcAlgorithm();
	cv::Mat recoverImage(const cv::Mat& measurementMatrix, const cv::Mat& observations, const cv::Mat& startingSolution);
private:
	cv::Mat Tvqc_Newton(int &niter, cv::Mat& x0, cv::Mat& t0,const cv::Mat A,const cv::Mat b, const cv::Mat Dv, const cv::Mat Dh, double epsilon, double tau, double newtontol, int newtonmaxiter, double cgtol, int cgmaxiter);
};


}}//namespace end