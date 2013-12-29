#ifndef GPU_MANAGER_H
#define GPU_MANAGER_H
#define VIENNACL_WITH_OPENCL
//#define VIENNACL_DEBUG_ALL

#include <memory>
#include <tuple>
#include <vector>
#include <opencv2/core/core.hpp>

#include "viennacl/ocl/backend.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/linalg/prod.hpp"       //generic matrix-vector product
#include "viennacl/linalg/norm_2.hpp"     //generic l2-norm for vectors
#include "viennacl/linalg/lu.hpp"

typedef std::vector<std::vector<float>> StdMatrix;

namespace CS {
namespace gpu {

typedef std::tuple<int, int, int> GenerationParameters;

class GPUSolver {
public:
	void createBinaryMeasurementMatrix(int rows, int cols, cv::Mat*);
	cv::Mat product(const cv::Mat& A, const cv::Mat& y);
	cv::Mat transProduct(const cv::Mat& A, const cv::Mat& y);
	cv::Mat linsolve(const cv::Mat& A,const cv::Mat& y);
private:
	//methods
	std::string getRandKernelSource();
	GenerationParameters getGenerationParameters(int, int);
	void performMatrixGeneration(GenerationParameters&, cv::Mat* output);
	void initializeSeeds(GenerationParameters&, viennacl::vector<int>* seedVector);
	void runRandomGeneratorKernel(GenerationParameters&, viennacl::vector<int>& seeds, viennacl::vector<float>* randomMatrix);

	//data
};


}} // namespace end

#endif /*GPU_MANAGER_H*/