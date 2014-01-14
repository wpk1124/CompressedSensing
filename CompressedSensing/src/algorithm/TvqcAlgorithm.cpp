#include "src/algorithm/TvqcAlgorithm.h"

using namespace CS::algorithm;

TvqcAlgorithm::TvqcAlgorithm() {}
TvqcAlgorithm::~TvqcAlgorithm() {}

cv::Mat TvqcAlgorithm::Tvqc_Newton(int &niter, cv::Mat& x0, cv::Mat& t0, const cv::Mat A, const cv::Mat b, const cv::Mat Dv, const cv::Mat Dh, double epsilon, double tau, double newtontol, int newtonmaxiter, double cgtol, int cgmaxiter) {
	double alpha = 0.01, beta = 0.5;
	int N = x0.rows, nn = (int) (std::sqrt(N)), K = A.rows;
	//initial point
	cv::Mat AtA = A.t() * A;
	cv::Mat x = x0.clone();
	cv::Mat t = t0.clone();
	cv::Mat r = A*x - b;
	cv::Mat Dhx = Dh * x;
	cv::Mat Dvx = Dv * x;

	cv::Mat ft = 0.5 * (Dhx.mul(Dhx) + Dvx.mul(Dvx) - t.mul(t));
	cv::Mat logft;
	cv::log(ft, logft);
	cv::Mat rTr = r.t() * r;
	std::cout << "\nTest---pierwszy---------\n\n"; 
	std::cout << "\nTest------------\n\n" << rTr.at<double>(0,0);
	double fe = 0.5 * (rTr.at<double>(0) - std::pow(epsilon,2));
	std::cout << "\nWyzej sie wysypuje\n\n";
	double f = cv::sum(t)[0] - (1/tau) * (cv::sum(logft)[0] + log(-fe));
	
	niter = 0;
	bool done = false;
	while(!done) {
		cv::Mat Atr = A.t() * r;
		cv::Mat ntgx = Dh.t()*((1/ft).mul(Dhx)) + Dv.t()*((1/ft).mul(Dvx)) + (1/fe)*Atr;
		cv::Mat ntgt = -tau * cv::Mat::ones(N,1,CV_32FC1) - t/ft;
		cv::Mat gradf = cv::Mat(2*N, 1, CV_32FC1);
		cv::vconcat(ntgx, ntgt, gradf);
		gradf = -(1/tau) * gradf;

		cv::Mat sig22 = 1/ft + t.mul(t)/ft.mul(ft);
		cv::Mat sig12 = -t/(ft.mul(ft));
		cv::Mat sigb = 1/(ft.mul(ft)) - sig12.mul(sig12)/sig22;

		cv::Mat w1p = ntgx = Dh.t() * ((Dhx.mul(sig12/sig22)).mul(ntgt)) - Dv.t()*((Dvx.mul(sig12/sig22)).mul(ntgt));

		cv::Mat H11p = Dh.t() * cv::Mat::diag(-1/ft + sigb.mul(Dhx.mul(Dhx))) * Dh
			+ Dv.t() * cv::Mat::diag(-1/ft + sigb.mul(Dvx.mul(Dvx))) * Dv
			+ Dh.t() * cv::Mat::diag(sigb.mul(Dhx.mul(Dvx))) * Dv
			+ Dv.t() * cv::Mat::diag(sigb.mul(Dhx.mul(Dvx))) * Dh
			- (1/fe) * AtA + (1/pow(fe,2))*Atr*Atr.t();

		cv::Mat dx = cv::Mat(N, 1, CV_32FC1);
		cv::solve(H11p, w1p, dx);
		cv::Mat Adx = A * dx;

		cv::Mat Dhdx = Dh * dx, Dvdx = Dv * dx;
		cv::Mat dt = (1/sig22).mul(ntgt - sig12.mul(Dhx.mul(Dhdx) + Dvx.mul(Dvdx)));

		//minimum step size that stays in the interior
		cv::Mat aqt = Dhdx.mul(Dhdx) + Dvdx.mul(Dvdx) - dt.mul(dt);
		cv::Mat bqt = 2*(Dhdx.mul(Dhx) + Dvdx.mul(Dvx) - t.mul(dt));
		cv::Mat cqt = Dhx.mul(Dhx) + Dvx.mul(Dvx) - t.mul(t);
		cv::Mat tsols = cv::Mat(2*N, 1, CV_32FC1);
		cv::Mat sqrt1 = cv::Mat(N, 1, CV_32FC1);
		cv::sqrt(bqt.mul(bqt)-4*aqt.mul(cqt), sqrt1);
		hconcat((-bqt+sqrt1)/(2*aqt),(-bqt-sqrt1)/(2*aqt),tsols);
		cv::Mat dqt = cv::Mat(2*N, 1, CV_32FC1);
		hconcat(bqt.mul(bqt) > 4*aqt.mul(cqt),bqt.mul(bqt) > 4*aqt.mul(cqt), dqt);
		cv::Mat indt = cv::Mat(2*N, 1, CV_32FC1);
		cv::findNonZero(dqt & (tsols > 0), indt);
		cv::Mat Maqe = Adx.t()*Adx;
		cv::Mat Mbqe = 2*r.t()*Adx;
		double aqe = Maqe.at<double>(0);
		double bqe = Mbqe.at<double>(0);
		double cqe = rTr.at<double>(0) - std::pow(epsilon,2);
		cv::Mat tsols1 = tsols.rowRange(indt.at<int>(0), indt.at<int>(indt.rows));
		double tsols1_min;
		cv::minMaxLoc(tsols1, &tsols1_min, NULL, NULL, NULL);
		double smax = std::min(1.0,std::min(tsols1_min,(-bqe+std::sqrt(pow(bqe,2)-4*aqe*cqe))/(2*aqe)));
		double s = 0.99 * smax;

		//backtracking line search
		bool suffdec;
		int backiter;
		cv::Mat xp = cv::Mat(N,1,CV_32FC1);
		cv::Mat tp = cv::Mat(N,1,CV_32FC1);
		cv::Mat rp = cv::Mat(N,1,CV_32FC1);
		cv::Mat Dhxp = cv::Mat(N,1,CV_32FC1);
		cv::Mat Dvxp = cv::Mat(N,1,CV_32FC1);
		cv::Mat ftp = cv::Mat(N,1,CV_32FC1);
		cv::Mat rpTrp = cv::Mat(1,1,CV_32FC1);
		double fep, fp, flin;
		cv::Mat logftp = cv::Mat(N,1,CV_32FC1);
		cv::Mat dxdt = cv::Mat(2*N,1,CV_32FC1);
		cv::vconcat(dx,dt,dxdt);
		cv::Mat gradfx = gradf.t() * dxdt;
		flin = f + alpha*s*gradfx.at<double>(0);

		while(!suffdec) {
			xp = x + s*dx;
			tp = t + s*dt;
			rp = r + s*Adx;
			Dhxp = Dhx + s*Dhdx;
			Dvxp = Dvx + s*Dvdx;
			ftp = 0.5 * (Dhxp.mul(Dhxp) + Dvxp.mul(Dvxp) - tp.mul(tp));
			rpTrp = rp.t()*rp;
			fep = 0.5 * (rpTrp.at<double>(0) - pow(epsilon,2));
			cv::log(ftp, logftp);
			fp = cv::sum(tp)[0] - (1/tau) * (cv::sum(logftp)[0] + log(-fep));
			suffdec = (fp <= flin);
			s = beta * s;
			backiter++;
			if (backiter > 32) { //stuck on backtracking line search, returning previous iterate
				return x;
			}
		}

		//set up for next iterations
		x = xp;
		t = tp;
		r = rp;
		Dvx = Dvxp;
		Dhx = Dhxp;
		ft = ftp;
		fe = fep;
		f = fp;
		double lambda2 = -gradfx.at<double>(0);
		double stepsize = s * cv::norm(dxdt);
		niter++;
		done = (lambda2/2 < newtontol) || (niter <= newtonmaxiter);
	}
	return x;
}

cv::Mat TvqcAlgorithm::recoverImage(const cv::Mat& measurementMatrix, const cv::Mat& observations, const cv::Mat& startingSolution) {
	double epsilon = 0.005, lbtol = 0.001, mu = 10, cgtol = 0.00000001, newtontol = lbtol;
	int cgmaxiter = 200, newtonmaxiter = 50, N = startingSolution.rows, nn = (int) (std::sqrt(N));

	//Create sparse (differencing) matrices for TV
	cv::Mat A = cv::Mat(nn,nn,CV_32FC1);
	cv::Mat A1 = cv::Mat::ones(nn-1, nn, CV_32FC1);
	cv::vconcat(A1*(-1),cv::Mat::zeros(1, nn, CV_32FC1),A);
	A = A.t();
	A = A.reshape(0,N);
	cv::Mat B = cv::Mat(nn,nn,CV_32FC1);
	cv::Mat B1 = cv::Mat::zeros(1, nn, CV_32FC1);
	cv::vconcat(B1,cv::Mat::ones(nn-1, nn, CV_32FC1),B);
	B = B.t();
	B = B.reshape(0,N);
	cv::Mat Dv = cv::Mat::diag(A);
	for(int i = 1;i < N; i++)
		Dv.at<int>(i-1,i) = B.at<int>(i);

	cv::Mat C = cv::Mat(nn,nn,CV_32FC1);
	cv::Mat C1 = cv::Mat::ones(nn, nn-1, CV_32FC1);
	cv::hconcat(C1*(-1),cv::Mat::zeros(nn, 1, CV_32FC1),C);
	C = C.t();
	C = C.reshape(0,N);
	cv::Mat D = cv::Mat(nn,nn,CV_32FC1);
	cv::Mat D1 = cv::Mat::zeros(nn, 1, CV_32FC1);
	cv::hconcat(D1,cv::Mat::ones(nn, nn-1, CV_32FC1),D);
	D = D.t();
	D = D.reshape(0,N);
	cv::Mat Dh = cv::Mat::diag(C);
	for(int i = 3;i < N; i++)
		Dh.at<int>(i-3,i) = D.at<int>(i);

	//starting point
	cv::Mat x = startingSolution.clone();

	cv::Mat Dhx = Dh * x;
	cv::Mat Dvx = Dv * x;
	cv::Mat Norm;
	sqrt(Dhx.mul(Dhx) + Dvx.mul(Dvx), Norm);
	double NormMaxElement;
	cv::minMaxLoc(Norm, NULL, &NormMaxElement, NULL, NULL);
	cv::Mat t = 0.95 * Norm + 0.1 * cv::Mat::ones(N, 1, CV_32FC1) * NormMaxElement;

	//chooose initial value of tau so that the duality gap after the first step will be about the original TV
	double tau = (N+1) / cv::sum(Norm)[0];

	int lbiter = std::ceil((log(N+1)-log(lbtol)-log(tau))/log(mu));
	std::cout << "Number of log barrier iterations: " << lbiter << std::endl;
	int niter = 0, totaliter = 0;
	double tvxp;
	cv::Mat xp = cv::Mat(N, 1, CV_32FC1);
	for(int i = 0; i < lbiter; i++) {
		xp = Tvqc_Newton(niter, x, t, measurementMatrix, observations, Dv, Dh, epsilon, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter);
		totaliter = totaliter + niter;
		cv::sqrt((Dh*xp).mul(Dh*xp) + (Dv*xp).mul(Dv*xp), Norm);
		tvxp = cv::sum(Norm)[0];
		x = xp;
		tau = mu * tau;
	}
	cv::Mat outputImage = x.clone();
	
	return outputImage;
}