#include "cv_sfm.h"
#include <time.h>
#include <sys/stat.h>
#include <vector>
#include <inttypes.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include "frame.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include "sparse_match.h"
#define FEATURE_TYPE_ORB 1
#define FEATURE_TYPE_SURF 2
#define HAMMING_DISTANCE_THRESHOLD 40

using namespace std;
using namespace Eigen;

#define FEATURE_TYPE_ORB 1
#define FEATURE_TYPE_SURF 2
#define HAMMING_DISTANCE_THRESHOLD 40
string output_folder;

int LoadRGBImage(string imageFileName, Frame &t, int index)
{
	t.frame_index = index;
	t.keypoints.clear();
	t.descriptor.release();
	t.depth_value.clear();
	t.rgb.release();
	t.depth.release();

#if use_rectify
	cv::Mat image = cv::imread(fileName);
	CameraPara cali;
	float camera_matrix[9] = { 367.481519978327754, 0, 328.535778962615268, 0, 366.991059667167065, 233.779960757465176, 0, 0, 1 };
	float distort[5] = { -0.293510277812333,
		0.065334967950619,
		-0.000117308680498,
		0.000304779905426,
		0.000000000000000 };
	cv::Mat Camera_Matrix = cv::Mat(3, 3, CV_32F, camera_matrix);
	cv::Mat Camera_Distort = cv::Mat(1, 5, CV_32F, distort);
	undistort(image, t.rgb, Camera_Matrix, Camera_Distort);
#else

	string fileName = "/home/vision/Downloads/fountain_dense/urd/0000.png";
	t.rgb = imread(imageFileName,IMREAD_COLOR);
	cv::cvtColor(t.rgb, t.gray, CV_RGB2GRAY);
	if (t.rgb.data == NULL)
	{
		cout << "load image error ! " << imageFileName << endl;
		return 0;
	}
	int width = t.rgb.cols;
	int height = t.rgb.rows;
#endif
}

template <typename T>
string ToString(T val)
{
    stringstream stream;
    stream << val;
    return stream.str();
}

Eigen::Vector3d
triangulate_point(const Eigen::MatrixXd& proj_matrix1,
const Eigen::MatrixXd& proj_matrix2,
const Eigen::Vector2d& point1,
const Eigen::Vector2d& point2) {

	Eigen::Matrix<double, 6, 4> A;

	const double x1 = point1(0);
	const double y1 = point1(1);
	const double x2 = point2(0);
	const double y2 = point2(1);

	// Set up Jacobian as point2D x (P x point3D) = 0
	for (size_t k = 0; k<4; ++k) {
		// first set of points
		A(0, k) = x1 * proj_matrix1(2, k) - proj_matrix1(0, k);
		A(1, k) = y1 * proj_matrix1(2, k) - proj_matrix1(1, k);
		A(2, k) = x1 * proj_matrix1(1, k) - y1 * proj_matrix1(0, k);
		// second set of points
		A(3, k) = x2 * proj_matrix2(2, k) - proj_matrix2(0, k);
		A(4, k) = y2 * proj_matrix2(2, k) - proj_matrix2(1, k);
		A(5, k) = x2 * proj_matrix2(1, k) - y2 * proj_matrix2(0, k);
	}
	// Homogeneous 3D point is eigen vector corresponding to smallest singular
	// value. JacobiSVD is the most accurate method, though generally slow -
	// but fast for small matrices.
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
	Eigen::Matrix<double, 4, 4> V = svd.matrixV();
	// Normalize point
	V(0, 3) /= V(3, 3);
	V(1, 3) /= V(3, 3);
	V(2, 3) /= V(3, 3);

	return V.block<3, 1>(0, 3);

}

void savePLYFiles(string fileName, vector<Eigen::Vector3d> p, vector<Eigen::Vector3i>color)
{

	std::ofstream output_file(fileName.c_str());
	int pointNum = min(p.size(), color.size());
	output_file << "ply" << endl;
	output_file << "format ascii 1.0           { ascii/binary, format version number }" << endl;
	output_file << "comment made by Greg Turk  { comments keyword specified, like all lines }" << endl;
	output_file << "comment this file is a cube" << endl;
	output_file << "element vertex " << pointNum << "           { define \"vertex\" element, 8 of them in file }" << endl;
	output_file << "property float x" << endl;
	output_file << "property float y" << endl;
	output_file << "property float z" << endl;
	output_file << "property uchar red" << endl;
	output_file << "property uchar green" << endl;
	output_file << "property uchar blue" << endl;
	output_file << "end_header" << endl;
	for (int i = 0; i < pointNum; i++)
	{
		output_file << p[i](0) << " " << p[i](1) << " " << p[i](2) << " "
			<< color[i](2) << " " << color[i](1) << " " << color[i](0) << " " << endl;
	}
	output_file.close();

}



int detectAndExtractFeatures(Frame &t, int feature_type)
{
	assert(t.rgb.data != 0);

	clock_t start, end;
	start = clock();

	if (FEATURE_TYPE_ORB == feature_type)
	{
		cout << "using ORB descriptor!" << endl;
		cv::ORB orb(50000);
	    orb.detect(t.gray, t.keypoints);
	    orb.compute(t.gray,t.keypoints,t.descriptor);
	}
	else
	{
	}

	end = clock();
	double duration = (double)(end - start) / (double) CLOCKS_PER_SEC;
	printf("feature detection : %f s\n", duration); // 4.015

	cv::Mat img_keypoints;
	drawKeypoints(t.rgb, t.keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	char imageName[256];
	memset(imageName, '\0', 256);
	sprintf(imageName, "./image_%04d.jpg", t.frame_index);
	if(sc.save_image)
	{
		cv::imwrite(imageName, img_keypoints);
		cv::waitKey(1);
	}

	return 1;
}



template <typename T>
static float distancePointLine(const cv::Point_<T> point, const cv::Vec<T, 3>& line)
{
	//Line is given as a*x + b*y + c = 0
	return std::fabs(line(0)*point.x + line(1)*point.y + line(2))
		/ std::sqrt(line(0)*line(0) + line(1)*line(1));
}

template <typename T1, typename T2>
static cv::Mat drawEpipolarLines(const std::string& title, const cv::Matx<T1, 3, 3> F,
	const cv::Mat& img1, const cv::Mat& img2,
	const std::vector<cv::Point_<T2> > points1,
	const std::vector<cv::Point_<T2> > points2,
	const float inlierDistance = -1)
{
	CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
	cv::Mat outImg(img1.rows, img1.cols * 2, CV_8UC3);
	cv::Rect rect1(0, 0, img1.cols, img1.rows);
	cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
	/*
	* Allow color drawing
	*/
	if (img1.type() == CV_8U)
	{
		cv::cvtColor(img1, outImg(rect1), CV_GRAY2BGR);
		cv::cvtColor(img2, outImg(rect2), CV_GRAY2BGR);
	}
	else
	{
		img1.copyTo(outImg(rect1));
		img2.copyTo(outImg(rect2));
	}
	std::vector<cv::Vec<T2, 3> > epilines1, epilines2;
	cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
	cv::computeCorrespondEpilines(points2, 2, F, epilines2);

	CV_Assert(points1.size() == points2.size() &&
		points2.size() == epilines1.size() &&
		epilines1.size() == epilines2.size());

	cv::RNG rng(0);
	for (size_t i = 0; i<points1.size(); i++)
	{
		if (inlierDistance > 0)
		{
			if (distancePointLine(points1[i], epilines2[i]) > inlierDistance ||
				distancePointLine(points2[i], epilines1[i]) > inlierDistance)
			{
				//The point match is no inlier
				continue;
			}
		}
		/*
		* Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
		*/
		cv::Scalar color(rng(256), rng(256), rng(256));
		cv::line(outImg(rect2),
			cv::Point(0, -epilines1[i][2] / epilines1[i][1]),
			cv::Point(img1.cols, -(epilines1[i][2] + epilines1[i][0] * img1.cols) / epilines1[i][1]),
			color);
		cv::circle(outImg(rect1), points1[i], 3, color, -1, CV_AA);

		cv::line(outImg(rect1),
			cv::Point(0, -epilines2[i][2] / epilines2[i][1]),
			cv::Point(img2.cols, -(epilines2[i][2] + epilines2[i][0] * img2.cols) / epilines2[i][1]),
			color);
		cv::circle(outImg(rect2), points2[i], 3, color, -1, CV_AA);
	}
	return outImg;
}

int calculate_hamming_distance(uint64_t * f1, uint64_t * f2)
{
	int hamming_distance = (__builtin_popcountll(*(f1) ^ *(f2)) +
			__builtin_popcountll(*(f1 + 1) ^ *(f2 + 1)) +
			__builtin_popcountll(*(f1 + 2) ^ *(f2 + 2)) +
			__builtin_popcountll(*(f1 + 3) ^ *(f2 + 3)));
#if DEBUG_MODE_MILD
	statistics_num_distance_calculation++;
#endif
	return hamming_distance;
}

void sparseMatch_ORB(cv::Mat d_ref, cv::Mat d_new, std::vector<cv::DMatch> &matches)
{
	MILD::SparseMatcher sparseMatcher(FEATURE_TYPE_ORB, 24, 0, HAMMING_DISTANCE_THRESHOLD);
	sparseMatcher.train(d_new);
	sparseMatcher.search(d_ref, matches);
}
void BFMatch_ORB(cv::Mat d1, cv::Mat d2, std::vector<cv::DMatch> &matches)
{
	matches.clear();
	int feature_f1_num = d1.rows;
	int feature_f2_num = d2.rows;
	matches = std::vector<cv::DMatch>(feature_f1_num);
	int hamming_distance;
	for (int f1 = 0; f1 < feature_f1_num; f1++)
	{
		cv::Mat feature1_desc = d1.row(f1);
		int best_corr_fid = 0;
		int min_distance = INT_MAX;
		for (int f2 = 0; f2 < feature_f2_num; f2++)
		{
			hamming_distance = calculate_hamming_distance(feature1_desc.ptr<uint64_t>(), d2.row(f2).ptr<uint64_t>());
			if (hamming_distance < min_distance)
			{
				min_distance = hamming_distance;
				best_corr_fid = f2;
			}
		}

		DMatch m;
		m.queryIdx = f1;
		m.trainIdx = best_corr_fid;
		m.distance = min_distance;
		matches[f1] = m;
	}
}



// return reprojection error
float estimateFundamentalMatrix_7pt(Frame &frame_ref, Frame & frame_new,std::vector<cv::DMatch> &matches, Eigen::Matrix3d &F)
{

	assert(matches.size() == 7);
	Eigen::MatrixXd A(matches.size(), 9);

	float x1mean = 0, y1mean = 0, x2mean = 0, y2mean = 0;
	for (int i = 0; i < matches.size(); i++)
	{
		float x1 = frame_ref.keypoints[matches[i].queryIdx].pt.x;
		float y1 = frame_ref.keypoints[matches[i].queryIdx].pt.y;
		float x2 = frame_new.keypoints[matches[i].trainIdx].pt.x;
		float y2 = frame_new.keypoints[matches[i].trainIdx].pt.y;
		x1mean += x1;
		y1mean += y1;
		x2mean += x2;
		y2mean += y2;
	}
	x1mean /= matches.size();
	x2mean /= matches.size();
	y1mean /= matches.size();
	y2mean /= matches.size();
	float x1std = 0, y1std = 0, x2std = 0, y2std = 0;
	for (int i = 0; i < matches.size(); i++)
	{
		float x1 = frame_ref.keypoints[matches[i].queryIdx].pt.x;
		float y1 = frame_ref.keypoints[matches[i].queryIdx].pt.y;
		float x2 = frame_new.keypoints[matches[i].trainIdx].pt.x;
		float y2 = frame_new.keypoints[matches[i].trainIdx].pt.y;
		x1std += (x1 - x1mean) * (x1 - x1mean);
		y1std += (y1 - y1mean) * (y1 - y1mean);
		x2std += (x2 - x2mean) * (x2 - x2mean);
		y2std += (y2 - y2mean) * (y2 - y2mean);
	}
	x1std = sqrt(x1std / matches.size());
	x2std = sqrt(x2std / matches.size());
	y1std = sqrt(y1std / matches.size());
	y2std = sqrt(y2std / matches.size());
	Eigen::Matrix3d T1, T2;
	T1.setIdentity();
	T2.setIdentity();
	T1 << 1 / x1std, 0, -x1mean / x1std, 0, 1 / y1std, -y1mean / y1std, 0, 0, 1;
	T2 << 1 / x2std, 0, -x2mean / x2std, 0, 1 / y2std, -y2mean / y2std, 0, 0, 1;
	for (int i = 0; i < matches.size(); i++)
	{
		float x1 = frame_ref.keypoints[matches[i].queryIdx].pt.x / x1std - x1mean / x1std;
		float y1 = frame_ref.keypoints[matches[i].queryIdx].pt.y / y1std - y1mean / y1std;
		float x2 = frame_new.keypoints[matches[i].trainIdx].pt.x / x2std - x2mean / x2std;
		float y2 = frame_new.keypoints[matches[i].trainIdx].pt.y / y2std - y2mean / y2std;
		A.block<1, 9>(i, 0) << x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1;
	}
	Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(A);
	Eigen::MatrixXd f = lu_decomp.kernel();
	Eigen::Matrix3d f1, f2;
	f1 << f(0, 0), f(0, 1), f(0, 2), f(0, 3), f(0, 4), f(0, 5), f(0, 6), f(0, 7), f(0, 8);
	f2 << f(1, 0), f(1, 1), f(1, 2), f(1, 3), f(1, 4), f(1, 5), f(1, 6), f(1, 7), f(1, 8);
	double a = 0;
}

// x1'Fx2 = 0 x_new' F x_ref = 0
void estimateFundamentalMatrix_8pt(Frame &frame_ref, Frame & frame_new, std::vector<cv::DMatch> &matches, Eigen::Matrix3d &F)
{
	assert(matches.size() >= 8);
	Eigen::MatrixXd A(matches.size(), 9);

	float x1mean = 0, y1mean = 0, x2mean = 0, y2mean = 0;
	for (int i = 0; i < matches.size(); i++)
	{
		float x1 = frame_new.keypoints[matches[i].trainIdx].pt.x;
		float y1 = frame_new.keypoints[matches[i].trainIdx].pt.y;
		float x2 = frame_ref.keypoints[matches[i].queryIdx].pt.x;
		float y2 = frame_ref.keypoints[matches[i].queryIdx].pt.y;
		x1mean += x1;
		y1mean += y1;
		x2mean += x2;
		y2mean += y2;
	}
	x1mean /= matches.size();
	x2mean /= matches.size();
	y1mean /= matches.size();
	y2mean /= matches.size();
	float x1std = 0, y1std = 0, x2std = 0, y2std = 0;
	for (int i = 0; i < matches.size(); i++)
	{
		float x1 = frame_new.keypoints[matches[i].trainIdx].pt.x;
		float y1 = frame_new.keypoints[matches[i].trainIdx].pt.y;
		float x2 = frame_ref.keypoints[matches[i].queryIdx].pt.x;
		float y2 = frame_ref.keypoints[matches[i].queryIdx].pt.y;
		x1std += (x1 - x1mean) * (x1 - x1mean);
		y1std += (y1 - y1mean) * (y1 - y1mean);
		x2std += (x2 - x2mean) * (x2 - x2mean);
		y2std += (y2 - y2mean) * (y2 - y2mean);
	}
	x1std = sqrt(x1std / matches.size());
	x2std = sqrt(x2std / matches.size());
	y1std = sqrt(y1std / matches.size());
	y2std = sqrt(y2std / matches.size());
	Eigen::Matrix3d T1, T2;
	T1.setIdentity();
	T2.setIdentity();
	T1 << 1 / x1std, 0, -x1mean / x1std, 0, 1 / y1std, -y1mean / y1std, 0, 0, 1;
	T2 << 1 / x2std, 0, -x2mean / x2std, 0, 1 / y2std, -y2mean / y2std, 0, 0, 1;
	for (int i = 0; i < matches.size(); i++)
	{
		float x1 = frame_new.keypoints[matches[i].trainIdx].pt.x / x1std - x1mean / x1std;
		float y1 = frame_new.keypoints[matches[i].trainIdx].pt.y / y1std - y1mean / y1std;
		float x2 = frame_ref.keypoints[matches[i].queryIdx].pt.x / x2std - x2mean / x2std;
		float y2 = frame_ref.keypoints[matches[i].queryIdx].pt.y / y2std - y2mean / y2std;
		A.block<1, 9>(i, 0) << x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1;
	}

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3d f1, f2;
	Eigen::MatrixXd f = svd.matrixV().block<9, 1>(0, 8);
	f1 << f(0, 0), f(1, 0), f(2, 0), f(3, 0), f(4, 0), f(5, 0), f(6, 0), f(7, 0), f(8, 0);
	Eigen::JacobiSVD<Eigen::MatrixXd> svd_f1(f1, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXd sValues = svd_f1.singularValues();
	Eigen::Matrix3d trueSV;
	for(int m = 0; m < 3; m++)
	{
		for(int n =0;n<3;n++)
		{
			trueSV(m,n) = 0;
		}
	}
	trueSV(0, 0) = sValues(0);
	trueSV(1, 1) = sValues(1);
	f2 = svd_f1.matrixU() * trueSV * svd_f1.matrixV().transpose();
	F = f2;
	F = T1.transpose() * F * T2;
	F = F / F(2, 2);
}



// x_newTFx_ref = 0;
Mat ransac2D2D(Frame frame_ref, Frame frame_new, std::vector< DMatch > &init_matches,
	float reprojectionThreshold_pixel, int max_iter)
{
	std::vector<cv::Point2f> init_points_ref;
	std::vector<cv::Point2f> init_points_new;
	std::vector<unsigned char> inliersMask;
	cv::Mat H;
	if (init_matches.size() < 50)
	{
		init_matches.clear();
		return H;
	}

#if 0
	for (size_t i = 0; i < init_matches.size(); i++)
	{
		cv::Point2f ref_point(
			(frame_ref.keypoints[init_matches[i].queryIdx].pt.x),
			(frame_ref.keypoints[init_matches[i].queryIdx].pt.y));
		cv::Point2f new_point(
			(frame_new.keypoints[init_matches[i].trainIdx].pt.x),
			(frame_new.keypoints[init_matches[i].trainIdx].pt.y));
		//-- Get the keypoints from the good matches
		init_points_ref.push_back(ref_point);
		init_points_new.push_back(new_point);
	}
	float reprojectionThreshold = reprojectionThreshold_pixel;
	H = findFundamentalMat(init_points_ref, init_points_new, inliersMask, FM_LMEDS, reprojectionThreshold, 0.99);
	std::vector< DMatch > ransac_inlier_matches;
	for (size_t i = 0; i < inliersMask.size(); i++)
	{
		if (inliersMask[i])
		{
			ransac_inlier_matches.push_back(init_matches[i]);
		}
	}
//	init_matches = ransac_inlier_matches;

	init_points_ref.clear();
	init_points_new.clear();
#endif
	for (size_t i = 0; i < init_matches.size(); i+= 10)
	{
		cv::Point2f ref_point(
			(frame_ref.keypoints[init_matches[i].queryIdx].pt.x),
			(frame_ref.keypoints[init_matches[i].queryIdx].pt.y));
		cv::Point2f new_point(
			(frame_new.keypoints[init_matches[i].trainIdx].pt.x),
			(frame_new.keypoints[init_matches[i].trainIdx].pt.y));
		//-- Get the keypoints from the good matches
		init_points_ref.push_back(ref_point);
		init_points_new.push_back(new_point);
	}
	char frameMatchName[256];

	if (sc.save_image)
	{

		memset(frameMatchName, '\0', 256);
		sprintf(frameMatchName, "%s/opencv_compute_F_%04d_VS_%04d.jpg", output_folder.c_str(), frame_ref.frame_index, frame_new.frame_index);
		cv::Mat img_epi = drawEpipolarLines(string(frameMatchName), cv::Matx<float, 3, 3>(H), frame_ref.rgb, frame_new.rgb, init_points_ref, init_points_new);
		imwrite(frameMatchName, img_epi);

	}
	std::vector<cv::Point2f> ref_pts;
	std::vector<cv::Point2f> new_pts;
	for (size_t i = 0; i < init_matches.size(); i++)
	{
		cv::Point2f ref_point(
			(frame_ref.keypoints[init_matches[i].queryIdx].pt.x),
			(frame_ref.keypoints[init_matches[i].queryIdx].pt.y));
		cv::Point2f new_point(
			(frame_new.keypoints[init_matches[i].trainIdx].pt.x),
			(frame_new.keypoints[init_matches[i].trainIdx].pt.y));
		//-- Get the keypoints from the good matches
		ref_pts.push_back(ref_point);
		new_pts.push_back(new_point);
	}

	// ransac estimate of fundamental matrix
	int best_median_index = 100;
	int inlier_cnt = 0;
	int best_cnt = 0;
	Eigen::Matrix3d bestF;
	int hist_error[100];
	for (int k = 0; k < max_iter; k++)
	{
		Eigen::Matrix3d F;
		memset(hist_error, 0, 100 * sizeof(int));
		std::vector< DMatch > inlier_matches;
		for (size_t i = 0; i < 9; i++)
		{
			int seed = rand() % init_matches.size();
			inlier_matches.push_back(init_matches[seed]);
		}
		estimateFundamentalMatrix_8pt(frame_ref, frame_new, inlier_matches, F);
		cv::Mat F_mat;
		eigen2cv(F,F_mat);


		std::vector<cv::Vec<float, 3> > epilines1, epilines2, l1H,l2H;
		cv::computeCorrespondEpilines(ref_pts, 1, F_mat, epilines1); //Index starts with 1
		cv::computeCorrespondEpilines(new_pts, 2, F_mat, epilines2); //Index starts with 1
//		cv::computeCorrespondEpilines(ref_pts, 1, H, l1H); //Index starts with 1
//		cv::computeCorrespondEpilines(new_pts, 2, H, l2H); //Index starts with 1
		int cnt = 0;
		for (int i = 0; i < init_matches.size(); i++)
		{
			float x1 = frame_ref.keypoints[init_matches[i].queryIdx].pt.x;
			float y1 = frame_ref.keypoints[init_matches[i].queryIdx].pt.y;
			float x2 = frame_new.keypoints[init_matches[i].trainIdx].pt.x;
			float y2 = frame_new.keypoints[init_matches[i].trainIdx].pt.y;
			Eigen::Vector3d X1(x1, y1, 1);
			Eigen::Vector3d X2(x2, y2, 1);
			Eigen::Vector3d l1 = X2.transpose() * F;
			Eigen::Vector3d l2 = F * X1;

			float d1 = distancePointLine(cv::Point2f(x2, y2), epilines1[i]);
			float d2 = distancePointLine(cv::Point2f(x1, y1), epilines2[i]);
//			float d1H = distancePointLine(cv::Point2f(x2, y2), l1H[i]);
//			float d2H = distancePointLine(cv::Point2f(x1, y1), l2H[i]);

			float distance = fmax(d1, d2) * 100;
			if (distance < reprojectionThreshold_pixel)
			{
				cnt++;
			}
			int index = distance;
			index = max(0, index);
			index = min(index, 100 - 1);
			hist_error[index]++;
		}
		// find median value of hist_error
#if 1
		int i = 0;
		int median_cnt = 0;
		while (median_cnt < init_matches.size() / 2 && i < 100)
		{
			median_cnt += hist_error[i];
			i++;
		}
		if(best_median_index > i)
		{
			best_median_index = i;
			bestF = F;
		}
#else
		if (best_cnt < cnt)
		{
			best_cnt = cnt;
			bestF = F;
			inlier_cnt = cnt;
		}
#endif
	}
	float best_median_distance = best_median_index / 100.0;
	cv::Matx<double, 3, 3> F_mat;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			F_mat(i, j) = bestF(i, j);
		}
	}
	if (sc.save_image)
	{
		memset(frameMatchName, '\0', 256);
		sprintf(frameMatchName, "%s/my_compute_F_%04d_VS_%04d.jpg", output_folder.c_str(), frame_ref.frame_index, frame_new.frame_index);
		cv::Mat img_epi = drawEpipolarLines(string(frameMatchName), cv::Matx<double, 3, 3>(F_mat), frame_ref.rgb, frame_new.rgb, init_points_ref, init_points_new);
		imwrite(frameMatchName, img_epi);
	}

	std::vector< DMatch > least_median_filter_inliers;
	std::vector<cv::Vec<float, 3> > epilines1, epilines2;
	cv::computeCorrespondEpilines(ref_pts, 1, F_mat, epilines1); //Index starts with 1
	cv::computeCorrespondEpilines(new_pts, 2, F_mat, epilines2); //Index starts with 1
	for (int i = 0; i < init_matches.size(); i++)
	{
		float x1 = frame_ref.keypoints[init_matches[i].queryIdx].pt.x;
		float y1 = frame_ref.keypoints[init_matches[i].queryIdx].pt.y;
		float x2 = frame_new.keypoints[init_matches[i].trainIdx].pt.x;
		float y2 = frame_new.keypoints[init_matches[i].trainIdx].pt.y;
		Eigen::Vector3d X1(x1, y1, 1);
		Eigen::Vector3d X2(x2, y2, 1);
		float d1 = distancePointLine(cv::Point2f(x2, y2), epilines1[i]);
		float d2 = distancePointLine(cv::Point2f(x1, y1), epilines2[i]);

		float distance = fmax(d1, d2);
		if (distance < best_median_distance * 2)
		{
			least_median_filter_inliers.push_back(init_matches[i]);
		}
	}

	init_matches = least_median_filter_inliers;
	cv::eigen2cv(bestF, H);
	return H;
}

void two_view_triangulation(Frame &frame_ref, Frame &frame_new, std::vector<cv::DMatch> matches, cv::Mat F_mat, cv::Mat K_mat,
	std::vector<Eigen::Vector3d> &points_3d, Matx34d &P_new)
{
	points_3d.clear();
	Eigen::Matrix3d F, K;
	cv::cv2eigen(F_mat,F);
	cv::cv2eigen(K_mat, K);
	Eigen::Matrix3d E = K.transpose() * F * K;
	cv::Matx<double, 3, 3> E_mat;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			E_mat(i, j) = E(i, j);
		}
	}
	SVD svd(E_mat);
	Matx33d W(0, -1, 0, 1, 0, 0, 0, 0, 1);
	Matx33d WT;
	transpose(W, WT);
	Mat_<double> R[2];
	R[0] = svd.u * Mat(W) * svd.vt;
	R[1] = svd.u * Mat(WT) * svd.vt;
	Mat_<double> t[2];
	t[0] = svd.u.col(2);
	t[1] = -svd.u.col(2);
	if (determinant(R[0]) < 0)
	{
		R[0] = -R[0];
	}
	if (determinant(R[1]) < 0)
	{
		R[1] = -R[1];
	}

	Matx34d P2_mat[4];
	Matx34d P1_mat(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			P2_mat[i * 2 + j] =
				Matx34d(R[i](0, 0), R[i](0, 1), R[i](0, 2), t[j](0),
				R[i](1, 0), R[i](1, 1), R[i](1, 2), t[j](1),
				R[i](2, 0), R[i](2, 1), R[i](2, 2), t[j](2));
		}
	}
	int count_positive[4];
	for (int k = 0; k < 4; k++)
	{
		count_positive[k] = 0;
		Eigen::MatrixXd P2, P1;
		cv2eigen(P2_mat[k], P2);
		cv2eigen(P1_mat, P1);
		std::vector<Eigen::Vector3d> pts;
		std::vector<Eigen::Vector3i> colors;
		for (int i = 0; i < matches.size(); i++)
		{
			float x1 = frame_ref.keypoints[matches[i].queryIdx].pt.x / K(0, 0) - K(0, 2) / K(0, 0);
			float y1 = frame_ref.keypoints[matches[i].queryIdx].pt.y / K(1, 1) - K(1, 2) / K(1, 1);
			float x2 = frame_new.keypoints[matches[i].trainIdx].pt.x / K(0, 0) - K(0, 2) / K(0, 0);
			float y2 = frame_new.keypoints[matches[i].trainIdx].pt.y / K(1, 1) - K(1, 2) / K(1, 1);
			Eigen::Vector2d p1(x1, y1), p2(x2, y2);
			Eigen::Vector3d p3d = triangulate_point(P1, P2, p1, p2);
			if (p3d(2) > 0)
			{
				count_positive[k]++;
			}
		}
	}
	int max_count = -1;
	int bestK = 0;
	for (int k = 0; k < 4; k++)
	{
		if (max_count < count_positive[k])
		{
			max_count = count_positive[k];
			bestK = k;
		}
	}
	Eigen::MatrixXd P2, P1;
	cv2eigen(P2_mat[bestK], P2);
	cv2eigen(P1_mat, P1);
	std::vector<Eigen::Vector3d> pts;
	std::vector<Eigen::Vector3i> colors;
	std::vector<float> reprojection_errors;
	for (int i = 0; i < matches.size(); i++)
	{
		float x1 = frame_ref.keypoints[matches[i].queryIdx].pt.x / K(0, 0) - K(0, 2) / K(0, 0);
		float y1 = frame_ref.keypoints[matches[i].queryIdx].pt.y / K(1, 1) - K(1, 2) / K(1, 1);
		float x2 = frame_new.keypoints[matches[i].trainIdx].pt.x / K(0, 0) - K(0, 2) / K(0, 0);
		float y2 = frame_new.keypoints[matches[i].trainIdx].pt.y / K(1, 1) - K(1, 2) / K(1, 1);
		Eigen::Vector2d p1(x1, y1), p2(x2, y2);


		Eigen::Vector3d p1h(x1, y1, 1), p2h(x2, y2, 1);
		Eigen::Vector3d p3d = triangulate_point(P1, P2, p1, p2);
		Eigen::Vector3d l1, l2;
		l1 = E * p1h; l2 = p2h.transpose() * E;
		float reprojection_error = distancePointLine(cv::Point2f(x2, y2), cv::Vec<float, 3>(l1(0), l1(1), l1(2)));
		reprojection_errors.push_back(reprojection_error);

		cv::Vec3b color = frame_ref.rgb.at<cv::Vec3b>(frame_ref.keypoints[matches[i].queryIdx].pt.y, frame_ref.keypoints[matches[i].queryIdx].pt.x);
		if (reprojection_error < 5)
		{
			pts.push_back(p3d);
			Eigen::Vector4d p3d_homogenous;
			p3d_homogenous << p3d(0), p3d(1), p3d(2), 1;
			colors.push_back(Eigen::Vector3i(color[0], color[1], color[2]));
//			cout << p1h.transpose() << "		" << p2h.transpose() << "		" <<  p3d.transpose() << endl;
//			cout << (P1 * p3d_homogenous).transpose() / (P1 * p3d_homogenous)(2) << "		" << (P2 * p3d_homogenous).transpose() / ((P2 * p3d_homogenous)(2)) << endl;
		}
		else
		{
			p3d(2) = -1;
		}

		points_3d.push_back(p3d);

	}
	string save_data = "./point_cloud_";
	save_data += ToString(frame_ref.frame_index) + ".ply";
	P_new = P2_mat[bestK];

	char frameMatchName[2560];
	memset(frameMatchName, '\0', 2560);
	sprintf(frameMatchName, "./point_cloud_%04d_VS_%04d.ply", frame_ref.frame_index, frame_new.frame_index);

	savePLYFiles(frameMatchName, pts, colors);

}

class frame_pair
{
	int frame_ref_index;
	int frame_new_index;
	std::vector<cv::DMatch> matches;
	std::vector<Eigen::Vector3d> points_3d;
	cv::Mat F;
};


void frame_match(Frame &frame_ref, Frame &frame_new, const cv::Mat K, MonoFrameCorrespondence &fcorr)
{
	clock_t start, end;
	fcorr.frame_ref_id = frame_ref.frame_index;
	fcorr.frame_new_id = frame_new.frame_index;
	fcorr.matches.clear();
	fcorr.points_3d.clear();
	std::vector<cv::DMatch> &true_matches = fcorr.matches;
	std::vector<cv::DMatch> matches_ref_to_new, matches_new_to_ref;
	true_matches.reserve(frame_ref.descriptor.rows);

	cout << "input feature : " << frame_ref.descriptor.rows << " " << frame_new.descriptor.rows << endl;
	start = clock();

	// use brute force match
	if(	sc.use_sparse_match)
	{
		// use sparse match
		matches_ref_to_new.clear();
		matches_new_to_ref.clear();
		sparseMatch_ORB(frame_ref.descriptor, frame_new.descriptor, matches_ref_to_new );
		sparseMatch_ORB(frame_new.descriptor, frame_ref.descriptor, matches_new_to_ref);
	}
	else
	{
		matches_ref_to_new.clear();
		matches_new_to_ref.clear();

		BFMatch_ORB(frame_ref.descriptor, frame_new.descriptor, matches_ref_to_new);
		BFMatch_ORB(frame_new.descriptor, frame_ref.descriptor, matches_new_to_ref);
	}

	end = clock();
	for (int i = 0; i < matches_ref_to_new.size(); i++)
	{
		if (matches_ref_to_new[i].queryIdx == matches_new_to_ref[matches_ref_to_new[i].trainIdx].trainIdx)
		{
			if (matches_ref_to_new[i].distance < HAMMING_DISTANCE_THRESHOLD)
			{
				true_matches.push_back(matches_ref_to_new[i]);
			}
		}
	}
	cout << "match over " << true_matches.size()  << endl;

	float time_match = (end - start)/ (double) CLOCKS_PER_SEC;
	start = clock();
	cv::Mat img_matches;
	char frameMatchName[256];
	if (sc.save_image)
	{
		cv::drawMatches(frame_ref.rgb, frame_ref.keypoints, frame_new.rgb, frame_new.keypoints,
			true_matches, img_matches);
		memset(frameMatchName, '\0', 256);
		sprintf(frameMatchName, "%s/match_%04d_VS_%04d.jpg", output_folder.c_str(), frame_ref.frame_index, frame_new.frame_index);
		cv::imwrite(frameMatchName, img_matches);

	}
	fcorr.F = ransac2D2D(frame_ref, frame_new, true_matches, 3, 10000);

	cout << "ransac finished!" << endl;
	if (sc.save_image)
	{
		cv::drawMatches(frame_ref.rgb, frame_ref.keypoints, frame_new.rgb, frame_new.keypoints,
			true_matches, img_matches);
		memset(frameMatchName, '\0', 256);
		sprintf(frameMatchName, "%s/match_inlier_%04d_VS_%04d.jpg", output_folder.c_str(), frame_ref.frame_index, frame_new.frame_index);
		cv::imwrite(frameMatchName, img_matches);
	}

	end = clock();
	float time_ransac2D2D = (end - start)/ (double) CLOCKS_PER_SEC;
	start = clock();
	Matx34d P_new;
	two_view_triangulation(frame_ref, frame_new, true_matches, fcorr.F, K, fcorr.points_3d, P_new);

	fcorr.P = P_new;
	end = clock();
	float time_triangulation = (end - start)/ (double) CLOCKS_PER_SEC;

	cout << "time match/ransac/triangulation: " << time_match << " " << time_ransac2D2D << " " << time_triangulation << endl;
}
system_constants sc;

float calculate_scale(MonoFrameCorrespondence &fC0, MonoFrameCorrespondence &fC1)
{
	std::vector<double> points_prev_depth, points_cur_depth;
	float scale_ATb = 0, scale_ATA = 0;
	int shared_features = 0;
	for (int i = 0; i < fC0.matches.size(); i++)
	{
		for (int j = 0; j < fC1.matches.size(); j++)
		{
			if (fC0.matches[i].trainIdx == fC1.matches[j].queryIdx)
			{
				if (fC0.points_3d[i](2) > 0 && fC1.points_3d[j](2) > 0)
				{
					shared_features++;
					Eigen::Vector4d homo_point;
					homo_point << fC0.points_3d[i](0), fC0.points_3d[i](1), fC0.points_3d[i](2), 1;
					Eigen::MatrixXd P;
					cv::cv2eigen(fC0.P, P);

					Eigen::Vector3d local_point_prev = P * homo_point;
					points_prev_depth.push_back( double(local_point_prev(2)) );
					points_cur_depth.push_back( double(fC1.points_3d[j](2)) );

					scale_ATb += local_point_prev(2) * fC1.points_3d[j](2);
					scale_ATA += local_point_prev(2) * local_point_prev(2);
				}
			}
		}
	}
	float scale = -1;
	if (shared_features > 20)
	{
		scale = scale_ATb / scale_ATA;
	}
	return scale;

}
void hw_1(int argc, char **argv)
{
	output_folder = "./";

	cv::FileStorage fSettings;
	fSettings = cv::FileStorage(argv[2], cv::FileStorage::READ);


	float fx, fy, cx, cy;
	fx = fSettings["fx"];
	fy = fSettings["fy"];
	cx = fSettings["cx"];
	cy = fSettings["cy"];
	sc.save_image = fSettings["save_image"];
	sc.use_sparse_match = fSettings["use_sparse_match"];
	clock_t start, end;
	cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
	K.at<float>(0, 0) = fx;
	K.at<float>(1, 1) = fy;
	K.at<float>(0, 2) = cx;
	K.at<float>(1, 2) = cy;
	K.at<float>(2, 2) = 1;
	const int dir_err = mkdir(output_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

	std::vector<std::string> fileName;
	std::vector<Frame> frame_list;
	frame_list.reserve(100);
	FILE * read_file = fopen(argv[1],"r");
	char line[256];
	while(!feof(read_file))
	{
		string input;
		memset(line,'\0',sizeof(line));
		fscanf(read_file,"%s\n",line);
		input = line;
		cout <<input.size() << endl;
		fileName.push_back(input);
	}

	for (int i = 0; i < fileName.size(); i++)
	{
		Frame curFrame;
		cout << "load image : " << fileName[i] << endl;
		LoadRGBImage(fileName[i], curFrame, i);
		frame_list.push_back(curFrame);
		detectAndExtractFeatures(frame_list[i], FEATURE_TYPE_ORB);
	}

	// feature match with left/right check
	cv::Mat F;
	std::vector<cv::DMatch> true_matches;
	std::vector<Eigen::Vector3d> points_3d;
	std::vector<MonoFrameCorrespondence> fClist;
	for (int i = 0; i < frame_list.size() - 1; i++)
	{

		int j = (i + 1)% frame_list.size() ;
			Frame &frame_ref = frame_list[i];
			Frame &frame_new = frame_list[j];
			clock_t start, end;
			start = clock();
			MonoFrameCorrespondence fcorr(frame_ref.frame_index, frame_new.frame_index);
			frame_match(frame_ref, frame_new, K, fcorr);
			fClist.push_back(fcorr);
			end = clock();


			//3d to 2d projection


			cv::Mat rvec, tvec, mask;
//			cv::solvePnPRansac(fcorr.points_3d, init_points_new_image,
//				K, cv::Mat(), rvec, tvec, false,
//				max_iter_num, reprojectionThreshold, 0.85);
			cout << "frame pair" << i << "  " << j;
			cout << "time cost : " << (end - start)/ (double) CLOCKS_PER_SEC << endl;
	}

	// accmulate shared points between fClist[0] and fCList[1]

	std::vector<Eigen::Matrix4d> Trajecotry;
	float scale = 1;
	for (int i = 0; i < fClist.size(); i++)
	{
		Eigen::Matrix4d T_camera_to_world;
		Eigen::Matrix4d T_cur_to_prev;
		if (i == 0)
		{
			scale = 1;
			T_camera_to_world.setIdentity();
			Trajecotry.push_back(T_camera_to_world);
		}
		Eigen::MatrixXd P;
		cv::cv2eigen(fClist[i].P, P);
		Eigen::Matrix3d R = P.block<3, 3>(0, 0);
		Eigen::Vector3d t = P.block<3, 1>(0,3);
		Eigen::Matrix3d R_cur_to_prev = R.inverse();
		Eigen::Vector3d t_cur_to_prev = -R.inverse() * t;

		float scale_prev_to_cur = 1;
		if (i > 0)
		{
			scale_prev_to_cur = calculate_scale(fClist[i - 1], fClist[i]);
		}
		scale = scale_prev_to_cur;

		cout << "prev scale : " << scale_prev_to_cur << "	cur scale: " << scale << endl;
		for (int j = 0; j < fClist[i].points_3d.size(); j++)
		{
			if (fClist[i].points_3d[j](2) > 0)
			{
				fClist[i].points_3d[j] = fClist[i].points_3d[j] / scale;
			}
		}
		if (scale > 0)
		{
			t_cur_to_prev /= scale;
		}
		else
		{
			cout << "warning ! match failed!" << endl;
		}
		T_cur_to_prev.setIdentity();
		T_cur_to_prev.block<3, 3>(0, 0) = R_cur_to_prev;
		T_cur_to_prev.block<3, 1>(0, 3) = t_cur_to_prev;
		T_camera_to_world = Trajecotry[i] * T_cur_to_prev;
		cout << Trajecotry[i] << endl;
		Trajecotry.push_back(T_camera_to_world);
	}

	// save all the points into ply file

	std::vector<Eigen::Vector3d> pts;
	std::vector<Eigen::Vector3i> colors;

	for (int i = 0; i < fClist.size(); i++)
	{

		Frame &frame_ref = frame_list[fClist[i].frame_ref_id];
		std::vector<cv::DMatch> &matches = fClist[i].matches;

		for (int j = 0; j < fClist[i].points_3d.size(); j++)
		{
			cv::Vec3b color = frame_ref.rgb.at<cv::Vec3b>(
				frame_ref.keypoints[matches[j].queryIdx].pt.y, frame_ref.keypoints[matches[j].queryIdx].pt.x);
			if (fClist[i].points_3d[j](2) > 0)
			{
				Eigen::Vector3d point_local = fClist[i].points_3d[j];
				Eigen::Vector4d homo_point_local;
				Eigen::Vector4d homo_point_global;
				homo_point_local << point_local[0], point_local[1], point_local[2], 1;
				homo_point_global = Trajecotry[i] * homo_point_local;

				Eigen::Vector3d global_pt;
				global_pt << homo_point_global(0),homo_point_global(1),homo_point_global(2);
				pts.push_back(global_pt);
				colors.push_back(Eigen::Vector3i(color[0], color[1], color[2]));
			}
		}
	}
	savePLYFiles("./global_points.ply", pts, colors);
}
