#ifndef CV_SFM_H
#define CV_SFM_H

#include <iostream>
#include <fstream>
#include <string>

#include "frame.h"


using namespace std;
void hw_1(int argc, char ** argv);
void hw_2();



class system_constants
{
public:
	system_constants()
	{

	}
	int save_image;
	int use_sparse_match;
};
extern system_constants sc;

class MonoFrameCorrespondence
{
public:

	MonoFrameCorrespondence()
	{
		frame_ref_id = 0;
		frame_new_id = 0;
		F = cv::Mat();
		matches.clear();
		points_3d .clear();
	}
	MonoFrameCorrespondence(int ref_id, int new_id,
		cv::Mat input_F = cv::Mat(),
		std::vector<cv::DMatch> input_true_matches = std::vector<cv::DMatch>(),
		std::vector<Eigen::Vector3d> input_points_3d = std::vector<Eigen::Vector3d>())
	{
		frame_ref_id = ref_id;
		frame_new_id = new_id;
		F = input_F.clone();
		matches = input_true_matches;
		points_3d = input_points_3d;
	}

	int frame_ref_id;
	int frame_new_id;
	cv::Mat F;
	cv::Matx34d P;
	std::vector<cv::DMatch> matches;			// query_index for ref
	std::vector<Eigen::Vector3d> points_3d;		// based on the matches

};
#endif
