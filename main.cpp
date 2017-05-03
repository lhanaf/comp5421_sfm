#include <iostream>
#include <opencv/cv.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <time.h>
#include <list>
#include <omp.h>
#include <stdio.h>
#include "frame.h"
#include "cv_sfm.h"
using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	string tumFolder = "D:\\HKUST\\data\\rgbd_dataset_freiburg1_xyz\\rgbd_dataset_freiburg1_xyz";
	string fileName2 = "D:\\HKUST\\course\\ELEC5640\\project\\image_list.txt";
	string folder = "D:\\HKUST\\data\\loop_closure_detection_datasets\\imageList_CityCentre.txt";
	string tag = folder.substr(folder.find_last_of("\\") + 1, folder.find_last_of(".") - folder.find_last_of("\\") - 1);
	if (argc > 1)
	{
		folder = argv[1];
	}

	// init logging environment 
	// Initialize Google's logging library.   
//	google::InitGoogleLogging(argv[0]);
//	google::SetLogDestination(google::GLOG_INFO, "output\\testRGBDSLAM\\logs\\");
#if 1
	hw_1(argc,argv);
//	test_sparse_hash_match(folder);
//	test_LCD(folder);
//	test_mild(folder);
//	test_rgbd_slam(argc, argv);
	return 0;
#endif
}
