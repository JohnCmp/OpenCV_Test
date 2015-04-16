#include "stdafx.h"
#include <opencv/highgui.h>
#include <opencv/cv.h>

using namespace std;
using namespace cv;

///	Rotated rectangle detect
//int _tmain()
//{
//    cv::Mat input = cv::imread("../inputData/RotatedRect.png");
//
//    // convert to grayscale (you could load as grayscale instead)
//    cv::Mat gray;
//    cv::cvtColor(input,gray, CV_BGR2GRAY);
//
//    // compute mask (you could use a simple threshold if the image is always as good as the one you provided)
//    cv::Mat mask;
//    cv::threshold(gray, mask, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
//
//    // find contours (if always so easy to segment as your image, you could just add the black/rect pixels to a vector)
//    std::vector<std::vector<cv::Point>> contours;
//    std::vector<cv::Vec4i> hierarchy;
//    cv::findContours(mask,contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
//
//    // Draw contours and find biggest contour (if there are other contours in the image, we assume the biggest one is the desired rect)
//    // drawing here is only for demonstration!
//    int biggestContourIdx = -1;
//    float biggestContourArea = 0;
//    cv::Mat drawing = cv::Mat::zeros( mask.size(), CV_8UC3 );
//    for( int i = 0; i< contours.size(); i++ )
//    {
//        cv::Scalar color = cv::Scalar(0, 100, 0);
//        drawContours( drawing, contours, i, color, 1, 8, hierarchy, 0, cv::Point() );
//
//        float ctArea= cv::contourArea(contours[i]);
//        if(ctArea > biggestContourArea)
//        {
//            biggestContourArea = ctArea;
//            biggestContourIdx = i;
//        }
//    }
//
//    // if no contour found
//    if(biggestContourIdx < 0)
//    {
//        std::cout << "no contour found" << std::endl;
//        return 1;
//    }
//
//    // compute the rotated bounding rect of the biggest contour! (this is the part that does what you want/need)
//    cv::RotatedRect boundingBox = cv::minAreaRect(contours[biggestContourIdx]);
//    // one thing to remark: this will compute the OUTER boundary box, so maybe you have to erode/dilate if you want something between the ragged lines
//
//
//
//    // draw the rotated rect
//    cv::Point2f corners[4];
//    boundingBox.points(corners);
//    cv::line(drawing, corners[0], corners[1], cv::Scalar(255,255,255));
//    cv::line(drawing, corners[1], corners[2], cv::Scalar(255,255,255));
//    cv::line(drawing, corners[2], corners[3], cv::Scalar(255,255,255));
//    cv::line(drawing, corners[3], corners[0], cv::Scalar(255,255,255));
//
//    // display
//    cv::imshow("input", input);
//    cv::imshow("drawing", drawing);
//    cv::waitKey(0);
//
//    cv::imwrite("rotatedRect.png",drawing);
//
//    return 0;
//}
///--------------------------------------------------------------------------------------------

void CannyThreshold( int, void*);
void Threshold(int, void*);
void Dilate(int, void*);
void Erose(int, void*);

Mat globalMat, dst, detected_edges;
int edgeThresh = 1;
int lowThreshold = 50;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Dilation result";

int dilateSize = 6;
int erodeSize = 7;

int _tmain(int argc, _TCHAR* argv[])			//pattern match
{
	cv::Mat sourceImg = cvLoadImage("D:\\example2.png");	// for global usage
	/// Load an image
	if( !sourceImg.data )
	{ return -1; }
	cv::Mat tempImg = cvLoadImage("D:\\template2.png");
	/*cv::Mat sourceImg = cvLoadImage("D:\\lena.bmp");
	cv::Mat tempImg = cvLoadImage("D:\\template.bmp");*/
	cv::imshow("template", tempImg);
	//cout << tempImg << endl;
	cv::Mat resultImg;
	cv::matchTemplate(sourceImg, tempImg, resultImg,  CV_TM_SQDIFF);
	double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;  cv::Point matchLoc;
	minMaxLoc( resultImg, &minVal, &maxVal, &minLoc, &maxLoc);
	Mat mattest = sourceImg;

	matchLoc = minLoc;

	Rect ICROI(matchLoc.x, matchLoc.y, tempImg.cols, tempImg.rows);			// get IC ROI

	globalMat = sourceImg(ICROI);
	Mat matchImg = sourceImg.clone();
	cv::rectangle(matchImg, matchLoc, cv::Point( matchLoc.x + tempImg.cols , matchLoc.y + tempImg.rows ),cvScalar(0,0,255));		// draw rectangle
	cv::resize(matchImg, matchImg, cv::Size(800,1000));
	//printf("%d,%d\r\n", maxLoc.x + tempImg.cols,  maxLoc.y + tempImg.rows);
	cv::imshow("match image", matchImg);
	//cv::waitKey(0);
	






	/// Create a window
	namedWindow( window_name, CV_WINDOW_AUTOSIZE );

	Mat ICImg = globalMat.clone();
	//threshold(ICImg, dst, 60, 255, CV_THRESH_BINARY);
	//imshow("Threshold 60", dst);
	dst = globalMat;
	//createTrackbar( "Dilation size", window_name, &dilateSize, 20, Dilate);
	Dilate(0, 0);
	Erose(0, 0);
	waitKey(0);

	/// Create a Trackbar for user to enter threshold
	//createTrackbar( "Threshold:", window_name, &lowThreshold, max_lowThreshold, Threshold );	
	/// Show the image
	//CannyThreshold(0, 0);

	//find contour
	//vector< vector<Point> >	contours;
	//Canny( sourceImg, dst, 80, 255, 3 );
	//threshold(sourceImg, dst, 100, 255, CV_THRESH_BINARY);
	
	//findContours(dst, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	//imshow("test", dst);
		//imshow("contour image", contourImg);
	cv::Mat input = dst;

    // convert to grayscale (you could load as grayscale instead)
    cv::Mat gray;
    cv::cvtColor(input,gray, CV_BGR2GRAY);

    // compute mask (you could use a simple threshold if the image is always as good as the one you provided)
    cv::Mat mask;
    cv::threshold(gray, mask, 50, 255, CV_THRESH_BINARY); //CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
	imshow("mask image", mask);
    // find contours (if always so easy to segment as your image, you could just add the black/rect pixels to a vector)
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask ,contours, CV_RETR_LIST , CV_CHAIN_APPROX_SIMPLE);

    // Draw contours and find biggest contour (if there are other contours in the image, we assume the biggest one is the desired rect)
    // drawing here is only for demonstration!
    int matchContourIdx = -1;
    float matchContourArea = 0;
	int upperLimitArea = 155000;
	int lowerLimitArea = 145000;
    cv::Mat drawing = cv::Mat::zeros( dst.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        cv::Scalar color = cv::Scalar(0, 100, 0);
        drawContours( drawing, contours, i, color, 1, 8, hierarchy, 0, cv::Point() );

        float ctArea= cv::contourArea(contours[i]);
        if(ctArea > lowerLimitArea && ctArea < upperLimitArea)
        {
            matchContourArea = ctArea;
            matchContourIdx = i;
        }
    }

    // if no contour found
    if(matchContourIdx < 0)
    {
        std::cout << "no contour found" << std::endl;
        return 1;
    }
	printf("contourIdx = %d, contourArea = %f\n", matchContourIdx, matchContourArea);
    // compute the rotated bounding rect of the biggest contour! (this is the part that does what you want/need)
    cv::RotatedRect boundingBox = cv::minAreaRect(contours[matchContourIdx]);
    // one thing to remark: this will compute the OUTER boundary box, so maybe you have to erode/dilate if you want something between the ragged lines



    // draw the rotated rect
    cv::Point2f corners[4];
    boundingBox.points(corners);
    cv::line(drawing, corners[0], corners[1], cv::Scalar(255,255,255));
    cv::line(drawing, corners[1], corners[2], cv::Scalar(255,255,255));
    cv::line(drawing, corners[2], corners[3], cv::Scalar(255,255,255));
    cv::line(drawing, corners[3], corners[0], cv::Scalar(255,255,255));

    // display
    //cv::imshow("input", input);
    cv::imshow("drawing", drawing);

    cv::imwrite("rotatedRect.png",drawing);

	/// Wait until user exit program by pressing a key
	waitKey(0);

	return 0;
}
///**
	// * @function CannyThreshold
	// * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
	// */
void CannyThreshold( int, void*)
{
	/// Reduce noise with a kernel 3x3
	blur( globalMat, detected_edges, Size(3,3) );

	/// Canny detector
	Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);

	globalMat.copyTo( dst, detected_edges);
	resize(dst, dst, Size(), 0.5, 0.5);
	imshow( window_name, dst );
}

void Threshold(int, void*)
{
	threshold(globalMat, dst, lowThreshold, 255, CV_THRESH_BINARY);
	resize(dst, dst, Size(800, 1000));
	imshow(window_name, dst);
}

void Dilate(int, void*)
{
	Mat dilatedImg;
	Mat element = getStructuringElement( MORPH_CROSS,
						Size( dilateSize, dilateSize ),
						Point( 0, 0 ) );
	/// Apply the dilation operation
	dilate( dst, dst, element );
	imshow( window_name, dst );

}

void Erose(int, void*)
{
	//Mat erosedImg;
	Mat element = getStructuringElement( MORPH_CROSS,
						Size( erodeSize, erodeSize ),
						Point( 0, 0 ) );
	/// Apply the dilation operation
	erode( dst, dst, element );
	imshow( "erosed image", dst );

}

///------------------------------------------------------------------
//erosion and dilation test

/// Global variables
//Mat src, erosion_dst, dilation_dst;
//
//int erosion_elem = 0;
//int erosion_size = 0;
//int dilation_elem = 0;
//int dilation_size = 0;
//int const max_elem = 2;
//int const max_kernel_size = 21;
//
///** Function Headers */
//void Erosion( int, void* );
//void Dilation( int, void* );
//
///**
// * @function main
// */
//int _tmain( int, char** argv )
//{
//  /// Load an image
//  src = cvLoadImage("D:\\hand.jpg");//imread( argv[1] );
//
//  if( src.empty() )
//    { return -1; }
//
//  /// Create windows
//  namedWindow( "Erosion Demo", WINDOW_AUTOSIZE );
//  namedWindow( "Dilation Demo", WINDOW_AUTOSIZE );
//  moveWindow( "Dilation Demo", src.cols, 0 );
//
//  /// Create Erosion Trackbar
//  createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo",
//          &erosion_elem, max_elem,
//          Erosion );
//
//  createTrackbar( "Kernel size:\n 2n +1", "Erosion Demo",
//          &erosion_size, max_kernel_size,
//          Erosion );
//
//  /// Create Dilation Trackbar
//  createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
//          &dilation_elem, max_elem,
//          Dilation );
//
//  createTrackbar( "Kernel size:\n 2n +1", "Dilation Demo",
//          &dilation_size, max_kernel_size,
//          Dilation );
//
//  /// Default start
//  Erosion( 0, 0 );
//  Dilation( 0, 0 );
//
//  waitKey(0);
//  return 0;
//}
//
///**
// * @function Erosion
// */
//void Erosion( int, void* )
//{
//  int erosion_type = 0;
//  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
//  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
//  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
//
//  Mat element = getStructuringElement( erosion_type,
//                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
//                       Point( erosion_size, erosion_size ) );
//  /// Apply the erosion operation
//  erode( src, erosion_dst, element );
//  imshow( "Erosion Demo", erosion_dst );
//}
//
///**
// * @function Dilation
// */
//void Dilation( int, void* )
//{
//  int dilation_type = 0;
//  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
//  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
//  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
//
//  Mat element = getStructuringElement( dilation_type,
//                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
//                       Point( dilation_size, dilation_size ) );
//  /// Apply the dilation operation
//  dilate( src, dilation_dst, element );
//  imshow( "Dilation Demo", dilation_dst );
//}
//------------------------------------------------------------------
//
////convex hull test
//Mat src; Mat src_gray;
//int thresh = 100;
//int max_thresh = 255;
//RNG rng(12345);
//
///// Function header
//void thresh_callback(int, void* );
//
///**
// * @function main
// */
//int _tmain( int, char** argv )
//{
//  /// Load source image and convert it to gray
//  src = cvLoadImage("D:\\hand.jpg");//imread( argv[1], 1 );
//
//  /// Convert image to gray and blur it
//  cvtColor( src, src_gray, COLOR_BGR2GRAY );
//  blur( src_gray, src_gray, Size(3,3) );
//
//  /// Create Window
//  const char* source_window = "Source";
//  namedWindow( source_window, WINDOW_AUTOSIZE );
//  imshow( source_window, src );
//
//  createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
//  thresh_callback( 0, 0 );
//
//  waitKey(0);
//  return(0);
//}
//
///**
// * @function thresh_callback
// */
//void thresh_callback(int, void* )
//{
//  Mat src_copy = src.clone();
//  Mat threshold_output;
//  vector<vector<Point> > contours;
//  vector<Vec4i> hierarchy;
//
//  /// Detect edges using Threshold
//  threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );
//
//  /// Find contours
//  findContours( threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
//
//  /// Find the convex hull object for each contour
//  vector<vector<Point> >hull( contours.size() );
//  for( size_t i = 0; i < contours.size(); i++ )
//     {   convexHull( Mat(contours[i]), hull[i], false ); }
//
//  /// Draw contours + hull results
//  Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
//  for( size_t i = 0; i< contours.size(); i++ )
//     {
//       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
//       drawContours( drawing, contours, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
//       drawContours( drawing, hull, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
//     }
//
//  /// Show in a window
//  namedWindow( "Hull demo", WINDOW_AUTOSIZE );
//  imshow( "Hull demo", drawing );
//}



////----------------------------------------------------------------
//// contour example
//int _tmain(int argc, const char * argv[]) {
//
//    cv::Mat image= cv::imread("../inputData/7.jpg");
//    if (!image.data) {
//        std::cout << "Image file not found\n";
//        return 1;
//    }
//
//    //Prepare the image for findContours
//    cv::cvtColor(image, image, CV_BGR2GRAY);
//    cv::threshold(image, image, 128, 255, CV_THRESH_BINARY);
//
//    //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
//    std::vector<std::vector<cv::Point> > contours;
//    cv::Mat contourOutput = image.clone();
//    cv::findContours( contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE );
//
//    //Draw the contours
//    cv::Mat contourImage(image.size(), CV_8UC3, cv::Scalar(0,0,0));
//    cv::Scalar colors[3];
//    colors[0] = cv::Scalar(255, 0, 0);
//    colors[1] = cv::Scalar(0, 255, 0);
//    colors[2] = cv::Scalar(0, 0, 255);
//    for (size_t idx = 0; idx < contours.size(); idx++) {
//        cv::drawContours(contourImage, contours, idx, colors[idx % 3]);
//    }
//
//    cv::imshow("Input Image", image);
//    cvMoveWindow("Input Image", 0, 0);
//    cv::imshow("Contours", contourImage);
//    cvMoveWindow("Contours", 200, 0);
//    cv::waitKey(0);
//
//    return 0;
//}
//
////---------------------------------------------------------------------------


/*			
int main()				// show a image and draw different patterns
{
	IplImage * img=cvLoadImage("D:\\lena.bmp");
	cvShowImage("a", img);
	cvWaitKey(0);
 
 	cvCircle(img, cvPoint(100,100), 50, cvScalar(255,0,0));
 	cvLine(img, cvPoint(100,100), cvPoint(200,200), cvScalar(0,255,0));
	cvRectangle(img, cvPoint(100,100), cvPoint(200,200), cvScalar(0,0,255));
	cvShowImage("a",img);
	cvWaitKey(0);
 
	return 0;	
}
*/

// OpenCV_test.cpp : Defines the entry point for the console application.
//
/*
#include "stdafx.h"
//opencv2 include
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int _tmain(int argc, _TCHAR* argv[])
{
//read an image
cv::Mat img = cv::imread("D:\\lena.bmp");

//create image window named
cv::namedWindow("My Image"); 

//show the image on window
cv::imshow("My Image", img);

//waitkey 
cv::waitKey(0);


return 0;
}
*/
 