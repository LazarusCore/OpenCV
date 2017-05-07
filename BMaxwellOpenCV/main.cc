//Include

#include <opencv2\opencv.hpp>
#include <iostream>
#include <math.h>



//Namespaces

using namespace cv;

//Function prototypes
Mat edgeDet(Mat input, unsigned int width, unsigned int height, float sigma, float thresLow, float thresHigh);	//Canny style edge detection
Mat sobelEdge(Mat input, unsigned int width, unsigned int height, float thresLow, float thresHigh);
void findEdge(int rowShift, int colShift, int row, int col, int dir, float thresLow);
void suppressNonMax(int rowShift, int colShift, int row, int col, int dir, float thresLow);


//Constants
const char* FILE_NAME = "image1.jpg";

//Globals
unsigned int imageX, imageY;									//Image Resoltion
float sigma = 1.0;												//Canny Filter Standard deviation
int gradX;														// Sum of Sobel mask products values in the x direction
int gradY;														// Sum of Sobel mask products values in the y direction
int GxMask[3][3] = { {-1,0,1}, {-2,0,2}, {-1,0,-1} };			// Sobel mask in the x direction
int GyMask[3][3] = { {1,2,1}, {0,0,0}, {-1,-2,-1} };			// Sobel mask in the y direction
int edgeDirection[3072][1728];									// Stores the edge direction of each pixel
double pixGradient[3072][1728];									// Stores the gradient strength of each pixel




int main(int argc, char* args[]) {

	Mat	src_image = imread(FILE_NAME, CV_LOAD_IMAGE_COLOR);		//Loads the image1.jpg into the variable src_image
	imageX = src_image.size().width;
	imageY = src_image.size().height;

	Mat gray_image;
	cvtColor(src_image, gray_image, CV_BGR2GRAY);				//Creates a dupilcate of the image and converts to gray scale

	imwrite("../../BMaxwellOpenCV/Gray_Image.jpg", gray_image);
	
	namedWindow("Image", CV_WINDOW_NORMAL);						//Opens images for comparison
	namedWindow("Gray Image", CV_WINDOW_NORMAL);

	imshow("Image", src_image);
	imshow("Gray Image", gray_image);

	Mat edge_image;												//Creates a file to house the edges

	edge_image = edgeDet(gray_image, imageX, imageY, sigma, 0.4, 0.7); //Runs the Edge Detection algorithm
	
	namedWindow("Edges", CV_WINDOW_NORMAL);
	//imshow("Edges", edge_image);

	waitKey(0);
	
	destroyAllWindows();

	return 0;
}

Mat edgeDet(Mat input, unsigned int width, unsigned int height, float sigma, float thresLow, float thresHigh){

	Mat nonMax;													//Non-Maximal Suppression result
	Mat buffer;													//Buffer

	Mat blur_image;
	Mat sobel_image;



	Mat kernel = (Mat_<double>(5, 5) << 1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1);	//Gaussian Kernel
	kernel = kernel / 256;										//Normalised

	filter2D(input, blur_image, -1, kernel, Point(-1, -1), 0);	//Convolve for Gaussian Blur creation

	imwrite("../../BMaxwellOpenCV/Blur_Image.jpg", blur_image);

	namedWindow("Blur Image", CV_WINDOW_NORMAL);
	imshow("Blur Image", blur_image);
	
	sobel_image = sobelEdge(blur_image, width, height, thresLow, thresHigh);
	imwrite("../../BMaxwellOpenCV/Sobel_Image.jpg", sobel_image);
	namedWindow("Sobel Image", CV_WINDOW_NORMAL);
	imshow("Sobel Image", sobel_image);

	return blur_image;

}

Mat sobelEdge(Mat input, unsigned int width, unsigned int height, float thresLow, float thresHigh) {

	unsigned int row, col;
	int xOffset = 0;
	int yOffset = 0;
	int xTotal = 0;
	int yTotal = 0;
	float origAngle = 0;
	int newAngle = 0;
	Mat sobel_image;
	bool edgeEnd;

	// Determine edge directions and gradient strengths
	for (row = 1; row < height - 1; row++) {
		for (col = 1; col < width - 1; col++) {
			gradX = 0;
			gradY = 0;
			// Calculate the sum of the Sobel mask times the nine surrounding pixels in the x and y direction 
			for (yOffset = -1; yOffset <= 1; yOffset++) {
				for (xOffset = -1; xOffset <= 1; xOffset++) {
					yTotal = row + yOffset;
					xTotal = col + xOffset;
					gradX = gradX + (input.at<uchar>(Point(xTotal, yTotal)) * GxMask[yOffset + 1][xOffset + 1]);
					gradY = gradY + (input.at<uchar>(Point(xTotal, yTotal)) * GyMask[yOffset + 1][xOffset + 1]);
				}
			}

			pixGradient[row][col] = sqrt(pow(gradX, 2.0) + pow(gradY, 2.0));	// Calculate gradient strength			
			origAngle = (atan2(gradX, gradY) / 3.14159) * 180.0;				// Calculate actual direction of edge

			//Convert actual edge direction to approximate value
			if (((origAngle < 22.5) && (origAngle > -22.5)) || (origAngle > 157.5) || (origAngle < -157.5))
				newAngle = 0;
			if (((origAngle > 22.5) && (origAngle < 67.5)) || ((origAngle < -112.5) && (origAngle > -157.5)))
				newAngle = 45;
			if (((origAngle > 67.5) && (origAngle < 112.5)) || ((origAngle < -67.5) && (origAngle > -112.5)))
				newAngle = 90;
			if (((origAngle > 112.5) && (origAngle < 157.5)) || ((origAngle < -22.5) && (origAngle > -67.5)))
				newAngle = 135;

			edgeDirection[row][col] = newAngle;									// Store the approximate edge direction of each pixel in one array
		}
	}

	// Trace along all the edges in the image
	for (row = 1; row < height - 1; row++) {
		for (col = 1; col < width - 1; col++) {
			edgeEnd = false;
			if (pixGradient[row][col] > thresHigh) {							// Check to see if current pixel has a high enough gradient strength to be part of an edge
				
				// Switch based on current pixel's edge direction
				switch (edgeDirection[row][col]) {
				case 0:
					findEdge(0, 1, row, col, 0, thresLow);
					break;
				case 45:
					findEdge(1, 1, row, col, 45, thresLow);
					break;
				case 90:
					findEdge(1, 0, row, col, 90, thresLow);
					break;
				case 135:
					findEdge(1, -1, row, col, 135, thresLow);
					break;
				default:
					sobel_image.at<uchar>(Point(row,col)) = 0;
					break;
				}
			}
			else {
				sobel_image.at<uchar>(Point(row, col)) = 0;
			}
		}
	}

	// Suppress any pixels not changed by the edge tracing
	for (row = 0; row < height; row++) {
		for (col = 0; col < width; col++) {
			// If a pixel's grayValue is not black or white make it black
			if ((sobel_image.at<uchar>(Point(row, col)) != 255) && (sobel_image.at<uchar>(Point(row, col)) != 0)) {
				sobel_image.at<uchar>(Point(row, col)) = 0;
			}
		}
	}
	return sobel_image;
}


void findEdge(int rowShift, int colShift, int row, int col, int dir, float thresLow)
{
}

void suppressNonMax(int rowShift, int colShift, int row, int col, int dir, float thresLow)
{
}
