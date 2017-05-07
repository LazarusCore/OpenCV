//Include

#include <opencv2\opencv.hpp>
#include <iostream>



//Namespaces

using namespace cv;

//Function prototypes
Mat edgeDet(Mat input, int width, int height, float sigma, float thresLow, float thresHigh);	//Canny style edge detection
void findEdge(int rowShift, int colShift, int row, int col, int dir, int lowerThreshold);
void suppressNonMax(int rowShift, int colShift, int row, int col, int dir, int lowerThreshold);


//Constants
const char* FILE_NAME = "image1.jpg";

//Globals
int	imageX, imageY;												//Image Resoltion
float sigma = 1.0;												//Canny Filter Standard deviation
int gradX;														// Sum of Sobel mask products values in the x direction
int gradY;														// Sum of Sobel mask products values in the y direction
int gradXmask[3][3] = { {-1,0,1}, {-2,0,2}, {-1,0,-1} };			// Sobel mask in the x direction
int gradYmask[3][3] = { {1,2,1}, {0,0,0}, {-1,-2,-1} };			// Sobel mask in the y direction
int edgeDirection[240][320];									// Stores the edge direction of each pixel
float pixGradient[240][320];									// Stores the gradient strength of each pixel



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

void edgeDet(Mat input, int width, int height, float sigma, float thresLow, float thresHigh)
{
	short gradX, gradY;											//X and Y Gradients
	unsigned short gradMagnitude;								//Gradient Magnitude
	Mat nonMax;													//Non-Maximal Suppression result
	Mat buffer;													//Buffer

	Mat blur_image;
	Mat gBlur;

	Mat kernel = (Mat_<double>(5, 5) << 1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1);	//Gaussian Kernel
	kernel = kernel / 256;										//Normalised

	filter2D(input, blur_image, -1, kernel, Point(-1, -1), 0);	//Convolve for Gaussian Blur creation

	imwrite("../../BMaxwellOpenCV/Blur_Image.jpg", blur_image);

	namedWindow("Blur Image", CV_WINDOW_NORMAL);
	imshow("Blur Image", blur_image);
	
	Mat edge_image;

	return edge_image;
}

void findEdge(int rowShift, int colShift, int row, int col, int dir, int lowerThreshold)
{
}

void suppressNonMax(int rowShift, int colShift, int row, int col, int dir, int lowerThreshold)
{
}
