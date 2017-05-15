//Include

#include <opencv2\opencv.hpp>
#include <math.h>
#include <stdio.h>
#include <vector>

//Benson Maxwell: 01504282

//Namespaces

using namespace cv;
using namespace std;

//Function prototypes
void edgeDet(Mat& input, int width, int height, float sigma);											//Canny style edge detection
void imagePro(int width, int height);																	//Canny Filter filter
void findEdge(int rowShift, int colShift, int row, int col, int direction, int width, int height);		//Find the edge provvided by the Sobel
void nonMaxima(int rowShift, int colShift, int row, int col, int direction, int width, int height);		//Remove nonMax edges
void houghCircle(Mat& original, int width, int height);

//Macros for greyscale
#define pixel(image ,y, x, step, channels) ( (uchar*) ((( image).data ) + (y) *(step) ) ) [( x)*(channels)]

//Constants
const char* FILE_NAME = "image1.jpg";

//Globals
int imageX, imageY;												// Image Resoltion
float sigma = 1.0;												// Canny Filter Standard deviation
int gradX;														// Sum of Sobel mask products values in the x direction
int gradY;														// Sum of Sobel mask products values in the y direction
int GxMask[3][3] = { {-3,0,3}, {-10,0,10}, {-3,0,-3} };			// Sobel mask in the x direction
int GyMask[3][3] = { {3,10,3}, {0,0,0}, {-3,-10,-3} };			// Sobel mask in the y direction
int edgeDirection[3072][1728];									// Stores the edge direction of each pixel
int pixGradient[3072][1728];									// Stores the gradient strength of each pixel
Mat impro_image(1728, 3072, CV_8UC1);							// Mat file for Sobel Operator
Mat grey_image(1728, 3072, CV_8UC1);							// Mat file for Gray scale Operator
Mat blur_image(1728, 3072, CV_8UC1);							// Gaussian Blur result
Mat hough_image(1728, 3072, CV_8UC1);							// Hough result
float thresLow = 0.100;											// Lower Threshold
float thresHigh = 0.5;											// Upper Threshold
int step = impro_image.step;
int channels = impro_image.channels();
int radMin = 130;												// Min radius to look for
int radMax = 195;												// Max Radius to look for
vector<Vec3i> houghAcc;											// Hough Accumulator (Row, Col, Radius)


int main(int argc, char* args[]) {

	Mat src_image = imread(FILE_NAME, CV_LOAD_IMAGE_COLOR);		// Loads the image1.jpg into the variable src_image
	imageX = src_image.size().width;
	imageY = src_image.size().height;


	cvtColor(src_image, grey_image, CV_BGR2GRAY);				// Creates a dupilcate of the image and converts to gray scale

	imwrite("../../BMaxwellOpenCV/Grey_Image.jpg", grey_image);
	
	//namedWindow("Image", CV_WINDOW_NORMAL);						// Opens images for comparison
	//namedWindow("Grey Image", CV_WINDOW_NORMAL);

	//imshow("Image", src_image);
	//imshow("Grey Image", grey_image);

	edgeDet(grey_image, imageX, imageY, sigma); //Runs the Edge Detection algorithm

	waitKey(0);
	
	destroyAllWindows();

	return 0;
}

void edgeDet(Mat &input, int width, int height, float sigma){

	Mat kernel = (Mat_<double>(5, 5) << 1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1);	//Gaussian Kernel
	kernel = kernel / 350;										//Normalised

	filter2D(input, blur_image, -1, kernel, Point(-1, -1), 0);	//Convolve for Gaussian Blur creation

	imwrite("../../BMaxwellOpenCV/Blur_Image.jpg", blur_image);
	//namedWindow("Blur Image", CV_WINDOW_NORMAL);
	//imshow("Blur Image", blur_image);
	
	imagePro(width, height);	//Sobel Function


	imwrite("../../BMaxwellOpenCV/impro_Image.jpg", impro_image);
	//namedWindow("Canny Image", CV_WINDOW_NORMAL);
	//imshow("Canny Image", impro_image);

	houghCircle(input, width, height);

	return ;

}

void imagePro(int width, int height){

	unsigned int row, col;
	int xOffset = 0;
	int yOffset = 0;
	int xTotal = 0;
	int yTotal = 0;
	float origAngle = 0;
	int newAngle = 0;
	
	bool edgeEnd;


	//initialise edgeDirection to 0 in all cells
	for (row = 0; row < height; row++) {
		for (col = 0; col < width; col++) {
			edgeDirection[col][row] = 0;
		}
	}

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
					gradX = gradX + (pixel(blur_image, yTotal, xTotal, step, channels) * GxMask[yOffset + 1][xOffset + 1]);
					gradY = gradY + (pixel(blur_image, yTotal, xTotal, step, channels) * GyMask[yOffset + 1][xOffset + 1]);
				}
			}

			pixGradient[col][row] = sqrt(pow(gradX, 2.0) + pow(gradY, 2.0));	// Calculate gradient strength
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

			edgeDirection[col][row] = newAngle;							// Store the approximate edge direction of each pixel in one array
		}
	}
	
	// Trace along all the edges in the image
	for (row = 1; row < height - 1; row++) {
		for (col = 1; col < width - 1; col++) {
			edgeEnd = false;
			if (pixGradient[col][row] > thresHigh) {					// Check to see if current pixel has a high enough gradient strength to be part of an edge
				
				// Switch based on current pixel's edge direction
				switch (edgeDirection[col][row]) {
				case 0:
					findEdge(0, 1, row, col, 0, width, height);
					break;
				case 45:
					findEdge(1, 1, row, col, 45, width, height);
					break;
				case 90:
					findEdge(1, 0, row, col, 90, width, height);
					break;
				case 135:
					findEdge(1, -1, row, col, 135, width, height);
					break;
				default:
					pixel(impro_image, row, col, step, channels) = 0;
					break;
				}
			}
			else {
				pixel(impro_image, row, col, step, channels) = 0;
			}
		}
	}
	
	// Suppress any pixels not changed by the edge tracing
	for (row = 0; row < height; row++) {
		for (col = 0; col < width; col++) {
			// If a pixel's grayValue is not black or white make it white
			if ((pixel(impro_image, row, col, step, channels) != 255) && (pixel(impro_image, row, col, step, channels) != 0)) {
				pixel(impro_image, row, col, step, channels) = 0;
			}
		}
	}

	// Non-maximum Suppression
	for (row = 1; row < height - 1; row++) {
		for (col = 1; col < width - 1; col++) {

			if (pixel(impro_image, row, col, step, channels) == 255) {				// Check to see if current pixel is an edge
				// Switch based on current pixel's edge direction
				switch (edgeDirection[col][row]) {
				case 0:
					nonMaxima(1, 0, row, col, 0, width, height);
					break;
				case 45:
					nonMaxima(1, -1, row, col, 45, width, height);
					break;
				case 90:
					nonMaxima(0, 1, row, col, 90, width, height);
					break;
				case 135:
					nonMaxima(1, 1, row, col, 135, width, height);
					break;
				default:
					break;
				}
			}
		}
	}

	return ;
}


void findEdge(int rowShift, int colShift, int row, int col, int direction, int width, int height) {

	int newRow;
	int newCol;
	bool edgeEnd = false;


	//Find the row and column values for the next possible pixel on the edge
	if (colShift < 0) {
		if (col > 0)
			newCol = col + colShift;
		else
			edgeEnd = true;
	}
	else if (col < width - 1) 
		newCol = col + colShift;
	
	else
		edgeEnd = true;																// If the next pixel would be off image, don't do the while loop

	if (rowShift < 0) {
		if (row > 0)
			newRow = row + rowShift;
		else
			edgeEnd = true;
	}
	else if (row < height - 1)
	{
		newRow = row + rowShift;
	}
	else
		edgeEnd = true;

	//Determine if edge directions and gradient strengths correspond
	while ((edgeDirection[newCol][newRow] == direction) && !edgeEnd && (pixGradient[newCol][newRow] > thresLow)) {
		
		//Set the pixel to black
		pixel(impro_image, row, col, step, channels) = 255;							//need to pass this back and forward somehow   //uncomment this line when error is solved

		if (colShift < 0) {
			if (newCol > 0)
				newCol = newCol + colShift;
			else
				edgeEnd = true;
		}
		else if (newCol < width - 1) {
			newCol = newCol + colShift;
		}
		else
			edgeEnd = true;
		if (rowShift < 0) {
			if (newRow > 0)
				newRow = newRow + rowShift;
			else
				edgeEnd = true;
		}
		else if (newRow < height - 1) {
			newRow = newRow + rowShift;
		}
		else
			edgeEnd = true;
	}
}

void nonMaxima(int rowShift, int colShift, int row, int col, int direction, int width, int height){


	int newRow = 0;
	int newCol = 0;
	bool edgeEnd = false;
	int nonMax[3072][3];															// Temporarily stores gradients and positions of pixels in parallel edges
	int pixelCount = 0;																// Stores the number of pixels in parallel edges
	int count;
	int max[3];																		// Maximum point in a wide edge


	if (colShift < 0) {
		if (col > 0)
			newCol = col + colShift;
		else
			edgeEnd = true;
	}
	else if (col < width - 1) {
		newCol = col + colShift;
	}
	else
		edgeEnd = true;																// If the next pixel would be off image, don't do the while loop
	if (rowShift < 0) {
		if (row > 0)
			newRow = row + rowShift;
		else
			edgeEnd = true;
	}
	else if (row < height - 1) {
		newRow = row + rowShift;
	}
	else
		edgeEnd = true;

	// Find non-maximum parallel edges tracing up
	while ((edgeDirection[newCol][newRow] == direction) && !edgeEnd && (pixel(impro_image, newRow, newCol, step, channels) == 255)) {
		if (colShift < 0) {
			if (newCol > 0)
				newCol = newCol + colShift;
			else
				edgeEnd = true;
		}
		else if (newCol < width - 1) {
			newCol = newCol + colShift;
		}
		else
			edgeEnd = true;
		if (rowShift < 0) {
			if (newRow > 0)
				newRow = newRow + rowShift;
			else
				edgeEnd = true;
		}
		else if (newRow < height - 1) {
			newRow = newRow + rowShift;
		}
		else
			edgeEnd = true;
		nonMax[pixelCount][0] = newRow;
		nonMax[pixelCount][1] = newCol;
		nonMax[pixelCount][2] = pixGradient[newRow][newCol];
		pixelCount++;
	
	}

	// Find non-maximum parallel edges tracing down
	edgeEnd = false;
	colShift *= -1;
	rowShift *= -1;
	if (colShift < 0) {
		if (col > 0)
			newCol = col + colShift;
		else
			edgeEnd = true;
	}
	else if (col < width - 1) {
		newCol = col + colShift;
	}
	else
		edgeEnd = true;
	if (rowShift < 0) {
		if (row > 0)
			newRow = row + rowShift;
		else
			edgeEnd = true;
	}
	else if (row < height - 1) {
		newRow = row + rowShift;
	}
	else
		edgeEnd = true;

	while ((edgeDirection[newCol][newRow] == direction) && !edgeEnd && (pixel(impro_image, newRow, newCol, step, channels) == 255)) {
		if (colShift < 0) {
			if (newCol > 0)
				newCol = newCol + colShift;
			else
				edgeEnd = true;
		}
		else if (newCol < width - 1) {
			newCol = newCol + colShift;
		}
		else
			edgeEnd = true;
		if (rowShift < 0) {
			if (newRow > 0)
				newRow = newRow + rowShift;
			else
				edgeEnd = true;
		}
		else if (newRow < height - 1) {
			newRow = newRow + rowShift;
		}
		else
			edgeEnd = true;
		nonMax[pixelCount][0] = newRow;
		nonMax[pixelCount][1] = newCol;
		nonMax[pixelCount][2] = pixGradient[newRow][newCol];
		pixelCount++;

	}

	// Suppress non-maximum edges
	max[0] = 0;
	max[1] = 0;
	max[2] = 0;
	for (count = 0; count < pixelCount; count++) {
		if (nonMax[count][2] > max[2]) {
			max[0] = nonMax[count][0];
			max[1] = nonMax[count][1];
			max[2] = nonMax[count][2];
		}
	}
	for (count = 0; count < pixelCount; count++) {
		pixel(impro_image, newRow, newCol, step, channels) = 0;
	}

	return ;
}


void houghCircle(Mat &original, int width, int height) {

	int i, j, k;
	int x = 0;
	int y = 0;
	Vec3i tempVec;


	//For each possible radius, go through the impro_image and create the hough space.
	for (k = radMin; k <= radMax; k++) 
	{
		hough_image = Scalar(0);

		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				//If the pixel is black (edge) then carry find the circles that it could fall on
				if (pixel(impro_image, i, j, step, channels) == 255) 
				{	
					for (y = -k; ( i + y ) < (i + k); y++)
					{
						for (x = -k; (j + x) < (j + k); x++) 
						{
							if ((i + y < 0) || (j + x < 0))
							{
								if (pow(y, 2) + pow(x, 2) == pow(k, 2)) 
								{
									//Check if we have intensity overflow on the hough image.
									if (pixel(hough_image, (i + y), (j + x), step, channels) == 255)
									{
										break;
									}
									else
									{
										//increase the intensity of that pixel by 1
										pixel(hough_image, (i + y), (j + x), step, channels) += 1;
									}
								}
							}
						}
					}
				}
			}
		}
		//check the map for maxima, and place maxima into the hough accumulator
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				if (pixel(hough_image, i, j, step, channels) > 10)
				{
					tempVec[0] = i;													//Row
					tempVec[1] = j;													//Col
					tempVec[2] = k;													//Radius
					houghAcc.push_back(tempVec);
					
				}
			}
		}
	}

	for (i = 0; i < houghAcc.size(); i++)
	{
		circle(original, Point(houghAcc[i][0], houghAcc[i][1]), houghAcc[i][2], Scalar(0, 255, 255), 1, 8, 0);
	}

	namedWindow("Image", CV_WINDOW_NORMAL);
	imshow("Image", original);

}