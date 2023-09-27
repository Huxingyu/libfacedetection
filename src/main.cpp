#include "facedetectcnn.h"
#include <opencv2/opencv.hpp>

#define DETECT_BUFFER_SIZE 0x20000

using namespace cv;

int main()
{
    unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
    Mat image = imread("C:/Users/huxingyu/Desktop/libfacedetection/build/Release/2.png");
    int * pResults = facedetect_cnn(pBuffer, (unsigned char*)(image.ptr(0)), image.cols, image.rows, (int)image.step);
    int faceCount = *pResults;
    std::cout<<faceCount;
    Mat result_image = image.clone();
    for (int i = 0; i < faceCount; i++) {
        short * p = ((short*)(pResults + 1)) + 16*i;

        // int x = p[0];
        // int y = p[1];
        // int w = p[2];
        // int h = p[3];
        int confidence = p[0];
		int x = p[1];
		int y = p[2];
		int w = p[3];
		int h = p[4];

        //show the score of the face. Its range is [0-100]
        char sScore[256];
        snprintf(sScore, 256, "%d", confidence);
        putText(result_image, sScore, cv::Point(x, y-3), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        //draw face rectangle
		rectangle(result_image, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
        //draw five face landmarks in different colors
        cv::circle(result_image, cv::Point(p[5], p[5 + 1]), 1, cv::Scalar(255, 0, 0), 2);
        cv::circle(result_image, cv::Point(p[5 + 2], p[5 + 3]), 1, cv::Scalar(0, 0, 255), 2);
        cv::circle(result_image, cv::Point(p[5 + 4], p[5 + 5]), 1, cv::Scalar(0, 255, 0), 2);
        cv::circle(result_image, cv::Point(p[5 + 6], p[5 + 7]), 1, cv::Scalar(255, 0, 255), 2);
        cv::circle(result_image, cv::Point(p[5 + 8], p[5 + 9]), 1, cv::Scalar(0, 255, 255), 2);
        
        //print the result
        printf("face %d: confidence=%d, [%d, %d, %d, %d] (%d,%d) (%d,%d) (%d,%d) (%d,%d) (%d,%d)\n", 
                i, confidence, x, y, w, h, 
                p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13],p[14]);
    }
    cv::imshow("Face Detection", result_image);
    cv::waitKey();
    free(pBuffer);
    return 0;
}