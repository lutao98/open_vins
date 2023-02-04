#ifndef LKTTRACKER_ORBDESCRIPTOR_H
#define LKTTRACKER_ORBDESCRIPTOR_H

// based on ORB_SLAM2::ORBextractor

#include <opencv2/opencv.hpp>
#include <vector>

namespace AMCVIO {
class ORBdescriptor {
public:
  ORBdescriptor(int _edgeThreshold = 19, int _patchSize = 31);

  ~ORBdescriptor() {}

  bool computeDescriptors(const cv::Mat &image, const std::vector<cv::Point2f> &pts, cv::Mat &descriptors);

  static int computeDescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;

    for (int i = 0; i < 8; i++, pa++, pb++) {
      unsigned int v = *pa ^ *pb;
      v = v - ((v >> 1) & 0x55555555);
      v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
      dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
  }

  static const float factorPI;

private:
  int edgeThreshold;
  int patchSize;
  int halfPatchSize;

  std::vector<cv::Point> pattern;
  std::vector<int> umax;

  cv::Mat mExpandedImage;
  cv::Mat mBluredImage;

private:
  // compute descriptor of a key point and save it into @desc
  void computeOrbDescriptor(const cv::KeyPoint &kpt, uchar *desc);

  // initialize layer information and image of pyramid
  void initializeIamge(const cv::Mat &image);

public:
  // calculate Angle for an ordinary point
  float IC_Angle(const int &levels, const cv::Point2f &pt);
};

} // namespace AMCVIO

#endif // LKTTRACKER_ORBDESCRIPTOR_H