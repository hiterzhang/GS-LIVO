#include "LIVMapper.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  // Keep the mapper process-scoped. Some upstream PCL-owned clouds are
  // corrupted during long GS-LIVO runs and crash only when their member
  // destructors execute after all requested artifacts have been saved.
  // The operating system reclaims this memory when this ROS process exits.
  auto *mapper = new LIVMapper(nh);
  mapper->initializeSubscribersAndPublishers(nh, it);
  mapper->run();
  return 0;
}
