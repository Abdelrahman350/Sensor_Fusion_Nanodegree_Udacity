
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}
/*
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    double dist_mean = 0;
    std::vector<cv::DMatch>  kptMatches_roi;
    for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        cv::KeyPoint kp = kptsCurr.at(it->trainIdx);
        if (boundingBox.roi.contains(cv::Point(kp.pt.x, kp.pt.y)) && boundingBox.roi.contains(kptsPrev.at(it->queryIdx).pt)) 
            kptMatches_roi.push_back(*it);
     }   
    for  (auto it = kptMatches_roi.begin(); it != kptMatches_roi.end(); ++it)  
         dist_mean += it->distance; 
    cout << "Find " << kptMatches_roi.size()  << " matches" << endl;
    if (kptMatches_roi.size() > 0)
         dist_mean = dist_mean/kptMatches_roi.size();  
    else return;
    
    double threshold = dist_mean * 0.7;
    for  (auto it = kptMatches_roi.begin(); it != kptMatches_roi.end(); ++it)
    {
       if (it->distance < threshold)
           boundingBox.kptMatches.push_back(*it);
    }
}/***/

// associate a given bounding box with the keypoints it contains

void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
  std::vector<cv::DMatch> kptMatchesROI;
  double sampleMean = 0;
  double sampleSTDV = 0;
  for(auto match = kptMatches.begin(); match != kptMatches.end(); ++match){
    cv::Point2f pointPrev = kptsPrev[match->queryIdx].pt;
    cv::Point2f pointCurr = kptsCurr[match->trainIdx].pt;
    if(boundingBox.roi.contains(pointPrev) && boundingBox.roi.contains(pointCurr)){
      kptMatchesROI.push_back(*match);
      sampleMean += match->distance;
    }
  }
  
  if(kptMatchesROI.size() > 0){
    sampleMean /= kptMatchesROI.size();
  }
  else{
    return;
  }
  
  for(auto match = kptMatchesROI.begin(); match != kptMatchesROI.end(); ++match){
    double delta = 0;
    delta = (match->distance - sampleMean);
    sampleSTDV += delta * delta;
  }
  sampleSTDV /= (kptMatchesROI.size() - 1);
  for(auto match = kptMatchesROI.begin(); match != kptMatchesROI.end(); ++match){
    if(abs(match->distance - sampleMean) < sampleSTDV){
      boundingBox.kptMatches.push_back(*match);
    }
  }
}
/***/
/*
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    double dist_mean = 0;
    std::vector<cv::DMatch>  kptMatches_roi;
    for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        cv::KeyPoint kp = kptsCurr.at(it->trainIdx);
        if (boundingBox.roi.contains(cv::Point(kp.pt.x, kp.pt.y))) 
            kptMatches_roi.push_back(*it);
     }   
    for  (auto it = kptMatches_roi.begin(); it != kptMatches_roi.end(); ++it)  
         dist_mean += it->distance; 
    cout << "Find " << kptMatches_roi.size()  << " matches" << endl;
    if (kptMatches_roi.size() > 0)
         dist_mean = dist_mean/kptMatches_roi.size();  
    else return;    
    double threshold = dist_mean * 0.7;        
    for  (auto it = kptMatches_roi.begin(); it != kptMatches_roi.end(); ++it)
    {
       if (it->distance < threshold)
           boundingBox.kptMatches.push_back(*it);
    }
    cout << "Leave " << boundingBox.kptMatches.size()  << " matches" << endl;
}/*****/  
// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
  // Compute distance ratios between all matched keypoints
  vector<double> distRatios;// stores the distance ratios for all keypoints between curr. and prev. frame
  for(auto it1 = kptMatches.begin(); it1 != kptMatches.end()-1; ++it1){
    // outer kpt. loop
    // get current keypoint and its matched partner in the prev. frame
    cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
    cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
    for(auto it2 = kptMatches.begin()+1; it2 != kptMatches.end(); ++it2){
      // inner kpt. loop
      double minDist = 100.0;// min required 
      // get next keypoint and its matched partner in the prev. frame
      cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
      cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);
      // compute distances and distances ratios
      double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
      double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
      if(distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist){
        double distRatio = distCurr / distPrev;
        distRatios.push_back(distRatio);
      }
    }
  }
  if(distRatios.size() == 0){
    TTC = NAN;
    return;
  }
  std::sort(distRatios.begin(), distRatios.end());
  long medIndex = floor(distRatios.size()/2.0);
  double medDistRatio = distRatios.size()%2 == 0 ? (distRatios[medIndex-1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex];
  double dT = 1 / frameRate;
  TTC = -dT / (1 - medDistRatio);
}/***/

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
  double dT = 1/frameRate;// time between two measurements in seconds
  double laneWidth = 4.0;// assumed width of the ego lane
  double minXPrev = 1e9;
  double minXCurr = 1e9;
  float distanceTol = 0.2;
  int minSize = 30;
  int maxSize = 250;
  // Take only the points within the ego lane and discard the others
  std::vector<LidarPoint> lidarPointsPrev_ego;
  std::vector<LidarPoint> lidarPointsCurr_ego;
  
  double sampleMeanXPrev = 0;
  double sampleSTDV_XPrev = 0;
  for(auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it){
    if(abs(it->y) <= laneWidth / 2.0){
      lidarPointsPrev_ego.push_back(*it);
      sampleMeanXPrev += it->x;
    }
  }
  sampleMeanXPrev /= lidarPointsPrev_ego.size();
  for(auto point = lidarPointsPrev_ego.begin(); point != lidarPointsPrev_ego.end(); ++point){
    double delta = (point->x - sampleMeanXPrev);
    sampleSTDV_XPrev += delta * delta;
  }
  sampleSTDV_XPrev /= (lidarPointsPrev_ego.size() - 1);
  for(auto point = lidarPointsPrev_ego.begin(); point != lidarPointsPrev_ego.end(); ++point){
    if((abs(point->x - sampleMeanXPrev) < sampleSTDV_XPrev) && (point->x < minXPrev)){
      minXPrev = point->x;
    }
  }
  
  double sampleMeanXCurr = 0;
  double sampleSTDV_XCurr = 0;
  for(auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it){
    if(abs(it->y) <= laneWidth / 2.0){
      lidarPointsCurr_ego.push_back(*it);
      sampleMeanXCurr += it->x;
    }
  }
  sampleMeanXCurr /= lidarPointsCurr_ego.size();
  for(auto point = lidarPointsCurr_ego.begin(); point != lidarPointsCurr_ego.end(); ++point){
    double delta = (point->x - sampleMeanXCurr);
    sampleSTDV_XCurr += delta * delta;
  }
  sampleSTDV_XCurr /= (lidarPointsCurr_ego.size() - 1);
  for(auto point = lidarPointsCurr_ego.begin(); point != lidarPointsCurr_ego.end(); ++point){
    if((abs(point->x - sampleMeanXCurr) < sampleSTDV_XCurr) && (point->x < minXCurr)){
      minXCurr = point->x;
    }
  }
  
  
  cout << "Prev min X = " << minXPrev << endl;
  cout << "Curr min X = " << minXCurr << endl;
  
  // compute TTC from both measurements
  TTC = minXCurr * dT / (minXPrev - minXCurr);
}

void clusterHelper(int indice, std::vector<LidarPoint> cloud, std::vector<int>& cluster, std::vector<bool>& processed, KdTree *tree, float distanceTol){
  processed[indice] = true;
  cluster.push_back(indice);
  std::vector<int> nearest = tree->search(cloud[indice], distanceTol);
  
  for(int id : nearest){
    if(!processed[id]){
      clusterHelper(id, cloud, cluster, processed, tree, distanceTol);
    }
  }
}

std::vector<std::vector<LidarPoint>> euclideanCluster(std::vector<LidarPoint> cloud, KdTree* tree, float distanceTol, int minSize, int maxSize)
{
  std::vector<std::vector<LidarPoint>> clusters;
  std::vector<bool> processed(cloud.size(), false);
  
  for(size_t idx = 0; idx < cloud.size(); ++idx){
    if(processed[idx] == false){
      std::vector<int> cluster_idx;
      std::vector<LidarPoint> cloudCluster;
      clusterHelper(idx, cloud, cluster_idx, processed, tree, distanceTol);
      
      if(cluster_idx.size() >= minSize && cluster_idx.size() <= maxSize){               
        for(int i = 0; i < cluster_idx.size(); i++){
          LidarPoint point;
          point = cloud[cluster_idx[i]];
          cloudCluster.push_back(point);
        }
        clusters.push_back(cloudCluster);
      }
      else{
        for(int i = 1; i < cluster_idx.size(); i++){
          processed[cluster_idx[i]] = false;
        }
      }
    }
  }
  return clusters;
}

/*
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
  double dT = 1/frameRate;// time between two measurements in seconds
  double laneWidth = 4.0;// assumed width of the ego lane
  double minXPrev = 1e9;
  double minXCurr = 1e9;
  float distanceTol = 0.2;
  int minSize = 30;
  int maxSize = 250;
  // Take only the points within the ego lane and discard the others
  std::vector<LidarPoint> lidarPointsPrev_ego;
  std::vector<LidarPoint> lidarPointsCurr_ego;
  
  for(auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it){
    if(abs(it->y) <= laneWidth / 2.0){
      lidarPointsPrev_ego.push_back(*it);
    }
  }
  for(auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it){
    if(abs(it->y) <= laneWidth / 2.0){
      lidarPointsCurr_ego.push_back(*it);
    }
  }
  
  // Clustering the lidar points in the ego laneWidth
  KdTree* treePrev = new KdTree;
  for(int i = 0; i < lidarPointsPrev_ego.size(); ++i){
    treePrev->insert(lidarPointsPrev_ego[i], i);
  }
  KdTree* treeCurr = new KdTree;
  for(int i = 0; i < lidarPointsCurr_ego.size(); ++i){
    treeCurr->insert(lidarPointsCurr_ego[i], i);
  }
  
  std::vector<std::vector<LidarPoint>> prevClusters;
  prevClusters = euclideanCluster(lidarPointsPrev_ego, treePrev, distanceTol, minSize, maxSize);
  std::vector<std::vector<LidarPoint>> currClusters;
  currClusters = euclideanCluster(lidarPointsCurr_ego, treeCurr, distanceTol, minSize, maxSize);
  
  std::vector<LidarPoint> maxCluster_prev;
  std::vector<LidarPoint> maxCluster_curr;
  for(auto cluster = prevClusters.begin(); cluster != prevClusters.end(); ++cluster){
    if(cluster->size() > maxCluster_prev.size()){
      maxCluster_prev = *cluster;
    }
  }
  for(auto cluster = currClusters.begin(); cluster != currClusters.end(); ++cluster){
    if(cluster->size() > maxCluster_curr.size()){
      maxCluster_curr = *cluster;
    }
  }
  
  for(auto it = maxCluster_prev.begin(); it != maxCluster_prev.end(); ++it){
    if(it->x < minXPrev){
      minXPrev = it->x;
    }
  }
  for(auto it = maxCluster_curr.begin(); it != maxCluster_curr.end(); ++it){
    if(it->x < minXCurr){
      minXCurr = it->x;
    }
  }
  
  cout << "Prev min X = " << minXPrev << endl;
  cout << "Curr min X = " << minXCurr << endl;
  
  // compute TTC from both measurements
  TTC = minXCurr * dT / (minXPrev - minXCurr);
}

/*template<typename KeyType, typename ValueType>
std::pair<KeyType, ValueType> get_max(const std::map<KeyType, ValueType>& x) {
    using pairtype = std::pair<KeyType, ValueType>;
    return *std::max_element(x.begin(), x.end(), [] (const pairtype & p1, const pairtype & p2) {
        return p1.second < p2.second;
    });
}/****/
/*void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    int lane_wide = 4;
    //just consider Lidar points within ego lane
    std::vector<float> ppx;
    std::vector<float> pcx;
    for(auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end() -1; ++it)
    {
        if(abs(it->y) < lane_wide/2) ppx.push_back(it->x);
    }
    for(auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end() -1; ++it)
    {
        if(abs(it->y) < lane_wide/2) pcx.push_back(it->x);
    }

    float min_px, min_cx;
    int p_size = ppx.size();
    int c_size = pcx.size();
    if(p_size > 0 && c_size > 0)
    {
        for(int i=0; i<p_size; i++)
        {
            min_px += ppx[i];
        }

        for(int j=0; j<c_size; j++)
        {
            min_cx += pcx[j];
        }
    }
    else 
    {
        TTC = NAN;
        return;
    }

    min_px = min_px /p_size;
    min_cx = min_cx /c_size;
    std::cout<<"lidar_min_px:"<<min_px<<std::endl;
    std::cout<<"lidar_min_cx:"<<min_cx<<std::endl;

    float dt = 1/frameRate;
    TTC = min_cx * dt / (min_px - min_cx);
}/****/


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
  // loop over the previos frame Bounding Boxes
  for(auto prevBBox = prevFrame.boundingBoxes.begin(); prevBBox != prevFrame.boundingBoxes.end(); ++prevBBox){
    std::map<int, int> prevCurr_match;
    for(auto currBBox = currFrame.boundingBoxes.begin(); currBBox != currFrame.boundingBoxes.end(); ++currBBox){
      for(auto match = matches.begin(); match != matches.end(); ++match){
        cv::KeyPoint prevKeyPoint = prevFrame.keypoints[match->queryIdx];
        if(prevBBox->roi.contains(prevKeyPoint.pt)){
          cv::KeyPoint currKeyPoint = currFrame.keypoints[match->trainIdx];
          if(currBBox->roi.contains(currKeyPoint.pt)){
            prevCurr_match[currBBox->boxID]++;
           
          }
        }
      }//eof loop over matches
      
    }//eof loop over current frame bounding boxes
    
    //Iterate to get the boxID with the maximum score 
    int maxID = -1;
    int maxScore = 0;
    
    for(auto it = prevCurr_match.begin(); it != prevCurr_match.end(); ++it){
      if(it->second > maxScore){
        maxID = it->first;
        maxScore = it->second;
      }
    }
    if(maxID == -1){
      continue;
    }
    bbBestMatches[prevBBox->boxID] = maxID;
    
    bool printMessage = true;
    if(printMessage){
      std::cout << "ID Matching: " << prevBBox->boxID << "  --->  " << maxID << std::endl;
    }
    
  }//eof loop over previous frame bounding boxes

}