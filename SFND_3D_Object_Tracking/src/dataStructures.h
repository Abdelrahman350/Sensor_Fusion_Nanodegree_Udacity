
#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <map>
#include <opencv2/core.hpp>

struct LidarPoint { // single lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
};

struct BoundingBox { // bounding box around a classified object (contains both 2D and 3D data)
    
    int boxID; // unique identifier for this bounding box
    int trackID; // unique identifier for the track to which this bounding box belongs
    
    cv::Rect roi; // 2D region-of-interest in image coordinates
    int classID; // ID based on class file provided to YOLO framework
    double confidence; // classification trust

    std::vector<LidarPoint> lidarPoints; // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints; // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches; // keypoint matches enclosed by 2D roi
};

struct DataFrame { // represents the available sensor information at the same time instance
    
    cv::Mat cameraImg; // camera image
    
    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes; // ROI around detected objects in 2D image coordinates
    std::map<int,int> bbMatches; // bounding box matches between previous and current frame
};
/*******************/
// Structure to represent node of kd tree
// Structure to represent node of kd tree
struct Node
{
  LidarPoint point;
  int id;
  Node* left;
  Node* right;

  Node(LidarPoint arr, int setId)
    : point(arr), id(setId), left(NULL), right(NULL)
    {}
};

struct KdTree
{
  Node* root;
  int kDimension = 3;
  KdTree()
    : root(NULL)
    {}
    
  void insertHelper(Node** node, uint depth, LidarPoint point, int id)
  {
    // Tree is empty
    if(*node==NULL){
      *node = new Node(point,id);
    }
    else
    {
      // Calculate current dim
      uint cd = depth % kDimension;
        
      if(cd == 0){
        if(point.x < (*node)->point.x){
           insertHelper(&(*node)->left, depth+1, point, id);
         }
         else{
           insertHelper(&(*node)->right, depth+1, point, id);
         }
      }
      else if(cd == 1){
        if(point.y < (*node)->point.y){
           insertHelper(&(*node)->left, depth+1, point, id);
         }
         else{
           insertHelper(&(*node)->right, depth+1, point, id);
         }
      }
      else if(cd == 2){
        if(point.z < (*node)->point.z){
           insertHelper(&(*node)->left, depth+1, point, id);
         }
         else{
           insertHelper(&(*node)->right, depth+1, point, id);
         }
      }
    }
  }

  void insert(LidarPoint point, int id)
  {
    // TODO: Fill in this function to insert a new point into the tree
    // the function should create a new node and place correctly with in the root
    insertHelper(&root,0,point,id);
  }

  // return a list of point ids in the tree that are within distance of target
  void searchHelper(LidarPoint target, Node* node, int depth, float distanceTol, std::vector<int>& ids)
  {
    if(node != NULL)
    {
      float deltaX = fabs(node->point.x - target.x);
      float deltaY = fabs(node->point.y - target.y);

      if( deltaX <= distanceTol && deltaY <= distanceTol)
      {
        float distance = sqrt(deltaX*deltaX + deltaY*deltaY);
        if(distance <= distanceTol)
        {
          ids.push_back(node->id);
        }
      }
      
      if(depth%kDimension == 0){
        if((target.x - distanceTol) < node->point.x){
          searchHelper(target, node->left, depth+1, distanceTol, ids);
        }
        if((target.x + distanceTol) > node->point.x){
          searchHelper(target, node->right, depth+1, distanceTol, ids);
        }
      }
      else if(depth%kDimension == 1){
        if((target.y - distanceTol) < node->point.y){
          searchHelper(target, node->left, depth+1, distanceTol, ids);
        }
        if((target.y + distanceTol) > node->point.y){
          searchHelper(target, node->right, depth+1, distanceTol, ids);
        }
      }
      else if(depth%kDimension == 2){
        if((target.z - distanceTol) < node->point.z){
          searchHelper(target, node->left, depth+1, distanceTol, ids);
        }
        if((target.z + distanceTol) > node->point.z){
          searchHelper(target, node->right, depth+1, distanceTol, ids);
        }
      }
    }
  }
  
  std::vector<int> search(LidarPoint target, float distanceTol)
  {
    std::vector<int> ids;
    searchHelper(target, root, 0, distanceTol, ids);
    return ids;
  }

};
#endif /* dataStructures_h */
