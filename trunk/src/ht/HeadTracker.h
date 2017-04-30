
// Billy Araujo, August 2011

#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "Tracker.h"
using namespace FACETRACKER;

namespace faceapi
{

    /*! @brief Defines a coordinate in 3 dimensions using floating point values. */
    struct pos3f {
        float x;  /*!< Coordinate X value */
        float y;  /*!< Coordinate Y value */
        float z;  /*!< Coordinate Z value */
    };

    /*! @brief Euler angle representation of orientation. All values are in radians.

    The euler angles are represented in the X-Y-Z convention.
    @see
    - http://en.wikipedia.org/wiki/Euler_angles */
    struct rotEuler {
        float x_rads;  /*!< Rotation angle around X-axis, in radians */
        float y_rads;  /*!< Rotation angle around Y-axis, in radians */
        float z_rads;  /*!< Rotation angle around Z-axis, in radians */
    };

    struct headPoseData
    {
        pos3f head_pos;        /*!< Position of the head relative to the camera. */
        pos3f left_eye_pos;    /*!< Position of the left eyeball center relative to the camera. */
        pos3f right_eye_pos;   /*!< Position of the right eyeball center relative to the camera. */
        rotEuler head_rot;     /*!< Rotation of the head around the X,Y and Z axes respectively in euler angles. @see http://en.wikipedia.org/wiki/Euler_angles */
        float confidence;        /*!< Confidence of the head-pose measurement [0..1]. A value of 0 indicates that the measurements cannot be trusted and may have undefined values. */
    };

    class HeadTracker{

    public:
        HeadTracker();
        ~HeadTracker();

        void start();

        bool facesInImage(IplImage *image);

        IplImage *getNextFrame();
        void imagePush(IplImage* iplimage);

        bool hasFailed();

        void setTolerance(float fTol);
        void setNumberIterations(int nIter);

        headPoseData HeadTracker::getHeadPose();

        cv::Mat getShape();
        cv::Mat getCon();
        cv::Mat getTri();
        cv::Mat getModel();

        void getWorldCoordMatrices(float fovy, float aspect, float zNear, float zFar, float *modelView, float *projection);
        void getFaceToWorldTransformMatrix(float *f2w_matrix);

    protected:

        bool m_failed;
        double m_scale;

        IplImage *m_image;
        CvCapture* m_camera;

        std::vector<int> m_wSize1;
        std::vector<int> m_wSize2;

        Tracker *m_tracker;
        cv::Mat m_tri, m_con;

        cv::Mat im, frame, gray;

        int m_fpd; 
        int m_nIter;
        double m_clamp;
        double m_fTol;
        bool m_fcheck; 

        headPoseData m_head_pose; 

        CvMemStorage* m_storage;
        CvHaarClassifierCascade *m_cascadeFrontal, *m_cascadeProfile;

    };

}

