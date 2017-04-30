
// Billy Araujo, August 2011

#include "HeadTracker.h"

namespace faceapi
{

    HeadTracker::HeadTracker() : 
        m_failed(true),
        m_scale(1.0),
        m_fpd(-1),
        m_nIter(10),
        m_clamp(3),
        m_fTol(0.01),
        m_fcheck(false)
    {

        m_camera = cvCreateCameraCapture(CV_CAP_ANY);

        m_tracker = new Tracker();

        m_wSize1.resize(1);
        m_wSize1[0] = 7;

        m_wSize2.resize(3);
        m_wSize2[0] = 11;
        m_wSize2[1] = 9;
        m_wSize2[2] = 7;

        std::string ftFile("model/face2.tracker");
        std::string triFile("model/face.tri");
        std::string conFile("model/face.con");

        m_tracker->Load(ftFile.c_str());
        m_tri = IO::LoadTri(triFile.c_str());
        m_con = IO::LoadCon(conFile.c_str());  // not being used right now

        // Load the HaarClassifierCascade
        m_cascadeFrontal = (CvHaarClassifierCascade*)cvLoad("model/haarcascade_frontalface_alt.xml", 0, 0, 0 );
        m_cascadeProfile = (CvHaarClassifierCascade*)cvLoad("model/haarcascade_profileface.xml", 0, 0, 0 );

        m_storage = cvCreateMemStorage(0);

    }

    void HeadTracker::setTolerance(float fTol)
    {

        m_fTol = fTol;

    }

    void HeadTracker::setNumberIterations(int nIter)
    {

        m_nIter = nIter;

    }

    HeadTracker::~HeadTracker()
    {

        cvReleaseCapture(&m_camera);

        delete m_tracker;

    }

    void HeadTracker::start()
    {

        m_tracker->FrameReset();
        m_failed = true;

    }

    bool HeadTracker::facesInImage(IplImage *image)
    {

        cvClearMemStorage(m_storage);

        if(m_cascadeFrontal)
        {

            // There can be more than one face in an image. So create a growable sequence of faces.
            // Detect the objects and store them in the sequence
            CvSeq* frontalFaces = cvHaarDetectObjects(image, m_cascadeFrontal, m_storage,
                1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT,
                cvSize(40, 40) );

            if (frontalFaces->total > 0)
                return true;

        }

        if(m_cascadeProfile)
        {

            // There can be more than one face in an image. So create a growable sequence of faces.
            // Detect the objects and store them in the sequence
            CvSeq* profileFaces = cvHaarDetectObjects(image, m_cascadeProfile, m_storage,
                1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT,
                cvSize(40, 40) );

            if (profileFaces->total > 0)
                return true;

        }

        return false;

    }

    void HeadTracker::imagePush(IplImage *image)
    {

        m_image = image;

        frame = image;

        if(m_scale == 1)
            im = m_image; 
        else 
            cv::resize(frame, im, cv::Size((int) (m_scale*frame.cols), (int) (m_scale*frame.rows)));

        // Mirror
        cv::flip(im, im, 1); 
        cv::cvtColor(im, gray, CV_BGR2GRAY);

        //track this image
        std::vector<int> wSize; 

        if(m_failed)
            wSize = m_wSize2; 
        else 
            wSize = m_wSize1; 

        if (m_tracker->Track(gray, wSize, m_fpd, m_nIter, m_clamp, m_fTol, m_fcheck) == 0)
        {

            m_failed = false;

        }
        else
        {

            m_tracker->FrameReset(); 
            m_failed = true;

        }

    }

    bool HeadTracker::hasFailed()
    {
        return m_failed;
    }

    IplImage *HeadTracker::getNextFrame()
    {

        return cvQueryFrame(m_camera); 

    }

    cv::Mat HeadTracker::getShape()
    {

        return m_tracker->_shape;

    }

    cv::Mat HeadTracker::getCon()
    {

        return m_con;

    }

    cv::Mat HeadTracker::getTri()
    {

        return m_tri;

    }

    headPoseData HeadTracker::getHeadPose()
    {

        cv::Mat pose;

        pose = m_tracker->_clm._pglobl;

        double sc = pose.at<double>(0,0);

        m_head_pose.head_pos.x = 0.0f;
        m_head_pose.head_pos.y = 0.0f;
        m_head_pose.head_pos.z = 0.0f;

        m_head_pose.head_rot.x_rads = 0.0f;
        m_head_pose.head_rot.y_rads = 0.0f;
        m_head_pose.head_rot.z_rads = 0.0f;

        if (sc > 0.0)
        {

            int n = (int) (m_tracker->_shape.rows*0.5); 

            // 27 center point between eyes
            double tx = m_tracker->_shape.at<double>(27,0);
            double ty = m_tracker->_shape.at<double>(27+n,0);

            double ox = m_image->width * 0.5;
            double oy = m_image->height * 0.5;

            double px = +(tx - ox);
            double py = -(ty - oy);

            m_head_pose.head_pos.x = (float) (px / (m_image->width * sc));
            m_head_pose.head_pos.y = (float) (py / (m_image->width * sc));
            m_head_pose.head_pos.z = (float) (1.2 / sc);

            double rotx = pose.at<double>(1,0);
            double roty = pose.at<double>(2,0);
            double rotz = pose.at<double>(3,0);

            m_head_pose.head_rot.x_rads = (float) -rotx;
            m_head_pose.head_rot.y_rads = (float) +roty;
            m_head_pose.head_rot.z_rads = (float) -rotz;

        }
        else
        {
            m_head_pose.confidence = 0.0;
        }

        return m_head_pose;

    }

    void HeadTracker::getWorldCoordMatrices(float fovy, float aspect, float zNear, float zFar, float *modelView, float *projection)
    {

        fovy *= 3.14159265f / 180.0f;

        //http://www.opengl.org/sdk/docs/man/xhtml/gluPerspective.xml

        float f = (float) (1.0 / tan(fovy * 0.5f));

        projection[0]  = -f / aspect;
        projection[1]  = 0.0f;
        projection[2]  = 0.0f;
        projection[3]  = 0.0f;
        projection[4]  = 0.0f;
        projection[5]  = f;
        projection[6]  = 0.0f;
        projection[7]  = 0.0f;
        projection[8]  = 0.0f;
        projection[9]  = 0.0f;
        projection[10] = (zFar + zNear) / (zNear + zFar);
        projection[11] = 1.0f;
        projection[12] = 0.0f;
        projection[13] = 0.0f;
        projection[14] = -(2.0f * zFar * zNear) / (zNear + zFar);
        projection[15] = 0.0f;

        modelView[0]  = 1.0f;
        modelView[1]  = 0.0f;
        modelView[2]  = 0.0f;
        modelView[3]  = 0.0f;
        modelView[4]  = 0.0f;
        modelView[5]  = 1.0f;
        modelView[6]  = 0.0f;
        modelView[7]  = 0.0f;
        modelView[8]  = 0.0f;
        modelView[9]  = 0.0f;
        modelView[10] = 1.0f;
        modelView[11] = 0.0f;
        modelView[12] = 0.0f;
        modelView[13] = 0.0f;
        modelView[14] = 0.0f;
        modelView[15] = 1.0f;

    }

    void HeadTracker::getFaceToWorldTransformMatrix(float *f2w_matrix)
    {

        //http://www.songho.ca/opengl/gl_anglestoaxes.html

        float A = m_head_pose.head_rot.x_rads;
        float B = m_head_pose.head_rot.y_rads;
        float C = m_head_pose.head_rot.z_rads;

        f2w_matrix[0] = cos(B)*cos(C);
        f2w_matrix[1] = sin(A)*sin(B)*cos(C)+cos(A)*sin(C);
        f2w_matrix[2] = -cos(A)*sin(B)*cos(C)+sin(A)*sin(C);
        f2w_matrix[3] = 0.0f;

        f2w_matrix[4] = -cos(B)*sin(C);
        f2w_matrix[5] = -sin(A)*sin(B)*sin(C)+cos(A)*cos(C);
        f2w_matrix[6] = cos(A)*sin(B)*sin(C)+sin(A)*cos(C);
        f2w_matrix[7] = 0.0f;

        f2w_matrix[8] = sin(B);
        f2w_matrix[9] = -sin(A)*cos(B);
        f2w_matrix[10] = cos(A)*cos(B);
        f2w_matrix[11] = 0.0f;

        float tx = m_head_pose.head_pos.x;
        float ty = m_head_pose.head_pos.y;
        float tz = m_head_pose.head_pos.z;

        f2w_matrix[12] = tx;
        f2w_matrix[13] = ty;
        f2w_matrix[14] = tz;
        f2w_matrix[15] = 1.0f;

    }

}
