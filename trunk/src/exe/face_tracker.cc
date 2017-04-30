
#include <iostream>

#include <GL/freeglut.h>
#include <GL/gl.h>

#include "HeadTracker.h"

IplImage* iplimage;
GLuint textId;

float projection[16];
float modelView[16];
float objectToWorld[16];

bool failed;

void DrawFPS(cv::Mat &im, int64 *t0, double *fps, unsigned int fnum)
{

    std::string text;
    char sss[256];

    //draw framerate on display image 
    if(fnum % 10 == 0)
    {
        int64 t1 = cvGetTickCount();
        *fps = 10.0/((double(t1 - *t0)/cvGetTickFrequency())/1e+6); 
        *t0 = t1; 
    }

    sprintf(sss,"%d frames/sec", (int)floor(*fps)); 
    text = sss;
    cv::putText(im, text, cv::Point(10,20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(140,0,0));

}

void DrawMesh(cv::Mat &im, cv::Mat &shape, cv::Mat &con,cv::Mat &tri,bool show_numbers = false)
{

    int n = (int) (shape.rows*0.5); 

    cv::Point p1,p2; 
    cv::Scalar c;

    //draw connections
    c = CV_RGB(255,255,0);
    for(int i = 0; i < con.cols; i++)
    {

        p1 = cv::Point((int) shape.at<double>(con.at<int>(0,i),0), (int) shape.at<double>(con.at<int>(0,i)+n,0));
        p2 = cv::Point((int) shape.at<double>(con.at<int>(1,i),0), (int) shape.at<double>(con.at<int>(1,i)+n,0));
        cv::line(im,p1,p2,c,1);

    }

    return;
}

void SetImage()
{

    if(!iplimage)
        return;

    // Upload image data to texture
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glBindTexture(GL_TEXTURE_2D, textId);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, iplimage->width, iplimage->height,0, GL_BGR_EXT, GL_UNSIGNED_BYTE, iplimage->imageData);

}

void DrawImage()
{

    float ww = (float) glutGet(GLUT_WINDOW_WIDTH);
    float wh = (float) glutGet(GLUT_WINDOW_HEIGHT);

    float iw = (float) iplimage->width;
    float ih = (float) iplimage->height;

    float image_aspect = iw / ih;
    float window_aspect = ww / wh;

    int vp_width, vp_height;

    if (image_aspect < window_aspect) 
    {
        vp_height = (int) wh;
        vp_width = (int) (vp_height * image_aspect);
    } else 
    {
        vp_width = (int) ww;
        vp_height = (int) (vp_width / image_aspect);
    }

    glViewport(0, 0, vp_width, vp_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(+0.5f, -0.5f, +0.5f, -0.5f, 0.1, 10);

    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -0.5f);

    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // select texture
    glBindTexture(GL_TEXTURE_2D, textId);

    // do drawing
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(-0.5f, -0.5f, 0.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex3f(-0.5f, 0.5f, 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3f(0.5f, 0.5f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex3f(0.5f, -0.5f, 0.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

}

void DrawAxis()
{

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);

    // x
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.03f, 0.0f, 0.0f);
    glEnd();

    // y
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.03f, 0.0f);
    glEnd();

    // z
    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.03f);
    glEnd();

}

void Draw()
{

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.0,0.0,0.0,1.0);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    SetImage();

    DrawImage();

    glClear(GL_DEPTH_BUFFER_BIT);

    if (!failed)
    {

        // Set perspective viewing frustum
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(&projection[0]);

        // Set modelview matrix in order to set scene
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(&modelView[0]);

        // Draw axis at origin of face-coordinate system
        glMultMatrixf(&objectToWorld[0]);

        DrawAxis();

    }

    glFlush();

    glutSwapBuffers();

}

int main(int argc, char** argv)
{

    faceapi::HeadTracker *faceTracker = new faceapi::HeadTracker();

    std::string windowName("Face Tracker");

    int64 t0 = cvGetTickCount(); 
    double fps = 0;

    glutInit(&argc, argv); //we cheat it ;P

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(320,240);
    glutCreateWindow("OpenGL");

    glutDisplayFunc(Draw);

    glGenTextures(1, &textId);

    bool facesInImage = true;

    int fnum = 0;

    faceTracker->setTolerance(0.01);
    faceTracker->setNumberIterations(20);

    while(1)
    {

        iplimage = faceTracker->getNextFrame();

        float aspect = (float) iplimage->width / (float) iplimage->height;

        fnum++;

        if (fnum % 60 == 0)
        {
            faceTracker->start();
            std::cout << "--> Restart" << std::endl;

            //facesInImage = faceTracker->facesInImage(iplimage);
        }

        if (facesInImage) 
        {
            faceTracker->imagePush(iplimage);

            failed = faceTracker->hasFailed();
        }
        else
        {
            failed = true;
        }

        cv::Mat im = iplimage;

        DrawFPS(im, &t0, &fps, fnum);

        if (!failed)
        {

            DrawMesh(im, faceTracker->getShape(), faceTracker->getCon(), faceTracker->getTri(), false); 

            faceapi::headPoseData pose = faceTracker->getHeadPose();

            /*
            std::cout << "FrameNum: " << fnum
            << " Head Pose:"
            << " head_pos(" << pose.head_pos.x << ", " << pose.head_pos.y << ", " << pose.head_pos.z << ")"
            << " head_rot(" << pose.head_rot.x_rads * 180.0 / 3.14 << ", " << pose.head_rot.y_rads  * 180.0 / 3.14 << ", " << pose.head_rot.z_rads * 180.0 / 3.14 << ")" 
            << std::endl;
            */

            faceTracker->getWorldCoordMatrices(34.0f, aspect, 0.1f, 10.0f, &modelView[0], &projection[0]);
            faceTracker->getFaceToWorldTransformMatrix(&objectToWorld[0]);

        }

        glutPostRedisplay();
        glutMainLoopEvent();

        imshow(windowName.c_str(), im); 

        int c = cvWaitKey(10);

        if (c == 27)
            break; 
        else if (char(c) == 'd')
        {
            faceTracker->start();
            std::cout << "--> Restart" << std::endl;
        }

    }

    delete faceTracker;

}

