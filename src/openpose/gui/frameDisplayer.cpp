#include <opencv2/highgui/highgui.hpp> // imshow, waitKey, namedWindow, setWindowProperty
#include <openpose/utilities/errorAndLog.hpp>
#include <openpose/gui/frameDisplayer.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdlib.h>
#define PI 3.14159265258979

#define M 0
#define T 1
#define Z 2
#define F 3
#define N 4
#define O 5
#define A 6
#define B 7
#define D 8
#define S 9

using namespace std;
using namespace cv;

namespace op
{
	FrameDisplayer::FrameDisplayer(const Point<int>& windowedSize, const string& windowedName, const bool fullScreen) :
		mWindowedSize{ windowedSize },
		mWindowName{ windowedName },
		mGuiDisplayMode{ (fullScreen ? GuiDisplayMode::FullScreen : GuiDisplayMode::Windowed) }
	{
	}

	void FrameDisplayer::initializationOnThread()
	{
		try
		{
			setGuiDisplayMode(mGuiDisplayMode);

			const Mat blackFrame{ mWindowedSize.y, mWindowedSize.x, CV_32FC3, {0,0,0} };
			FrameDisplayer::displayFrame(blackFrame);
			waitKey(1); // This one will show most probably a white image (I guess the program does not have time to render in 1 msec)
			// waitKey(1000); // This one will show the desired black image
		}
		catch (const exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	void FrameDisplayer::setGuiDisplayMode(const GuiDisplayMode displayMode)
	{
		try
		{
			mGuiDisplayMode = displayMode;

			// Setting output resolution
			namedWindow(mWindowName, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
			if (mGuiDisplayMode == GuiDisplayMode::FullScreen)
				setWindowProperty(mWindowName, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
			else if (mGuiDisplayMode == GuiDisplayMode::Windowed)
			{
				resizeWindow(mWindowName, mWindowedSize.x, mWindowedSize.y);
				setWindowProperty(mWindowName, CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
			}
			else
				error("Unknown GuiDisplayMode", __LINE__, __FUNCTION__, __FILE__);
		}
		catch (const exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	void FrameDisplayer::switchGuiDisplayMode()
	{
		try
		{
			if (mGuiDisplayMode == GuiDisplayMode::FullScreen)
				setGuiDisplayMode(GuiDisplayMode::Windowed);
			else if (mGuiDisplayMode == GuiDisplayMode::Windowed)
				setGuiDisplayMode(GuiDisplayMode::FullScreen);
			else
				error("Unknown GuiDisplayMode", __LINE__, __FUNCTION__, __FILE__);
		}
		catch (const exception& e)
		{
			error(e.what(), __LINE__, __FUNCTION__, __FILE__);
		}
	}

	/*�摜��\��t����*/
	void paste(Mat dst, Mat src, int x, int y, int width, int height) {
		Mat resized_img;
		resize(src, resized_img, Size(width, height));

		if (x >= dst.cols || y >= dst.rows) return;
		int w = (x >= 0) ? min(dst.cols - x, resized_img.cols) : min(max(resized_img.cols + x, 0), dst.cols);
		int h = (y >= 0) ? min(dst.rows - y, resized_img.rows) : min(max(resized_img.rows + y, 0), dst.rows);
		int u = (x >= 0) ? 0 : min(-x, resized_img.cols - 1);
		int v = (y >= 0) ? 0 : min(-y, resized_img.rows - 1);
		int px = max(x, 0);
		int py = max(y, 0);

		Mat roi_dst = dst(Rect(px, py, w, h));
		Mat roi_resized = resized_img(Rect(u, v, w, h));
		roi_resized.copyTo(roi_dst);
	}


	void FrameDisplayer::displayFrame(const Mat& frame, const int waitKeyValue)
	{
		int i = 0;
		int mask_type = D;
		int m_scale = 8;	//�傫���قǃ��U�C�N���e���Ȃ�
		Rect rect;		//��̈��؂�o��
		Mat mask_img;	//�}�X�N�摜
		int mask_width = head_bottom[i].y - head_top[i].y;//(LShoulder[i].x - RShoulder[i].x) / 2 + 50;
		int mask_height = head_bottom[i].y - head_top[i].y;

		Mat head, body, Rleg, Lleg, Rarm, Larm, tmp1, tmp2, mask_head, mask_body, mask_Rarm, mask_Larm, mask_Rleg, mask_Lleg, black;
		int body_width = 0, body_height = 0;
		int Rleg_width = 0, Rleg_height = 0;
		int Lleg_width = 0, Lleg_height = 0;
		int RArm_width = 0, RArm_height = 0, RArm_distance = 0;
		int LArm_width = 0, LArm_height = 0;

		//try
		//{
		for (i = 0; i <= n; i++)
		{
			//	if (mask_type == B) {
			//		//�̂̍��W���畝�ƍ������v�Z
			//		mask_width = abs((LShoulder[i].x - RShoulder[i].x) / 2);
			//		mask_height = abs(head_bottom[i].y - head_top[i].y);
			//		body_width = abs(LShoulder[i].x - RShoulder[i].x);
			//		body_height = abs(rWest[i].y - RShoulder[i].y);
			//		Rleg_width = abs(body_width / 2);
			//		Rleg_height = abs(RAnkle[i].y - rWest[i].y);
			//		Lleg_width = abs(body_width / 2);
			//		Lleg_height = abs(LAnkle[i].y - lWest[i].y);
			//		RArm_height = abs(RWrist[i].y - RShoulder[i].y);
			//		RArm_width = Rleg_width / 2;
			//		LArm_height = abs(LWrist[i].y - LShoulder[i].y);
			//		LArm_width = Lleg_width / 2;
			//		//�̂ɏd�˂�p�̉摜�̓ǂݍ���
			//		head = imread("examples/media/head.jpg"); 
			//		body = imread("examples/media/body.jpg"); 
			//		Rleg = imread("examples/media/Rleg.jpg"); 
			//		Lleg = imread("examples/media/Lleg.jpg"); 
			//		Larm = imread("examples/media/Larm.jpg"); 
			//		Rarm = imread("examples/media/Rtest.png"); 
			//		//�}�X�N�p�̉摜�̓ǂݍ���
			//		black = imread("examples/media/test.png");
			//		mask_head = imread("examples/media/mask_head.jpg");
			//		mask_body = imread("examples/media/mask_body.jpg");
			//		mask_Rarm = imread("examples/media/mask_Rarm.jpg");
			//		mask_Larm = imread("examples/media/mask_Larm.jpg");
			//		mask_Rleg = imread("examples/media/mask_Rleg.jpg");
			//		mask_Lleg = imread("examples/media/mask_Lleg.jpg");
			//		//���̏���
			//		if ((head_top[i].x - mask_width / 2) > 0 && head_top[i].y - 10 > 0 && mask_width > 0 && mask_height > 0) {
			//			//�}�X�N����
			//			tmp1 = frame.clone(); //�R�s�[
			//			tmp2 = frame.clone();
			//			paste(tmp1, black, 0, 0, tmp1.cols, tmp1.rows); //���T�C�Y�̐^�����ȉ�ʂ��쐬
			//			paste(tmp2, black, 0, 0, tmp2.cols, tmp2.rows);
			//			paste(tmp1, mask_head, head_top[i].x - mask_width / 2, head_top[i].y - 10, mask_width, mask_height + 50); //�����ɓ��̕���������
			//			paste(tmp2, head, head_top[i].x - mask_width / 2, head_top[i].y - 10, mask_width, mask_height + 50); //�����ɓ��̉摜
			//			bitwise_and(tmp1, tmp2, tmp2); //��񖇂̃}�X�N���Ƃ邱�ƂŔw�i�^�����ȓ��̉摜���쐬(�T�C�Y��frame)
			//			bitwise_not(tmp1, tmp1); //���n�ɓ��̕���������
			//			bitwise_and(tmp1, frame, tmp1); //�w�i�ɓ��̕���������
			//			bitwise_or(tmp1, tmp2, frame); //�񖇂̉摜�𑫂�
			//			//debug
			//			//imshow("tmp1", tmp1);
			//			//imshow("tmp2", tmp2);
			//		}
			//		//�̂̏���
			//		if (body_height > 0 && body_width > 0 && RShoulder[i].x > 0 && RShoulder[i].y - 25 > 0) {
			//			//�}�X�N����
			//			tmp1 = frame.clone();
			//			tmp2 = frame.clone();
			//			paste(tmp1, black, 0, 0, tmp1.cols, tmp1.rows);
			//			paste(tmp2, black, 0, 0, tmp2.cols, tmp2.rows);
			//			paste(tmp1, mask_body, RShoulder[i].x, RShoulder[i].y - 25, body_width, body_height + 25);
			//			paste(tmp2, body, RShoulder[i].x, RShoulder[i].y - 25, body_width, body_height + 25);
			//			bitwise_and(tmp1, tmp2, tmp2);
			//			bitwise_not(tmp1, tmp1);
			//			bitwise_and(tmp1, frame, tmp1);
			//			bitwise_or(tmp1, tmp2, frame);
			//		}
			//		//�E���̏���
			//		if (Rleg_height > 0 && Rleg_width > 0 && rWest[i].x - 25 > 0 && rWest[i].y - 75 > 0) {
			//			tmp1 = frame.clone();
			//			tmp2 = frame.clone();
			//			paste(tmp1, black, 0, 0, tmp1.cols, tmp1.rows);
			//			paste(tmp2, black, 0, 0, tmp2.cols, tmp2.rows);
			//			paste(tmp1, mask_Rleg, rWest[i].x - 25, rWest[i].y - 25, Rleg_width, Rleg_height + 75);
			//			paste(tmp2, Rleg, rWest[i].x - 25, rWest[i].y - 25, Rleg_width, Rleg_height + 75);
			//			bitwise_and(tmp1, tmp2, tmp2);
			//			bitwise_not(tmp1, tmp1);
			//			bitwise_and(tmp1, frame, tmp1);
			//			bitwise_or(tmp1, tmp2, frame);
			//		}
			//		//�����̏���
			//		if (Lleg_height > 0 && Lleg_width > 0 && rWest[i].x - 25 > 0 && rWest[i].y - 75 > 0) {
			//			tmp1 = frame.clone();
			//			tmp2 = frame.clone();
			//			paste(tmp1, black, 0, 0, tmp1.cols, tmp1.rows);
			//			paste(tmp2, black, 0, 0, tmp2.cols, tmp2.rows);
			//			paste(tmp1, mask_Lleg, rWest[i].x - 25 + Lleg_width, rWest[i].y - 25, Lleg_width, Lleg_height + 75);
			//			paste(tmp2, Lleg, rWest[i].x - 25 + Lleg_width, rWest[i].y - 25, Lleg_width, Lleg_height + 75);
			//			bitwise_and(tmp1, tmp2, tmp2);
			//			bitwise_not(tmp1, tmp1);
			//			bitwise_and(tmp1, frame, tmp1);
			//			bitwise_or(tmp1, tmp2, frame);
			//			}
			//		//���r�̏���
			//		if (LShoulder[i].x - 15 > 0 && LShoulder[i].y - 25 > 0) {
			//			tmp1 = frame.clone();
			//			tmp2 = frame.clone();
			//			paste(tmp1, black, 0, 0, tmp1.cols, tmp1.rows);
			//			paste(tmp2, black, 0, 0, tmp2.cols, tmp2.rows);
			//			paste(tmp1, mask_Larm, LShoulder[i].x - 15, LShoulder[i].y - 25, LArm_width + 10, LArm_height + 75);
			//			paste(tmp2, Larm, LShoulder[i].x - 15, LShoulder[i].y - 25, LArm_width + 10, LArm_height + 75);
			//			bitwise_and(tmp1, tmp2, tmp2);
			//			bitwise_not(tmp1, tmp1);
			//			bitwise_and(tmp1, frame, tmp1);
			//			bitwise_or(tmp1, tmp2, frame);
			//		}
			//		//�E�r�̏���
			//		if (RShoulder[i].x - 15 > 0 && RShoulder[i].y - 25 > 0) {
			//			tmp1 = frame.clone();
			//			tmp2 = frame.clone();
			//			paste(tmp1, black, 0, 0, tmp1.cols, tmp1.rows);
			//			paste(tmp2, black, 0, 0, tmp2.cols, tmp2.rows);
			//			paste(tmp1, mask_Rarm, RShoulder[i].x - 25, RShoulder[i].y - 25, RArm_width + 10, RArm_height + 75);
			//			paste(tmp2, Rarm, RShoulder[i].x - 25, RShoulder[i].y - 25, RArm_width + 10, RArm_height + 75);
			//			bitwise_and(tmp1, tmp2, tmp2);
			//			bitwise_not(tmp1, tmp1);
			//			bitwise_and(tmp1, frame, tmp1);
			//			bitwise_or(tmp1, tmp2, frame);
			//		}
					//�������Ԃɍ���Ȃ��H
						////�E�r�̏���
						//RArm_height = abs(RWrist[i].y - RShoulder[i].y)*2;
						//RArm_width = abs(RWrist[i].x - RShoulder[i].x)*2;
						//  //���Ǝ�̓�_�Ԃ̋���
						//RArm_distance = hypot( (RWrist[i].x - RShoulder[i].x), (RWrist[i].y - RShoulder[i].y)*2 );
						//RArm = imread("examples/media/Rarm.jpg"); // �摜�̓ǂݍ���
						//double degree = atan2(RShoulder[i].y - RWrist[i].y, RShoulder[i].x - RWrist[i].x);
						//degree = 265 - degree * (180 / PI);
						//Point2f center(RArm.cols*0.5, 0.0);
						//const Mat affine_matrix = getRotationMatrix2D(center, degree, 1.0);
						//Mat dst_img;
						//warpAffine(RArm, dst_img, affine_matrix, mask_img.size());
						//if ( (RShoulder[i].x - (RArm_width+50)) > 0 && (RShoulder[i].y - (RArm_height+25) ) > 0)
						//	paste(frame, RArm, RShoulder[i].x - (RArm_width/2+50), RShoulder[i].y - (RArm_height+25), RArm_width + 10, RArm_height + 75);

				//}

				//�烂�U�C�N
			if (mask_type == M && head_top[i].x > 50) {
				rect = Rect(head_top[i].x - 70, head_top[i].y, mask_width + 20, mask_height);
				mask_img = Mat(frame, rect).clone();
				//���U�C�N����
				resize(mask_img, mask_img, Size(), (double)1 / m_scale, (double)1 / m_scale);
				resize(mask_img, mask_img, Size(), m_scale, m_scale, INTER_NEAREST);
			}
			//	//�Ǖ����U�C�N
			//	if (mask_type == N) {
			//		rect = Rect(rWest[i].x, rWest[i].y, 200, 200);
			//		mask_img = Mat(frame, rect).clone();
			//		//���U�C�N����
			//		resize(mask_img, mask_img, Size(), (double)1 / m_scale, (double)1 / m_scale);
			//		resize(mask_img, mask_img, Size(), m_scale, m_scale, INTER_NEAREST);
			//	}
			//	//�������
			//	else if (mask_type == T) {
			//		rect = Rect(head_top[i].x + 100, head_top[i].y, 10, 10);
			//		mask_img = Mat(frame, rect).clone();
			//	}
			//	//���m����
			//	else if (mask_type == F) {
			//		mask_img = imread("examples/media/fukushi.png"); // �摜�̓ǂݍ���
			//		if (mask_img.empty())
			//			printf("mask image is empty.\n");
			//	}
			//	//zombie
			//	else if (mask_type == Z) {
			//		mask_img = imread("examples/media/head.jpg"); // �摜�̓ǂݍ���
			//		if (mask_img.empty())
			//			printf("mask image is empty.\n");
			//	}
			//	//�S�̎�
			//	else if (mask_type == O) {
			//		mask_img = imread("examples/media/oninote.jpg"); // �摜�̓ǂݍ���
			//		if (mask_img.empty())
			//			printf("mask image is empty.\n");
			//	}
			//	//�S�g�]���r
			//	else if (mask_type == A) {
			//		mask_img = imread("examples/media/zombie_stand2.png", IMREAD_UNCHANGED); // �摜�̓ǂݍ���
			//		if (mask_img.empty())
			//			printf("mask image is empty.\n");
			//		mask_height = RAnkle[i].y - head_top[i].y;
			//		mask_width = mask_height * 0.3;
			//	}
			else if (mask_type == S) {
				mask_img = imread("examples/media/saiha.png"); // �摜�̓ǂݍ���
				if (mask_img.empty()){
					printf("mask image is empty.\n");
					exit(1);
				}
				if ( RWrist[i].x > 300 && RWrist[i].y > 200 ){
					//�}�X�N����
					tmp1 = frame.clone(); //�R�s�[
					tmp2 = frame.clone();
					paste(tmp1, black, 0, 0, tmp1.cols, tmp1.rows); //���T�C�Y�̐^�����ȉ�ʂ��쐬
					paste(tmp2, black, 0, 0, tmp2.cols, tmp2.rows);
					paste(tmp1, mask_img, RWrist[i].x-300, RWrist[i].y-200, 300, 200); //�����ɓ��̕���������
					paste(tmp2, head, head_top[i].x - mask_width / 2, head_top[i].y - 10, mask_width, mask_height + 50); //�����ɓ��̉摜
					bitwise_and(tmp1, tmp2, tmp2); //��񖇂̃}�X�N���Ƃ邱�ƂŔw�i�^�����ȓ��̉摜���쐬(�T�C�Y��frame)
					bitwise_not(tmp1, tmp1); //���n�ɓ��̕���������
					bitwise_and(tmp1, frame, tmp1); //�w�i�ɓ��̕���������
					bitwise_or(tmp1, tmp2, frame); //�񖇂̉摜�𑫂�
					//debug
					imshow("tmp1", tmp1);
					imshow("tmp2", tmp2);
				}
			}
			//}
			//		
			//	//�}�X�N�摜���d�˂�
			//	if (mask_type == N) {
			//		paste(frame, mask_img, rWest[i].x, rWest[i].y, 200, 200);
			//		//����F���ł��ĂȂ��ƃ��U�C�N���ǐ����Ȃ�
			//	}
			//	else if (mask_type == O) {
			//		double degree = atan2(LElbow[i].y - LWrist[i].y, LElbow[i].x - LWrist[i].x);
			//		degree = 160 - degree * (180 / PI);
			//		Point2f center(mask_img.cols*0.5, mask_img.rows*0.5);
			//		const Mat affine_matrix = getRotationMatrix2D(center, degree, 1.0);
			//		Mat dst_img;
			//		warpAffine(mask_img, dst_img, affine_matrix, mask_img.size());
			//		paste(frame, dst_img, LWrist[i].x + 20, LWrist[i].y - 50, 200, 200);
			//	}
			if (mask_type == S) {
				if ( RWrist[i].x > 300 && RWrist[i].y > 200 )
					paste(frame, mask_img, RWrist[i].x-300, RWrist[i].y-200, 300, 200);
			}
			else if (mask_type != B && mask_type != D) {
				if (mask_height > 0 && head_top[i].x > mask_width / 2 && head_top[i].y > 10)
					paste(frame, mask_img, head_top[i].x - mask_width / 2, head_top[i].y - 10, mask_width, mask_height + 50);
			}
		}

		imshow(mWindowName, frame);
		if (waitKeyValue != -1)
			waitKey(waitKeyValue);
	}
	//}
	//      catch (const exception& e)
	//      {
	//          error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	//      }
	  //}
}