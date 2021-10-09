#include <iostream>  
#include <string> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <opencv2/opencv.hpp> 
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <math.h>
#include <time.h>

using namespace std;
using namespace cv;

// Parameters
// choose the starting frame. 
const int frameToStart = 172;  
// square example, 20^2=400.
// ������ֵ�������ж�����Ŀ������Ƿ���������С�������ֵ�Ļ����ͽ����������ο�ϲ���һ����ľ��ο�
const int Segment_merge = 3000;
// square
// �����ֵ����� blob �����С�������ֵ�Ļ�����ɾ��
const int Min_blob_size = 1200;

long currentFrame = frameToStart;

// input video name
// 1.1_rouen_video.avi
// 2.2_Atrium.avi
// 3.3_Sherbrooke.avi
// 4.4_StMarc.avi
// 5.5_levesque.mov
char filename[100] = "video/rouen_video.avi";
//char filename[100] = "video/atrium_video.avi";
//char filename[100] = "video/sherbrooke_video.avi";
//char filename[100] = "video/stmarc_video.avi";
//char filename[100] = "video/levesque_video.mov";

// Parameters for tracking and counting
// parameters with 'inFrame' means that they are still in the full image.
Mat gray, gray_prev, frame;

// to save objects' bounding box  
// ����������һ֡�е�Ŀ����ο��С������
vector<Rect> boundRect_inFrame;         
// save objects' ID
// �������浱ǰ֡��Ŀ��� id��Ŀ�� id ��ȫ�ֱ��� obj_num ������
vector<int> boundRect_labelinFrame;
// objects not visible for 8 frames will be discarded
// delay_toDeleteinFrame �б��м�¼�ľ��Ǹ��� kcf tracker û��ƥ���֡����
// ����ƥ���֡���ﵽ 8 ʱ������Ϊ��������Ѿ��뿪�˻���
vector<int> delay_toDeleteinFrame;
// Group objects by saving their bounding box of previous frame.
// ����ͬ�� kcf tracker �ڵ�ǰ֡�л����ڵ�ʱ��Ҳ����һ�� COR ��Ӧ�Ŷ�� TO,
// ���� kcf tracker[0], kcf tracker[2] �����ڵ�������һ֡�У�(COR)i ��ռ��������� (TO)0��(TO)2
// ��ˣ�������Ҫ�� group_whenOcclusion[0] = group_whenOcclusion[2] = Boundrect[i]
// �� Boundrect[i] ��ֵ���� boundRect_inFrame[i] �ָ�ֵ���� tracker->update���� update ���������Ŀ���λ��
// group_whenOcclusion �б���ľ��ǵ�ǰ�����ڵ������� kcf tracker �ڵ�ǰ֡���º��λ�� bounding box
vector<Rect2d> group_whenOcclusion;
// objects' substantially overlap for 8 frames will be discarded
// vector<int> KCF_occlusionTime;      
// save ID's will be reused in the future
vector<int> turn_back;              
// save objects' KCF trackers
// ��������ÿһ��Ŀ���Ӧ�� kcf tracker
vector<Ptr<Tracker>> tracker_vector; 
// for labeling object
// ��������Ŀ��� id �ַ�������������Ƶ֡����ʾʱ���õ�
vector<string> showMsg;       
// �ڽ� id ת��Ϊ�ַ���ʱ����Ϊ���������洢 str(id)
string save_label;
// stringstream ��Ҫ����������������ת����������ǽ� int ת��Ϊ string ����
stringstream ss;
// if there is no objects previous.
// ���ǰһ֡��û��һ��Ŀ��Ļ����ͽ� prevNo_obj ����Ϊ 1
int prevNo_obj = 1;                   
// total number of objects
// ��Ƶ֡�е�Ŀ������
int obj_num = 0;                      

// to save bounding boxes' path for output xml
// BoundRect_save ��һ����ά�б���������ÿһ�� kcf tracker ���ٵ����������ֵĹ켣 bounding box
vector<vector<Rect>> BoundRect_save;    
// to save the frame_numbers that the object appears. 
// Rectsave_Frame ��һ����ά�б���������ÿһ�� kcf tracker ���ٵ����������ֵ�֡��
vector<vector<int>> Rectsave_Frame;     
// to save objects' starting frame. within this val, bounding box can rematch between two frames float x,float y
vector<int> obj_appear_frame;           

// �ж� a, b ����֮��ľ����Ƿ�С���趨����ֵ
bool CentroidCloseEnough(Point a, Point b) {
	return (((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)) < Segment_merge);
}

// another strategy for merging segments
bool isOverlapping(Rect rc, Rect rc2) {
	Rect rc1;
	rc1.x = rc.x - 5;
	rc1.y = rc.y - 5;
	rc1.width = rc.width + 5;
	rc1.height = rc.height + 5;
	return (rc1.x + rc1.width > rc2.x) && (rc2.x + rc2.width > rc1.x) && (rc1.y + rc1.height > rc2.y) && (rc2.y + rc2.height > rc1.y);
}

// calculate overlap rate of two blobs with the first blob.
float bbOverlap(const Rect &box1, const Rect &box2)
{
	if (box1.x > box2.x + box2.width) { return 0.0; }
	if (box1.y > box2.y + box2.height) { return 0.0; }
	if (box1.x + box1.width < box2.x) { return 0.0; }
	if (box1.y + box1.height < box2.y) { return 0.0; }
	float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
	float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);
	float intersection = colInt * rowInt;
	float area1 = box1.width*box1.height;
	float area2 = box2.width*box2.height;
	return (intersection / area1);
}

// this mask is only for video Rene
void MyPolygon(Mat& img)
{
	/** Create some points */
	Point rook_points[1][20];
	rook_points[0][0] = Point(628, 95);
	rook_points[0][1] = Point(780, 105);
	rook_points[0][2] = Point(718, 501);
	rook_points[0][3] = Point(1279, 544);
	rook_points[0][4] = Point(1279, 636);
	rook_points[0][5] = Point(727, 600);
	rook_points[0][6] = Point(727, 720);
	rook_points[0][7] = Point(233, 720);
	rook_points[0][8] = Point(329, 560);
	rook_points[0][9] = Point(270, 542);
	rook_points[0][10] = Point(290, 485);
	rook_points[0][11] = Point(387, 468);

	const Point* ppt[1] = { rook_points[0] };
	int npt[] = { 12 };

	fillPoly(img, ppt, npt, 1, Scalar(255, 255, 255), 8);
	polylines(img, ppt, npt, 1, 1, CV_RGB(0, 0, 0), 2, 8, 0);
}

// calculate the centroid/gravity center
// �ο����� http://webdoc.sub.gwdg.de/ebook/serien/ah/UU-CS/2001-57.pdf �� gravity center �ļ��㷽ʽ
// ����Ŀ����� area ����������Ŀ�� A �ĵ�� x ����� y ����ֱ���ӣ��ٳ���Ŀ�� A �е������ܵĸ���:
// (sum(A �� x ����) + sum(A �� y ����)) / card(A)��card(A) ������� A Ŀ����������ظ���
Point aoiGravityCenter(Mat &src, Rect area){

	float sumx = 0, sumy = 0;
	float num_pixel = 0;
	Mat ROI = src(area);

	for (int x = 0; x < ROI.cols; x++) {
		for (int y = 0; y < ROI.rows; y++) {
			int val = ROI.at<uchar>(y, x);

			// val >= 50 ˵���������λ��Ŀ������
			if (val >= 50) {
				sumx += x;
				sumy += y;
				num_pixel++;
			}//if

		}//for
	}//for

	Point p(sumx / num_pixel, sumy / num_pixel);
	p.x += area.x;
	p.y += area.y;
	return p;

}

// stringstream ��Ҫ����������������ת����������ǽ� int ת��Ϊ string ����
// pass label
string pass_label(int obj_num) {
	ss.clear();
	ss << obj_num;
	ss >> save_label;
	return save_label;
}

// create
void Create_new_obj(Rect2d Boundrect) {
	// for objects exist less than 8 frames, their ID's should be turned back.
	if (turn_back.size() != 0) {
		int obj_num = turn_back[0];
		string save_label = pass_label(obj_num);

		turn_back.erase(turn_back.begin() + 0);
		showMsg.insert(showMsg.end(), save_label);
		boundRect_labelinFrame.insert(boundRect_labelinFrame.end(), obj_num);
	}// if
	else {
		obj_num++;
		// create a new object.
		string save_label = pass_label(obj_num);

		showMsg.insert(showMsg.end(), save_label);
		boundRect_labelinFrame.insert(boundRect_labelinFrame.end(), obj_num);
	}// else

	// boundRect_inFrame �б��б���ľ�����һ֡�и��� kcf tracker ���º�ľ��ο�
	boundRect_inFrame.insert(boundRect_inFrame.end(), Boundrect);
	// delay_toDeleteinFrame �б��м�¼���� kcf tracker ������ʧ���֡����
	// ��ʧ���֡���ﵽ 8 ʱ������Ϊ��� kcf tracker ���ٵ�Ŀ���Ѿ��ӻ������뿪��
	// �Ӷ��Ὣ��� kcf tracker �� active tracks ��ɾ����
	delay_toDeleteinFrame.insert(delay_toDeleteinFrame.end(), 0);

	Rect temp;
	temp.x = temp.y = temp.height = temp.width = 0;
	group_whenOcclusion.insert(group_whenOcclusion.end(), temp);

	TrackerKCF::Params param;
	// KCF ��������ʹ�õ�����������ʹ�� CN + GRAY
	param.desc_pca = TrackerKCF::MODE::CN | TrackerKCF::MODE::GRAY;
	Ptr<TrackerKCF> tracker = TrackerKCF::create(param);

	// ��ʼ�� KCF �����������ҽ����������浽 tracker_vector �б���
	tracker->init(frame, Boundrect);
	tracker_vector.insert(tracker_vector.end(), tracker);

	// rect_save ��һ���б������������ kcf tracker ���ٵ�������ֵĹ켣 bounding box
	vector<Rect> rect_save;
	rect_save.insert(rect_save.end(), Boundrect);
	BoundRect_save.insert(BoundRect_save.end(), rect_save);

	// rectsave_obj ��һ���б������������ kcf tracker ���ٵ�������ֵ�֡��
	vector<int> rectsave_obj;
	rectsave_obj.insert(rectsave_obj.end(), currentFrame);
	Rectsave_Frame.insert(Rectsave_Frame.end(), rectsave_obj);
}

 // delete an object
void delete_obj(vector<int> &Find_Tracker, vector<int> &add_this_frame, int i) {
	boundRect_inFrame.erase(boundRect_inFrame.begin() + i);
	boundRect_labelinFrame.erase(boundRect_labelinFrame.begin() + i);
	delay_toDeleteinFrame.erase(delay_toDeleteinFrame.begin() + i);
	group_whenOcclusion.erase(group_whenOcclusion.begin() + i);
	Find_Tracker.erase(Find_Tracker.begin() + i);

	tracker_vector[i].release();
	tracker_vector.erase(tracker_vector.begin() + i);

	showMsg.erase(showMsg.begin() + i);
	BoundRect_save.erase(BoundRect_save.begin() + i);
	Rectsave_Frame.erase(Rectsave_Frame.begin() + i);
	add_this_frame.erase(add_this_frame.begin() + i);
}//delete

void deliver_tracker(int index, Rect2d &Boundrect) {
	tracker_vector[index].release();
	tracker_vector.erase(tracker_vector.begin() + index);
	BoundRect_save[index].erase(BoundRect_save[index].end() - 1);
	Rectsave_Frame[index].erase(Rectsave_Frame[index].end() - 1);

	TrackerKCF::Params param;
	param.desc_pca = TrackerKCF::MODE::CN | TrackerKCF::MODE::GRAY;
	Ptr<TrackerKCF> tracker = TrackerKCF::create(param);

	tracker_vector.insert(tracker_vector.begin() + index, tracker);
	tracker_vector[index]->init(frame, Boundrect);
	boundRect_inFrame[index] = Boundrect;
	tracker_vector[index]->update(frame, boundRect_inFrame[index]);

	// save frame
	Rectsave_Frame[index].insert(Rectsave_Frame[index].end(), currentFrame);
	// save rect
	BoundRect_save[index].insert(BoundRect_save[index].end(), boundRect_inFrame[index]);
	rectangle(frame, boundRect_inFrame[index], Scalar(255, 0, 0), 2, 1);
}

//save properties to xml. 
void SaveToXML(int i) {
	ofstream outdata;
	outdata.open("bool.txt", ios::app);//ios::app��β��׷�ӵ���˼
	outdata << "	<Trajectory obj_id=" << char(34) << boundRect_labelinFrame[i] << char(34) << " obj_type=" << char(34) << "Human" << char(34) << " start_frame = " << char(34) << Rectsave_Frame[i][0] << char(34) << " end_frame = " << char(34) << Rectsave_Frame[i][0] + BoundRect_save[i].size() - 1 << char(34) << ">" << endl;
	for (int j = 0; j < BoundRect_save[i].size(); j++) {
		if (j >= 1) {
			if (Rectsave_Frame[i][j] - Rectsave_Frame[i][j - 1] > 1) {

				int interval = Rectsave_Frame[i][j] - Rectsave_Frame[i][j - 1];
				int d_value_x, d_value_y, d_value_height, d_value_width;
				d_value_x = (BoundRect_save[i][j].x - BoundRect_save[i][j - 1].x) / interval;
				d_value_y = (BoundRect_save[i][j].y - BoundRect_save[i][j - 1].y) / interval;
				d_value_width = (BoundRect_save[i][j].width - BoundRect_save[i][j - 1].width) / interval;
				d_value_height = (BoundRect_save[i][j].height - BoundRect_save[i][j - 1].height) / interval;

				for (int k = 1; k < interval; k++) {
					outdata << "		<Frame frame_no=" << char(34) << Rectsave_Frame[i][j - 1] + k
						<< char(34) << " x=" << char(34) << BoundRect_save[i][j - 1].x + d_value_x * k
						<< char(34) << " y=" << char(34) << BoundRect_save[i][j - 1].y + d_value_y * k
						<< char(34) << " width=" << char(34) << BoundRect_save[i][j - 1].width + d_value_width * k
						<< char(34) << " height=" << char(34) << BoundRect_save[i][j - 1].height + d_value_height * k
						<< char(34) << " observation=" << char(34) << 0 << char(34) << " annotation=" << char(34)
						<< 0 << char(34) << " contour_pt=" << char(34) << 0 << char(34) << "></Frame>" << endl;
				}//for
			}
		}// j>1
		outdata << "		<Frame frame_no=" << char(34) << Rectsave_Frame[i][j] << char(34) << " x=" << char(34)
			<< BoundRect_save[i][j].x << char(34) << " y=" << char(34) << BoundRect_save[i][j].y << char(34)
			<< " width=" << char(34) << BoundRect_save[i][j].width << char(34) << " height=" << char(34)
			<< BoundRect_save[i][j].height << char(34) << " observation=" << char(34) << 0 << char(34)
			<< " annotation=" << char(34) << 0 << char(34) << " contour_pt=" << char(34) << 0 << char(34)
			<< "></Frame>" << endl;
	}// for
	outdata << "	</Trajectory>" << endl;
	outdata.close();
}


void KCF_tracker(Mat &frame, vector<Rect2d> Boundrect, vector<Point2f> Centroids) {

	// initial -> no object in previous frame.
	// create properties for each object appears in this frame
	// prevNo_obj Ϊ 1 ��ʾǰһ֡��û������
	if (prevNo_obj == 1) {
		for (int i = 0; i < Boundrect.size(); i++){
			// ǰһ֡����û�����壬���Ϊ��һ֡�³��ֵ�Ŀ�괴����Ӧ�ĸ����� kcf tracker
			// ���Ҵ�������һά�б��ֱ��������汻���ٵ�����켣������ĳ��ֵ�֡��
			Create_new_obj(Boundrect[i]);
		}
		prevNo_obj = 0;
	}// if
	else {
		// no object now, come back to initial status
		// ��� tracker_vector �� size Ϊ 0������ǰһ֡��û�� tracker
		if (tracker_vector.size() == 0) {
			prevNo_obj = 1;
			return;
		}

		// identify whether KCF tracker match in this frame or not
		vector<int> Find_Tracker;
		// �� Find_Tracker ���������� boundRect_inFrame.size() �� -1���������ж� Find_Tracker[i] �Ƿ�Ϊ -1��
		// ���Ϊ -1 �Ļ�����ô��˵�� kcf tracker �ڵ�ǰ֡��û���κ� COR ����ƥ��
		Find_Tracker.insert(Find_Tracker.end(), boundRect_inFrame.size(), -1);

		// identify whether it's property saved to xml or not
		vector<int> add_this_frame(boundRect_inFrame.size(), 0);

		// calculate how many existing objects in Bounding Boxes of current frame. 
		// KCF_Num_Blob ��ʼ��Ϊһ������Ϊ Boundrect.size() ����ȫ��Ϊ 0 ������
		// KCF_Num_Blob[i] = j, ������ǰ֡�е� i ��Ŀ��� (COR)i �� j �� kcf tracker ��ƥ��
		vector<int> KCF_Num_Blob(Boundrect.size(), 0);

		// save match property for each existing objects.
		// KCF_match ��ʼ��Ϊһ������Ϊ boundRect_inFrame.size() ����ȫ��Ϊ 0 ����������ʾ kcf tracker[i] �͵�ǰ֡����һ�� BGS ��ƥ��
		// KCF_match[i] = j, ������ i �� kcf tracker �͵�ǰ֡�е� BGS �ĵ� j ��Ŀ����ƥ��
		// KCF_match[i] = -1, ������ i �� kcf tracker �͵�ǰ֡�е��κ� BGS Ŀ�궼��ƥ��
		vector<int> KCF_match(boundRect_inFrame.size(), 0);

		// match objects by comparing overlapping rates of objects saved in previous frame
		// ͨ������ǰһ֡�� kcf tracker �� bbx �͵�ǰ֡�� BGS �� bbx �� iou ֵ��������ƥ��
		for (int i = 0; i < boundRect_inFrame.size(); i++) {
			float max = 0;
			// label �ͱ�ʾ��ǰ֡�к� kcf tracker[i] ��ƥ�䣨iou ֵ��ߣ��� bounding box Ϊ Boundrect[label] 
			int label = 0;

			for (int j = 0; j < Boundrect.size(); j++) {
				// find the most suitable one.
				float overlap = bbOverlap(boundRect_inFrame[i], Boundrect[j]);
				if (overlap > max) {
					max = overlap;
					label = j;
				}// if
			}// for

			// object matches.
			if (max > 0) {
				KCF_Num_Blob[label] += 1;
				KCF_match[i] = label;
			}// if
			// object missing
			else {
				KCF_match[i] = -1;
			}
		}// for

		/**********************  Occlusion occurs   ************************/
		for (int i = 0; i < Boundrect.size(); i++) {

			// occlusion occurs
			// ������廥���ڵ���״̬��Occluded��
			// KCF_Num_Blob[i] >= 2 �ͱ���һ�� COR ��Ӧ��� TO��Ҳ���Ƕ�������ڵ�ǰ֡�з����˻����ڵ�
			if (KCF_Num_Blob[i] >= 2) {
				int label = -1;
				float max = 0;
				// within_label �д洢���Ǻ͵�ǰ֡�е� i ��Ŀ��� COR ��ƥ��Ķ�� kcf tracker ��Ŀ��� bounding box 
				vector<int> within_label;

				for (int j = 0; j < boundRect_inFrame.size(); j++) {
					if (KCF_match[j] == i) {
						within_label.insert(within_label.end(), j);
					}// if
				}// for

				for (int j = 0; j < within_label.size(); j++) {
					// within_label[j] ��� kcf tracker �ڵ�ǰ֡���ҵ��˵� i ��ƥ��
					// ��˽����� Find_Tracker ����Ϊ i��ͬʱ������ delay ����Ϊ 0
					Find_Tracker[within_label[j]] = i;
					delay_toDeleteinFrame[within_label[j]] = 0;

					float save = bbOverlap(Boundrect[i], boundRect_inFrame[within_label[j]]);
					// find the best matching object.
					if (save > max) {
						max = save;
						label = within_label[j];
					}// if

					// when objects occlude, we group them by saving this unidentified bounding box.
					if (group_whenOcclusion[within_label[j]].width * group_whenOcclusion[within_label[j]].height == 0) {
						group_whenOcclusion[within_label[j]] = Boundrect[i];
					} else {
						int area = group_whenOcclusion[within_label[j]].width * group_whenOcclusion[within_label[j]].height;
						int new_area = Boundrect[i].width * Boundrect[i].height;

						// update objects' group rectangle
						// ��� new_area ����� area �ı仯�����ر����ô��ʹ�� new_area �����¾ɵ� area
						if ((new_area > area * 0.8) && (new_area < area * 2.2)) {
							group_whenOcclusion[within_label[j]] = Boundrect[i];
						}// if
					}// else

					// update ������
					// frame: the current frame
					// boundingBox: The bounding box that represent the new target location, if true was returned, not modified otherwise
					// ��� kcf tracker �������˵Ļ����͵��� update ������update ������� bounding box �����������и��£����� bounding box Ϊ�������ͣ�������ΪĿ�����λ��
					tracker_vector[within_label[j]]->update(frame, boundRect_inFrame[within_label[j]]);

					// �ж� within_label[j] ��� kcf tracker ����Ϣ�������ֵ�֡���Լ�·�����Ƿ񱻱�����
					if (add_this_frame[within_label[j]] == 0) {
						add_this_frame[within_label[j]] = 1;
						// save frame
						// ������� kcf tracker ���ٵ�������ֵ�֡���� Rectsave_Frame ��
						Rectsave_Frame[within_label[j]].insert(Rectsave_Frame[within_label[j]].end(), currentFrame);
						// save rect
						// ������� kcf tracker ���ٵ������λ�úʹ�С�� BoundRect_save ��
						BoundRect_save[within_label[j]].insert(BoundRect_save[within_label[j]].end(), boundRect_inFrame[within_label[j]]);
						rectangle(frame, boundRect_inFrame[within_label[j]], Scalar(255, 0, 0), 2, 1);
					}// if
				}// for

			}// occlusion occurs

			/************************ Object tracking alone  *****************************/
			// �����ĸ���״̬��Tracked��
			// 1. ��� BGS/(CORi)t ���Ǻ�׼ȷ��, Ҳ����˵, ��������Ƭ��������segmentation��,��ʱ��ʹ�� kcf tracker ���µõ���λ����Ϣ�����и��� 
			// 2. ��� BGS/(CORi)t ��׼ȷ�ģ����ʱ���ֱ��ʹ�� (CORi)t �����и��£�ʹ�� (CORi)t �ĺô��ǿ����� kcf tracker �Գ߶ȴ�С��������Ӧ��������Ծ���
			// ʹ�� (CORi)t �����³�ʼ��һ���µ� kcf tracker������ɾ�����ɵ� kcf tracker
			else if (KCF_Num_Blob[i] == 1) {
				
				int label = -1;
				// �� label �� kcf tracker �͵�ǰ֡�е� i ��Ŀ����ƥ��
				for (int j = 0; j < boundRect_inFrame.size(); j++) {
					if (KCF_match[j] == i) {
						label = j;
					}
				}// for

				// Find_Tracker[label] ����Ϊ i��˵���� label �� kcf tracker �ҵ���ƥ��ĵ� i �� (CORi)
				Find_Tracker[label] = i;
				delay_toDeleteinFrame[label] = 0;

				// area_new = A(CORi)t
				float area_new = Boundrect[i].width * Boundrect[i].height;
				// area_previous = A(TOi)t
				float area_previous = boundRect_inFrame[label].width * boundRect_inFrame[label].height;

				// ��� Tol <= (A(TOi)t / A(CORi)t) <= Toh �Ļ���˵�� BGS ��Ϊ��Ƭ����segmentation��������Ϊ (TOi)t �����ţ�
				// ���򣬾���Ϊ (CORi)t �����ţ���ʹ�� BGS ��ȡ����Ŀ��λ�úʹ�С
				if ((area_previous >= 1.4 * area_new) && (area_previous <= 1.8 * area_new)) {
					// ����� boundRect_inFrame[within_label[i]] ��ʾΪ kcf tracker �ڵ�ǰ֡��Ŀ��λ�� (TOi)t
					// ʹ�� (TOi)t ���� kcf tracker ���и���, ����ʹ�� (TOi)t ���� kcf tracker ���и��²����� kcf tracker ӵ�г߶�����Ӧ�Ĺ���
					// ����� boundRect_inFrame[label] ���ݵ� update ����֮��ᱻ����ΪĿ�����λ��
					tracker_vector[label]->update(frame, boundRect_inFrame[label]);
				}// if
				else {
					/// Background substraction is more precise
					tracker_vector[label].release();
					tracker_vector.erase(tracker_vector.begin() + label);

					// ����һ���µ� kcf tracker
					// ʹ�� BGS Ҳ���Ǳ���ģ����ȡ����Ŀ��λ�úʹ�С�Ļ�������ʹ�� kcf tracker �ĳ߶Ƚ�������Ӧ
					// �������������ɾ��֮ǰ�ɵ� kcf tracker��Ȼ�����³�ʼ��һ���µ� kcf tracker
					TrackerKCF::Params param;
					param.desc_pca = TrackerKCF::MODE::CN | TrackerKCF::MODE::GRAY;
					Ptr<TrackerKCF> tracker = TrackerKCF::create(param);
					tracker_vector.insert(tracker_vector.begin() + label, tracker);

					/// BGS and KCF are nearly the same.
					// ʹ�õ�ǰ֡�е� BGS ����ʼ�� kcf tracker�����Ҷ� kcf tracker ���и���
					tracker_vector[label]->init(frame, Boundrect[i]);
					boundRect_inFrame[label] = Boundrect[i];
					// tracker->update �ڸ��µĹ����У�ͬʱҲ����´���� boundRect_inFrame ʹ���������ڵ�ǰ֡�е���λ��
					// �������е��� kcf tracker �������� (TOj)t
					tracker_vector[label]->update(frame, boundRect_inFrame[label]);
				}// else

				/// save in 'xml' file
				if (add_this_frame[label] == 0) {
					add_this_frame[label] = 1;
					// save frame
					// ���� kcf tracker ���ٵ�Ŀ����ֵ�֡��
					Rectsave_Frame[label].insert(Rectsave_Frame[label].end(), currentFrame);
					// save rect
					// ���� kcf tracker ���ٵ�Ŀ��Ĺ켣
					BoundRect_save[label].insert(BoundRect_save[label].end(), boundRect_inFrame[label]);
					rectangle(frame, Boundrect[i], Scalar(255, 0, 0), 2, 1);
				}//if

			}// end if  object tracking alone
			else {   
			/************** With no trackers match *****************/
			/********** firstly, we try to figure out whether two KCF trackers are tracking the same object or not. ***********/
			// �� Boundrect[i] û���κ� kcf tracker ��������ƥ��ʱ���� KCF_Num_Blob[i] = 0�����������ֿ��ܣ�
			// 1.Boundrect[i] ����һ֡�����������ڵ������������һ��������һ֡�У����� kcf tracker ֻ����������һ������
			// 2.Boundrect[i] ����һ֡��ĳһ��Ŀ��� BGS ��Ƭ��segmentation�����������ڱ���ģ��ͨ�����ܺܺõش�����ա������ȵ�ͻȻ�仯
			// 3.Boundrect[i] ��һ���³��ֵ�����

				int judge = 0;
				int label = 0;

                // ����ǵ�һ�������Boundrect[i]/(COR)i ����һ֡�����������ڵ������������һ��������һ֡�У����� kcf tracker ֻ����������һ������
				for (int m = 0; m < boundRect_inFrame.size(); m++) {
					vector<int> tracker_overlap;

					// group_whenOcclusion[m].height * group_whenOcclusion[m].width != 0 ��ʾ��һ֡�У�
					// �� m �� kcf tracker �� group �У�Ҳ���Ǻ�������Ŀ�껥���ڵ�
					if ((group_whenOcclusion[m].height * group_whenOcclusion[m].width) != 0) {

                        // ����� m �� kcf tracker �͵� i �� Boundrect ���ص��Ĳ��֣���ô�ͱ��� boundRect_inFrame
                        // Ҳ�����ж��Ƿ��������� kcf tracker ����� m �� kcf tracker ���ص�
						if (bbOverlap(group_whenOcclusion[m], Boundrect[i]) > 0.20) {
							tracker_overlap.insert(tracker_overlap.end(), m);

                            // �ҵ�����һ�� kcf tracker ����� m �� kcf tracker ���ص�������˵��ͬһ�� group ���棨���ж� iou > 0.9��
							for (int k = 0; k < boundRect_inFrame.size(); k++) {
								if (k != m) {
									if (group_whenOcclusion[k].height * group_whenOcclusion[k].width != 0) {
										if (bbOverlap(group_whenOcclusion[m], group_whenOcclusion[k]) > 0.9) {
											/// they are in the same group in the previous frame
											tracker_overlap.insert(tracker_overlap.end(), k);
										}
									}
								}
							}// for

							// find the bounding box it matches them give the other one to this object
							// ������ m, k ���� kcf tracker ���ص�����ô���ڵ�ǰ�� BGS ������һ�� COR �������� kcf tracker ���ص�
							// ����ҵ��˵Ļ���Ҳ���� bound_find ==> bf ��� COR������ô�����
							// iou((COR)bf, kcf tracker[0]) > iou((COR)bf, kcf tracker[1]) ��ô�Ͱ� (COR)bf -> kcf tracker[0]��(COR)i -> kcf tracker[1]
							// iou((COR)bf, kcf tracker[0]) < iou((COR)bf, kcf tracker[1]) ��ô�Ͱ� (COR)bf -> kcf tracker[1]��(COR)i -> kcf tracker[0]
							// �����֮���ǰ��ص��ȸ��ߵ� COR ������Ӧ�� kcf tracker ���и���
							if (tracker_overlap.size() == 2) {
								int bound_find = 0; 
								float max = 0;

								for (int k = 0; k < Boundrect.size(); k++) {
									if (k != i) {
										float judge = bbOverlap(Boundrect[k], boundRect_inFrame[tracker_overlap[0]]) + bbOverlap(Boundrect[k], boundRect_inFrame[tracker_overlap[1]]);
										if (judge > max){
											max = judge;
											bound_find = k;
										}
									}
								}// for

								if (max > 0) {
									/// deliver the second KCF tracker to it.
									if (bbOverlap(Boundrect[bound_find], boundRect_inFrame[tracker_overlap[0]]) > bbOverlap(Boundrect[bound_find], boundRect_inFrame[tracker_overlap[1]])){
										deliver_tracker(tracker_overlap[1], Boundrect[i]);
									}
									/// deliver the first KCF tracker to it.
									else {
										deliver_tracker(tracker_overlap[0], Boundrect[i]);
									}

									judge = 1;
								}// if max > 0
							}
						}// if > 0.2
					}
				}// for

				// Then we find if it is segment of some objects.
				if (judge == 0) {
					for (int k = 0; k < boundRect_inFrame.size(); k++) {
						if (bbOverlap(Boundrect[i], boundRect_inFrame[k]) > 0.20) {
							// judge = 1 ������ʾ Boundrect[i] ����� kcf tracker[k] Ŀ���ϵ�һ����Ƭ��segmentation��
							judge = 1;
							// ������ǰ��� for ѭ��
							break;
						}
					}// for
				}// if

				// it is an new object
				// ���� Boundrect[i] �Ȳ��Ǹ��ٵ� BGS �е���Ƭ��Ҳ������Ϊ����������������ɵ��ڵ�����ô��Ӧ�����³����ڻ����е�����
				if (judge == 0) {
					Create_new_obj(Boundrect[i]);
					Find_Tracker.insert(Find_Tracker.end(), i);
					add_this_frame.insert(add_this_frame.end(), 0);
					KCF_match.insert(KCF_match.end(), boundRect_inFrame.size(), 0);
				}// if

			}// else
		}// for occlusion occurs.

		/// the boundrect not found.
		// ��� kcf tracker ���κ�һ�� BGS ���ɵ�Ŀ�궼�޷�ƥ��Ļ���˵����� kcf tracker ���ٵ���������Ѿ��뿪������
		for (int i = 0; i < boundRect_inFrame.size(); i++) {
			if (Find_Tracker[i] == -1 || ((currentFrame - Rectsave_Frame[i][Rectsave_Frame[i].size() - 1]) > 1)) {
				delay_toDeleteinFrame[i]++;
			}
		}// for

		/// delete KCF trackers that not exist for 8 frames
		// ���һ���������� 8 ֡�����ɼ�, ��ô��˵����������Ѿ��뿪������
		// ��Ҫ����� kcf tracker ��ɾ����
		for (int i = 0; i < boundRect_inFrame.size();) {
			if (delay_toDeleteinFrame[i] >= 8) {
				if (BoundRect_save[i].size() >= 5) {
					SaveToXML(i);
				} else {
					turn_back.insert(turn_back.end(), boundRect_labelinFrame[i]);
				}
				// �� active tracks ��ɾ������� kcf tracker
				delete_obj(Find_Tracker, add_this_frame, i);
			} else { 
				i++; 
			}
		}

		/// find out redundant KCF tracker that overlap substantially.
		vector<int> calculated;
		calculated.insert(calculated.end(), boundRect_inFrame.size(), 0);

	}//not init status

	for (int i = 0; i < boundRect_inFrame.size(); i++) {
		if (delay_toDeleteinFrame[i] != 0)
			rectangle(frame, boundRect_inFrame[i], Scalar(255, 0, 0), 2, 1);
	}

	cout << "current_frame=" << currentFrame << endl;
}// KCF_tracker

int main() {

	// clock() �������س���ʼִ�к����õ�ʱ��
	clock_t start, finish;
	start = clock();
	// �� filename ��ȡ��Ƶ��
	cv::VideoCapture capture(filename);
	capture.set(cv::CAP_PROP_POS_FRAMES, currentFrame);


	if (!capture.isOpened()){
		cout << "load video fails." << endl;
		return -1;
	}

	// calculate whole numbers of frames. 
	// ������Ƶ�ܹ���֡��
	long totalFrameNumber = capture.get(cv::CAP_PROP_FRAME_COUNT);
	cout << "Total= " << totalFrameNumber << " frames" << endl;

	// �趨����Ƶ��ʼ�ͽ�����֡��
	int frameToStop = totalFrameNumber;
	cout << "Start from " << frameToStart << " frame" << endl;

	// set end frame 
	if (frameToStop < frameToStart) {
		cout << "err, no much frame" << endl;
		return 0;
	} else {
		cout << "End frame is " << frameToStop << " " << endl;
	}

	double rate = capture.get(cv::CAP_PROP_FPS);
	int delay = 1000 / rate;

	bool stop(false);

	while (!stop) {
		// �����Ƶ��ȡ��ϣ���ô�� goto label ���֣������ݱ��浽 xml �ļ���
		if (!capture.read(frame)) {
			cout << "  Cannot read video.  " << endl;
			goto label;
		}

		// the input path of background subtraction image.
		// filepath ��ʾ���ǻ�ϸ�˹����ģ�ʹ������ͼƬ·��
		char filepath[100];
		// sprintf_s ��һ���������亯�������ǽ����ݸ�ʽ��������ַ���, sprintf_s ���ڸ�ʽ�� string �еĸ�ʽ�����ַ�����Ч�Խ����˼��
		// ��������������������ǽ���ַ F:/visual studio/repo/MKCF/MKCF/rouen_bgs/%08d.png �� %08d �滻�� currentFrame ����
		// �����滻�� F:/visual studio/repo/MKCF/MKCF/rouen_bgs/00000150.png����ȡ����ַ��Ϳ��Զ�ȡ bg ͼƬ
		sprintf_s(filepath, 500, "F:/visual studio/repo/MKCF/MKCF/bgs/rouen_bgs/%08d.png", currentFrame);// rouen

		// add mask for rene video
		// ��ȡ��Ӧ��Ƶ���ı���ģ��ͼƬ���� background substraction
		Mat foreground = imread(filepath, CV_8U);

		//two dimensional Points
		vector<vector<Point>> contours;  

		// findContours���������������������
		// countours����һ��˫��������������ÿ��Ԫ�ر�����һ���������� Point ���ɵĵ�ļ��ϵ�������ÿһ��㼯����һ���������ж���������contours���ж���Ԫ�أ�
		// cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE����ֻ�����������������ұ������������е�
		findContours(foreground, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, Point(0, 0));
		
		// a vector for storing boundRect for each blob.
		// �����б���������ǰ���ȡ����������С����
		vector<Rect2d> boundRect(contours.size());
		
		for (int i = 0; i < contours.size(); i++) {
			// boundingRect �������������Ĵ�ֱ�߽���С���Σ���������ͼ�����±߽�ƽ�е�
			boundRect[i] = boundingRect(Mat(contours[i]));
		}

		// select potential objects.
		// ɸ����������������Ŀ�����
		for (int i = 0; i < boundRect.size(); ) {
			// ��������������������е�һ���Ļ���
			// 1.width / height > 6.7
			// 2.width / height < 0.15
			// 3.height * width < blob_size
			// �Ͱ����Ŀ�����ɾ������������Ŀ��
			if (((boundRect[i].width * 1.0 / boundRect[i].height) > 6.7)
				|| ((boundRect[i].width * 1.0 / boundRect[i].height) < 0.15)
				|| (boundRect[i].height * boundRect[i].width < Min_blob_size)) {
				boundRect.erase(boundRect.begin() + i);
			} else {
				i++;
			}
		}//for

		drawContours(foreground, contours, -1, cv::Scalar(255), 2);

		vector<int> flag(boundRect.size());
		vector<Point2f> centroid(boundRect.size());

		// �������Ŀ���������� gravity center
		for (int i = 0; i < boundRect.size(); i++)
			centroid[i] = aoiGravityCenter(foreground, boundRect[i]);

		// flag ����������ʾÿһ��Ŀ���Ƿ�Ӧ�ñ�ɾ��
		for (int i = 0; i < boundRect.size(); i++)
			flag[i] = 0;

		// �������Ŀ����ο�֮��ľ��룬�������̫С���ͽ����������κϲ���һ������ľ���
		for (int i = 0; i < boundRect.size(); i++) {
			if (flag[i] == 1)
				continue;

			if (boundRect[i].width * boundRect[i].height == 0) {
				flag[i] = 1; 
				continue;
			}

			for (int j = i + 1; j < boundRect.size(); j++) {
				// �ж�����Ŀ����ο�����ľ����Ƿ�С����ֵ
				if (CentroidCloseEnough(centroid[i], centroid[j])) {
					// boundRect[i] �� Rect2d ���͵Ķ���Rect2d ��λ����� | ���������أ��������£�
					//
					// x1 = min(a.x, b.x)
					// y1 = min(a.y, b.y)
					// a.width = max(a.x + a.width, b.x + b.width) - x1
					// a.height = max(a.y + a.height, b.y + b.height) - y1
					// a.x = x1
					// a.y = y1
					//
					// boundRect[i] | boundRect[j] ���������ͬʱ����������Ŀ����εĽ�����Σ�������������㣬
					// ��ʵҲ���ǽ��������κϲ���һ������ľ��Σ����ұ������е� i����ɾ���� j
					boundRect[i] = boundRect[i] | boundRect[j];
					// boundRect[j] is going to be deleted.
					// ��� boundRect[j] Ҫ��ɾ�� 
					flag[j] = 1; 
				}
			}// for

		}// for

		for (int i = 0; i < boundRect.size();) {
			if (flag[i] == 1) {
				// erase ������������ɾ�� vector �����е�һ������һ��Ԫ�أ���ɾ��һ��Ԫ�ص�ʱ�������Ϊָ����ӦԪ�صĵ�����
				// boundRect.begin ���������ڻ�ȡһ�����������������������ָ�� vector �ĵ�һ��Ԫ��
				boundRect.erase(boundRect.begin() + (i));
				centroid.erase(centroid.begin() + (i));
				flag.erase(flag.begin() + i);
			} else {
				i++;
			}
		}// for

		// ʹ�ö�� KCF �˲������ж�Ŀ�����
		KCF_tracker(frame, boundRect, centroid);

		for (int i = 0; i < boundRect_inFrame.size(); i++) {
			putText(frame, showMsg[i], cv::Point(boundRect_inFrame[i].x, boundRect_inFrame[i].y + boundRect_inFrame[i].height * 0.5), cv::FONT_HERSHEY_COMPLEX, 0.7, Scalar(255, 255, 255));
		}

		for (int i = 0; i < boundRect.size(); i++) {
			rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(100, 255, 0), 2, 8, 0);
		}

		imshow("Original video", frame);
		imshow("foreground", foreground);

		//Esc to quit.
		int c = cv::waitKey(delay);

		if ((char)c == 27 || currentFrame >= frameToStop) {
			stop = true;
		}

		if (c >= 0) {
			cv::waitKey(0);
		}

		currentFrame++;
	}//while

	// when the video is end. save data of all the objects.
label:
	cout << "label" << endl;

	for (int i = 0; i < boundRect_inFrame.size(); i++) {
		if (BoundRect_save[i].size() >= 5) {
			SaveToXML(i);
		}
	}//for
	ofstream outdata;
	outdata.open("bool.txt", ios::app);//ios::app��β��׷�ӵ���˼
	outdata << "</Video>";

	finish = clock();
	double totaltime;
	totaltime = (double)(finish - start) * 1000 / CLOCKS_PER_SEC;        //�����ms
	cout << "total_time=" << totaltime << "ms" << endl;
	cout << "--------------------------------------------------------------------------------";

}//main