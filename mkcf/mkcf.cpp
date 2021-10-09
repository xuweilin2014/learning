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
// 距离阈值，用来判断两个目标距离是否过近，如果小于这个阈值的话，就将这两个矩形框合并成一个大的矩形框
const int Segment_merge = 3000;
// square
// 面积阈值，如果 blob 的面积小于这个阈值的话，就删除
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
// 用来保存上一帧中的目标矩形框大小和坐标
vector<Rect> boundRect_inFrame;         
// save objects' ID
// 用来保存当前帧中目标的 id，目标 id 由全局变量 obj_num 来生成
vector<int> boundRect_labelinFrame;
// objects not visible for 8 frames will be discarded
// delay_toDeleteinFrame 列表中记录的就是各个 kcf tracker 没有匹配的帧数，
// 当无匹配的帧数达到 8 时，就认为这个物体已经离开了画面
vector<int> delay_toDeleteinFrame;
// Group objects by saving their bounding box of previous frame.
// 当不同的 kcf tracker 在当前帧中互相遮挡时，也就是一个 COR 对应着多个 TO,
// 比如 kcf tracker[0], kcf tracker[2] 互相遮挡，在新一帧中，(COR)i 所占的面积等于 (TO)0、(TO)2
// 因此，我们需要把 group_whenOcclusion[0] = group_whenOcclusion[2] = Boundrect[i]
// 而 Boundrect[i] 赋值给了 boundRect_inFrame[i] 又赋值给了 tracker->update，而 update 方法会更新目标的位置
// group_whenOcclusion 列表保存的就是当前互相遮挡的物体 kcf tracker 在当前帧更新后的位置 bounding box
vector<Rect2d> group_whenOcclusion;
// objects' substantially overlap for 8 frames will be discarded
// vector<int> KCF_occlusionTime;      
// save ID's will be reused in the future
vector<int> turn_back;              
// save objects' KCF trackers
// 用来保存每一个目标对应的 kcf tracker
vector<Ptr<Tracker>> tracker_vector; 
// for labeling object
// 用来保存目标的 id 字符串，这里在视频帧上显示时会用到
vector<string> showMsg;       
// 在将 id 转变为字符串时会作为变量用来存储 str(id)
string save_label;
// stringstream 主要用来进行数据类型转换，这里就是将 int 转换为 string 类型
stringstream ss;
// if there is no objects previous.
// 如果前一帧中没有一个目标的话，就将 prevNo_obj 设置为 1
int prevNo_obj = 1;                   
// total number of objects
// 视频帧中的目标总数
int obj_num = 0;                      

// to save bounding boxes' path for output xml
// BoundRect_save 是一个二维列表，用来保存每一个 kcf tracker 跟踪的物体所出现的轨迹 bounding box
vector<vector<Rect>> BoundRect_save;    
// to save the frame_numbers that the object appears. 
// Rectsave_Frame 是一个二维列表，用来保存每一个 kcf tracker 跟踪的物体所出现的帧数
vector<vector<int>> Rectsave_Frame;     
// to save objects' starting frame. within this val, bounding box can rematch between two frames float x,float y
vector<int> obj_appear_frame;           

// 判断 a, b 两点之间的距离是否小于设定的阈值
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
// 参考文献 http://webdoc.sub.gwdg.de/ebook/serien/ah/UU-CS/2001-57.pdf 中 gravity center 的计算方式
// 即将目标矩形 area 中所有属于目标 A 的点的 x 坐标和 y 坐标分别相加，再除以目标 A 中的像素总的个数:
// (sum(A 的 x 坐标) + sum(A 的 y 坐标)) / card(A)，card(A) 用来求出 A 目标区域的像素个数
Point aoiGravityCenter(Mat &src, Rect area){

	float sumx = 0, sumy = 0;
	float num_pixel = 0;
	Mat ROI = src(area);

	for (int x = 0; x < ROI.cols; x++) {
		for (int y = 0; y < ROI.rows; y++) {
			int val = ROI.at<uchar>(y, x);

			// val >= 50 说明这个像素位于目标区域
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

// stringstream 主要用来进行数据类型转换，这里就是将 int 转换为 string 类型
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

	// boundRect_inFrame 列表中保存的就是上一帧中各个 kcf tracker 更新后的矩形框
	boundRect_inFrame.insert(boundRect_inFrame.end(), Boundrect);
	// delay_toDeleteinFrame 列表中记录的是 kcf tracker 跟踪器失配的帧数，
	// 当失配的帧数达到 8 时，就认为这个 kcf tracker 跟踪的目标已经从画面中离开，
	// 从而会将这个 kcf tracker 从 active tracks 中删除掉
	delay_toDeleteinFrame.insert(delay_toDeleteinFrame.end(), 0);

	Rect temp;
	temp.x = temp.y = temp.height = temp.width = 0;
	group_whenOcclusion.insert(group_whenOcclusion.end(), temp);

	TrackerKCF::Params param;
	// KCF 跟踪器所使用的特征，这里使用 CN + GRAY
	param.desc_pca = TrackerKCF::MODE::CN | TrackerKCF::MODE::GRAY;
	Ptr<TrackerKCF> tracker = TrackerKCF::create(param);

	// 初始化 KCF 跟踪器，并且将跟踪器保存到 tracker_vector 列表中
	tracker->init(frame, Boundrect);
	tracker_vector.insert(tracker_vector.end(), tracker);

	// rect_save 是一个列表，用来保存这个 kcf tracker 跟踪的物体出现的轨迹 bounding box
	vector<Rect> rect_save;
	rect_save.insert(rect_save.end(), Boundrect);
	BoundRect_save.insert(BoundRect_save.end(), rect_save);

	// rectsave_obj 是一个列表，用来保存这个 kcf tracker 跟踪的物体出现的帧数
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
	outdata.open("bool.txt", ios::app);//ios::app是尾部追加的意思
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
	// prevNo_obj 为 1 表示前一帧中没有物体
	if (prevNo_obj == 1) {
		for (int i = 0; i < Boundrect.size(); i++){
			// 前一帧里面没有物体，因此为这一帧新出现的目标创建对应的跟踪器 kcf tracker
			// 并且创建两个一维列表，分别用来保存被跟踪的物体轨迹和物体的出现的帧数
			Create_new_obj(Boundrect[i]);
		}
		prevNo_obj = 0;
	}// if
	else {
		// no object now, come back to initial status
		// 如果 tracker_vector 的 size 为 0，表明前一帧中没有 tracker
		if (tracker_vector.size() == 0) {
			prevNo_obj = 1;
			return;
		}

		// identify whether KCF tracker match in this frame or not
		vector<int> Find_Tracker;
		// 往 Find_Tracker 向量最后插入 boundRect_inFrame.size() 个 -1，在最后会判断 Find_Tracker[i] 是否为 -1，
		// 如果为 -1 的话，那么就说明 kcf tracker 在当前帧中没有任何 COR 和其匹配
		Find_Tracker.insert(Find_Tracker.end(), boundRect_inFrame.size(), -1);

		// identify whether it's property saved to xml or not
		vector<int> add_this_frame(boundRect_inFrame.size(), 0);

		// calculate how many existing objects in Bounding Boxes of current frame. 
		// KCF_Num_Blob 初始化为一个长度为 Boundrect.size() 并且全部为 0 的向量
		// KCF_Num_Blob[i] = j, 表明当前帧中第 i 个目标框 (COR)i 和 j 个 kcf tracker 相匹配
		vector<int> KCF_Num_Blob(Boundrect.size(), 0);

		// save match property for each existing objects.
		// KCF_match 初始化为一个长度为 boundRect_inFrame.size() 并且全部为 0 的向量，表示 kcf tracker[i] 和当前帧中哪一个 BGS 相匹配
		// KCF_match[i] = j, 表明第 i 个 kcf tracker 和当前帧中的 BGS 的第 j 个目标相匹配
		// KCF_match[i] = -1, 表明第 i 个 kcf tracker 和当前帧中的任何 BGS 目标都不匹配
		vector<int> KCF_match(boundRect_inFrame.size(), 0);

		// match objects by comparing overlapping rates of objects saved in previous frame
		// 通过计算前一帧中 kcf tracker 的 bbx 和当前帧中 BGS 的 bbx 的 iou 值，来进行匹配
		for (int i = 0; i < boundRect_inFrame.size(); i++) {
			float max = 0;
			// label 就表示当前帧中和 kcf tracker[i] 最匹配（iou 值最高）的 bounding box 为 Boundrect[label] 
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
			// 多个物体互相遮挡的状态（Occluded）
			// KCF_Num_Blob[i] >= 2 就表明一个 COR 对应多个 TO，也就是多个物体在当前帧中发生了互相遮挡
			if (KCF_Num_Blob[i] >= 2) {
				int label = -1;
				float max = 0;
				// within_label 中存储的是和当前帧中第 i 个目标框 COR 相匹配的多个 kcf tracker 的目标框 bounding box 
				vector<int> within_label;

				for (int j = 0; j < boundRect_inFrame.size(); j++) {
					if (KCF_match[j] == i) {
						within_label.insert(within_label.end(), j);
					}// if
				}// for

				for (int j = 0; j < within_label.size(); j++) {
					// within_label[j] 这个 kcf tracker 在当前帧中找到了第 i 个匹配
					// 因此将它的 Find_Tracker 设置为 i，同时将它的 delay 设置为 0
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
						// 如果 new_area 相比于 area 的变化不是特别大，那么就使用 new_area 来更新旧的 area
						if ((new_area > area * 0.8) && (new_area < area * 2.2)) {
							group_whenOcclusion[within_label[j]] = Boundrect[i];
						}// if
					}// else

					// update 方法：
					// frame: the current frame
					// boundingBox: The bounding box that represent the new target location, if true was returned, not modified otherwise
					// 如果 kcf tracker 跟踪上了的话，就调用 update 方法，update 方法会对 bounding box 方法参数进行更新（这里 bounding box 为引用类型），更新为目标的新位置
					tracker_vector[within_label[j]]->update(frame, boundRect_inFrame[within_label[j]]);

					// 判断 within_label[j] 这个 kcf tracker 的信息（即出现的帧数以及路径）是否被保存了
					if (add_this_frame[within_label[j]] == 0) {
						add_this_frame[within_label[j]] = 1;
						// save frame
						// 保存这个 kcf tracker 跟踪的物体出现的帧数到 Rectsave_Frame 中
						Rectsave_Frame[within_label[j]].insert(Rectsave_Frame[within_label[j]].end(), currentFrame);
						// save rect
						// 保存这个 kcf tracker 跟踪的物体的位置和大小到 BoundRect_save 中
						BoundRect_save[within_label[j]].insert(BoundRect_save[within_label[j]].end(), boundRect_inFrame[within_label[j]]);
						rectangle(frame, boundRect_inFrame[within_label[j]], Scalar(255, 0, 0), 2, 1);
					}// if
				}// for

			}// occlusion occurs

			/************************ Object tracking alone  *****************************/
			// 正常的跟踪状态（Tracked）
			// 1. 如果 BGS/(CORi)t 不是很准确的, 也就是说, 发生了碎片化的现象（segmentation）,这时就使用 kcf tracker 更新得到的位置信息来进行更新 
			// 2. 如果 BGS/(CORi)t 是准确的，这个时候就直接使用 (CORi)t 来进行更新，使用 (CORi)t 的好处是可以让 kcf tracker 对尺度大小进行自适应，具体而言就是
			// 使用 (CORi)t 来重新初始化一个新的 kcf tracker，并且删除掉旧的 kcf tracker
			else if (KCF_Num_Blob[i] == 1) {
				
				int label = -1;
				// 第 label 个 kcf tracker 和当前帧中第 i 个目标相匹配
				for (int j = 0; j < boundRect_inFrame.size(); j++) {
					if (KCF_match[j] == i) {
						label = j;
					}
				}// for

				// Find_Tracker[label] 设置为 i，说明第 label 个 kcf tracker 找到了匹配的第 i 个 (CORi)
				Find_Tracker[label] = i;
				delay_toDeleteinFrame[label] = 0;

				// area_new = A(CORi)t
				float area_new = Boundrect[i].width * Boundrect[i].height;
				// area_previous = A(TOi)t
				float area_previous = boundRect_inFrame[label].width * boundRect_inFrame[label].height;

				// 如果 Tol <= (A(TOi)t / A(CORi)t) <= Toh 的话，说明 BGS 因为碎片化（segmentation），就认为 (TOi)t 更可信，
				// 否则，就认为 (CORi)t 更可信，即使用 BGS 提取到的目标位置和大小
				if ((area_previous >= 1.4 * area_new) && (area_previous <= 1.8 * area_new)) {
					// 这里的 boundRect_inFrame[within_label[i]] 表示为 kcf tracker 在当前帧的目标位置 (TOi)t
					// 使用 (TOi)t 来对 kcf tracker 进行更新, 不过使用 (TOi)t 来对 kcf tracker 进行更新不能让 kcf tracker 拥有尺度自适应的功能
					// 这里的 boundRect_inFrame[label] 传递到 update 方法之后会被更新为目标的新位置
					tracker_vector[label]->update(frame, boundRect_inFrame[label]);
				}// if
				else {
					/// Background substraction is more precise
					tracker_vector[label].release();
					tracker_vector.erase(tracker_vector.begin() + label);

					// 创建一个新的 kcf tracker
					// 使用 BGS 也就是背景模型提取到的目标位置和大小的话，可以使得 kcf tracker 的尺度进行自适应
					// 具体的做法就是删除之前旧的 kcf tracker，然后重新初始化一个新的 kcf tracker
					TrackerKCF::Params param;
					param.desc_pca = TrackerKCF::MODE::CN | TrackerKCF::MODE::GRAY;
					Ptr<TrackerKCF> tracker = TrackerKCF::create(param);
					tracker_vector.insert(tracker_vector.begin() + label, tracker);

					/// BGS and KCF are nearly the same.
					// 使用当前帧中的 BGS 来初始化 kcf tracker，并且对 kcf tracker 进行更新
					tracker_vector[label]->init(frame, Boundrect[i]);
					boundRect_inFrame[label] = Boundrect[i];
					// tracker->update 在更新的过程中，同时也会更新传入的 boundRect_inFrame 使其变成物体在当前帧中的新位置
					// 即论文中的由 kcf tracker 所产生的 (TOj)t
					tracker_vector[label]->update(frame, boundRect_inFrame[label]);
				}// else

				/// save in 'xml' file
				if (add_this_frame[label] == 0) {
					add_this_frame[label] = 1;
					// save frame
					// 保存 kcf tracker 跟踪的目标出现的帧数
					Rectsave_Frame[label].insert(Rectsave_Frame[label].end(), currentFrame);
					// save rect
					// 保存 kcf tracker 跟踪的目标的轨迹
					BoundRect_save[label].insert(BoundRect_save[label].end(), boundRect_inFrame[label]);
					rectangle(frame, Boundrect[i], Scalar(255, 0, 0), 2, 1);
				}//if

			}// end if  object tracking alone
			else {   
			/************** With no trackers match *****************/
			/********** firstly, we try to figure out whether two KCF trackers are tracking the same object or not. ***********/
			// 当 Boundrect[i] 没有任何 kcf tracker 和它进行匹配时，即 KCF_Num_Blob[i] = 0，有以下三种可能：
			// 1.Boundrect[i] 是上一帧中两个互相遮挡的物体的其中一个，在这一帧中，两个 kcf tracker 只跟踪了另外一个物体
			// 2.Boundrect[i] 是上一帧中某一个目标的 BGS 碎片（segmentation），这是由于背景模型通常不能很好地处理光照、环境等的突然变化
			// 3.Boundrect[i] 是一个新出现的物体

				int judge = 0;
				int label = 0;

                // 这就是第一种情况，Boundrect[i]/(COR)i 是上一帧中两个互相遮挡的物体的其中一个，在这一帧中，两个 kcf tracker 只跟踪了另外一个物体
				for (int m = 0; m < boundRect_inFrame.size(); m++) {
					vector<int> tracker_overlap;

					// group_whenOcclusion[m].height * group_whenOcclusion[m].width != 0 表示上一帧中，
					// 第 m 个 kcf tracker 在 group 中，也就是和其它的目标互相遮挡
					if ((group_whenOcclusion[m].height * group_whenOcclusion[m].width) != 0) {

                        // 如果第 m 个 kcf tracker 和第 i 个 Boundrect 有重叠的部分，那么就遍历 boundRect_inFrame
                        // 也就是判断是否还有其它的 kcf tracker 和这第 m 个 kcf tracker 相重叠
						if (bbOverlap(group_whenOcclusion[m], Boundrect[i]) > 0.20) {
							tracker_overlap.insert(tracker_overlap.end(), m);

                            // 找到另外一个 kcf tracker 和这第 m 个 kcf tracker 相重叠，或者说在同一个 group 里面（即判断 iou > 0.9）
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
							// 假设有 m, k 两个 kcf tracker 相重叠，那么就在当前的 BGS 中再找一个 COR 与这两个 kcf tracker 相重叠
							// 如果找到了的话（也就是 bound_find ==> bf 这个 COR），那么如果：
							// iou((COR)bf, kcf tracker[0]) > iou((COR)bf, kcf tracker[1]) 那么就把 (COR)bf -> kcf tracker[0]，(COR)i -> kcf tracker[1]
							// iou((COR)bf, kcf tracker[0]) < iou((COR)bf, kcf tracker[1]) 那么就把 (COR)bf -> kcf tracker[1]，(COR)i -> kcf tracker[0]
							// 简而言之就是把重叠度更高的 COR 赋给对应的 kcf tracker 进行更新
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
							// judge = 1 用来表示 Boundrect[i] 这个是 kcf tracker[k] 目标上的一个碎片（segmentation）
							judge = 1;
							// 跳出当前这个 for 循环
							break;
						}
					}// for
				}// if

				// it is an new object
				// 由于 Boundrect[i] 既不是跟踪的 BGS 中的碎片，也不是因为两个物体相遇而造成的遮挡，那么就应该是新出现在画面中的物体
				if (judge == 0) {
					Create_new_obj(Boundrect[i]);
					Find_Tracker.insert(Find_Tracker.end(), i);
					add_this_frame.insert(add_this_frame.end(), 0);
					KCF_match.insert(KCF_match.end(), boundRect_inFrame.size(), 0);
				}// if

			}// else
		}// for occlusion occurs.

		/// the boundrect not found.
		// 如果 kcf tracker 和任何一个 BGS 生成的目标都无法匹配的话，说明这个 kcf tracker 跟踪的物体可能已经离开画面了
		for (int i = 0; i < boundRect_inFrame.size(); i++) {
			if (Find_Tracker[i] == -1 || ((currentFrame - Rectsave_Frame[i][Rectsave_Frame[i].size() - 1]) > 1)) {
				delay_toDeleteinFrame[i]++;
			}
		}// for

		/// delete KCF trackers that not exist for 8 frames
		// 如果一个物体连续 8 帧都不可见, 那么就说明这个物体已经离开画面了
		// 需要把这个 kcf tracker 给删除掉
		for (int i = 0; i < boundRect_inFrame.size();) {
			if (delay_toDeleteinFrame[i] >= 8) {
				if (BoundRect_save[i].size() >= 5) {
					SaveToXML(i);
				} else {
					turn_back.insert(turn_back.end(), boundRect_labelinFrame[i]);
				}
				// 从 active tracks 中删除掉这个 kcf tracker
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

	// clock() 函数返回程序开始执行后所用的时间
	clock_t start, finish;
	start = clock();
	// 从 filename 读取视频流
	cv::VideoCapture capture(filename);
	capture.set(cv::CAP_PROP_POS_FRAMES, currentFrame);


	if (!capture.isOpened()){
		cout << "load video fails." << endl;
		return -1;
	}

	// calculate whole numbers of frames. 
	// 计算视频总共的帧数
	long totalFrameNumber = capture.get(cv::CAP_PROP_FRAME_COUNT);
	cout << "Total= " << totalFrameNumber << " frames" << endl;

	// 设定好视频开始和结束的帧数
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
		// 如果视频读取完毕，那么就 goto label 部分，将数据保存到 xml 文件中
		if (!capture.read(frame)) {
			cout << "  Cannot read video.  " << endl;
			goto label;
		}

		// the input path of background subtraction image.
		// filepath 表示的是混合高斯背景模型处理过的图片路径
		char filepath[100];
		// sprintf_s 是一个函数，其函数功能是将数据格式化输出到字符串, sprintf_s 对于格式化 string 中的格式化的字符的有效性进行了检查
		// 所以下面这个函数，就是将地址 F:/visual studio/repo/MKCF/MKCF/rouen_bgs/%08d.png 中 %08d 替换成 currentFrame 的数
		// 比如替换成 F:/visual studio/repo/MKCF/MKCF/rouen_bgs/00000150.png，获取到地址后就可以读取 bg 图片
		sprintf_s(filepath, 500, "F:/visual studio/repo/MKCF/MKCF/bgs/rouen_bgs/%08d.png", currentFrame);// rouen

		// add mask for rene video
		// 读取对应视频流的背景模型图片，即 background substraction
		Mat foreground = imread(filepath, CV_8U);

		//two dimensional Points
		vector<vector<Point>> contours;  

		// findContours：函数用来检测物体轮廓
		// countours：是一个双重向量，向量内每个元素保存了一组由连续的 Point 构成的点的集合的向量，每一组点集就是一个轮廓，有多少轮廓，contours就有多少元素；
		// cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE：即只检测最外层轮廓，并且保存轮廓上所有点
		findContours(foreground, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, Point(0, 0));
		
		// a vector for storing boundRect for each blob.
		// 矩形列表，用来保存前面获取到的轮廓最小矩形
		vector<Rect2d> boundRect(contours.size());
		
		for (int i = 0; i < contours.size(); i++) {
			// boundingRect 用来计算轮廓的垂直边界最小矩形，矩形是与图像上下边界平行的
			boundRect[i] = boundingRect(Mat(contours[i]));
		}

		// select potential objects.
		// 筛除掉不符合条件的目标矩形
		for (int i = 0; i < boundRect.size(); ) {
			// 如果满足以下三个条件中的一个的话：
			// 1.width / height > 6.7
			// 2.width / height < 0.15
			// 3.height * width < blob_size
			// 就把这个目标矩形删除掉，不属于目标
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

		// 求出所有目标矩形区域的 gravity center
		for (int i = 0; i < boundRect.size(); i++)
			centroid[i] = aoiGravityCenter(foreground, boundRect[i]);

		// flag 数组用来表示每一个目标是否应该被删除
		for (int i = 0; i < boundRect.size(); i++)
			flag[i] = 0;

		// 检查所有目标矩形框之间的距离，如果距离太小，就将这两个矩形合并成一个更大的矩形
		for (int i = 0; i < boundRect.size(); i++) {
			if (flag[i] == 1)
				continue;

			if (boundRect[i].width * boundRect[i].height == 0) {
				flag[i] = 1; 
				continue;
			}

			for (int j = i + 1; j < boundRect.size(); j++) {
				// 判断两个目标矩形框的中心距离是否小于阈值
				if (CentroidCloseEnough(centroid[i], centroid[j])) {
					// boundRect[i] 是 Rect2d 类型的对象，Rect2d 对位运算符 | 进行了重载，具体如下：
					//
					// x1 = min(a.x, b.x)
					// y1 = min(a.y, b.y)
					// a.width = max(a.x + a.width, b.x + b.width) - x1
					// a.height = max(a.y + a.height, b.y + b.height) - y1
					// a.x = x1
					// a.y = y1
					//
					// boundRect[i] | boundRect[j] 求出来的是同时包含这两个目标矩形的结果矩形，类似于求或运算，
					// 其实也就是将两个矩形合并成一个更大的矩形，并且保留其中的 i，而删除掉 j
					boundRect[i] = boundRect[i] | boundRect[j];
					// boundRect[j] is going to be deleted.
					// 标记 boundRect[j] 要被删除 
					flag[j] = 1; 
				}
			}// for

		}// for

		for (int i = 0; i < boundRect.size();) {
			if (flag[i] == 1) {
				// erase 函数可以用于删除 vector 容器中的一个或者一段元素，在删除一个元素的时候，其参数为指向相应元素的迭代器
				// boundRect.begin 方法就用于获取一个迭代器，并且这个迭代器指向 vector 的第一个元素
				boundRect.erase(boundRect.begin() + (i));
				centroid.erase(centroid.begin() + (i));
				flag.erase(flag.begin() + i);
			} else {
				i++;
			}
		}// for

		// 使用多个 KCF 滤波器进行多目标跟踪
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
	outdata.open("bool.txt", ios::app);//ios::app是尾部追加的意思
	outdata << "</Video>";

	finish = clock();
	double totaltime;
	totaltime = (double)(finish - start) * 1000 / CLOCKS_PER_SEC;        //换算成ms
	cout << "total_time=" << totaltime << "ms" << endl;
	cout << "--------------------------------------------------------------------------------";

}//main