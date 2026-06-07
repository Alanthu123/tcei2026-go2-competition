/*

* 2025无人系统具身智能算法挑战赛 专用代码
* 版权所有 (c) 2025 无人系统具身智能算法挑战赛组委会
* 
* 本源码仅限本赛事参赛团队在比赛过程中使用，
* 禁止任何形式的商业用途、非授权传播或用于其他非比赛场景。
* 
* 依照 GNU 通用公共许可证（GPL）条款授权：
* 参赛者可基于赛事目的对源码进行修改和扩展，
* 但修改后的代码仍受限于本声明的约束条款。
* 
* 本源码按"现状"提供，组委会不承担任何明示或暗示的担保责任，
* 包括但不限于适销性或特定用途适用性的保证。

*/
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <xmlrpcpp/XmlRpcValue.h>
#include <vector>
#include <sensor_msgs/image_encodings.h>

class AdvancedLineFollower
{
public:
    explicit AdvancedLineFollower(ros::NodeHandle &nh)
        : it_(nh), run_line_follow_(false), ep_(0.0), ei_(0.0), el_(0.0)
    {
        nh.param("linear_speed", v_forward_, 0.8);
        nh.param("linear_speed_min", v_min_, 0.3);
        nh.param("pid_kp", kp_, 0.025);
        nh.param("pid_ki", ki_, 0.0);
        nh.param("pid_kd", kd_, 0.002);
        nh.param("ang_limit", ang_lim_, 1.5);
        nh.param("scan_rows", scan_rows_, 10);
        nh.param("v_thresh", v_thresh_, 60.0);
        nh.param("err_smooth_alpha", err_alpha_, 0.4);
        nh.param("speed_decay_width", speed_decay_width_, 80.0);

        nh.param("perspective/enable", use_bev_, true);
        if (use_bev_)
        {
            std::vector<double> s, d;
            if (!arr(nh, "perspective/src", s) ||
                !arr(nh, "perspective/dst", d) ||
                s.size() != 8 || d.size() != 8)
            {
                use_bev_ = false;
            }
            else
            {
                cv::Point2f src[4], dst[4];
                for (int i = 0; i < 4; ++i)
                {
                    src[i] = cv::Point2f(float(s[2 * i]), float(s[2 * i + 1]));
                    dst[i] = cv::Point2f(float(d[2 * i]), float(d[2 * i + 1]));
                }
                M_ = cv::getPerspectiveTransform(src, dst);
                size_ = cv::Size(int(d[2]), int(d[5]));
                if (M_.empty() || size_.width <= 0 || size_.height <= 0)
                    use_bev_ = false;
            }
        }

        sub_ = it_.subscribe("/camera/front", 1, &AdvancedLineFollower::imageCallback, this);
        cmd_sub_ = nh.subscribe("/line_follow_control", 1, &AdvancedLineFollower::controlCallback, this);
        pub_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

        cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
        cv::namedWindow("BEV or TopRegion", cv::WINDOW_NORMAL);
    }

    ~AdvancedLineFollower()
    {
        cv::destroyAllWindows();
    }

private:
    bool arr(ros::NodeHandle &nh, const std::string &k, std::vector<double> &v)
    {
        XmlRpc::XmlRpcValue x;
        if (!nh.getParam(k, x) || x.getType() != XmlRpc::XmlRpcValue::TypeArray)
            return false;
        v.clear();
        for (int i = 0; i < x.size(); ++i)
        {
            if (x[i].getType() == XmlRpc::XmlRpcValue::TypeInt)
                v.push_back(int(x[i]));
            else if (x[i].getType() == XmlRpc::XmlRpcValue::TypeDouble)
                v.push_back(double(x[i]));
            else
                return false;
        }
        return true;
    }

    void controlCallback(const std_msgs::String::ConstPtr &msg)
    {
        if (msg->data == "start")
            run_line_follow_ = true;
        else if (msg->data == "stop")
        {
            run_line_follow_ = false;
            geometry_msgs::Twist cmd;
            cmd.linear.x = 0.0;
            cmd.angular.z = 0.0;
            pub_.publish(cmd);
        }
    }

    void imageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
        if (!run_line_follow_)
            return;

        cv_bridge::CvImageConstPtr p;
        try
        {
            p = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (const cv_bridge::Exception &)
        {
            return;
        }

        cv::Mat full = p->image;
        if (full.empty())
            return;
        cv::imshow("Original Image", full);

        cv::Mat b;
        if (use_bev_)
        {
            cv::warpPerspective(full, b, M_, size_);
            if (b.empty())
            {
                b = full.clone();
                use_bev_ = false;
            }
        }
        else
        {
            b = full;
        }
        if (b.empty())
            return;
        cv::imshow("BEV or TopRegion", b);

        cv::Mat hsv, mask;
        cv::cvtColor(b, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> ch;
        cv::split(hsv, ch);
        cv::threshold(ch[2], mask, v_thresh_, 255, cv::THRESH_BINARY_INV);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
        if (mask.empty())
            return;
        cv::imshow("Mask on TopRegion", mask);

        std::vector<cv::Point> pts;
        // 只扫描底部 scan_rows_ 行, 聚焦靠近机器人的线段, 响应更快
        if (scan_rows_ > 0 && mask.rows > scan_rows_)
        {
            cv::Rect roi(0, mask.rows - scan_rows_, mask.cols, scan_rows_);
            cv::findNonZero(mask(roi), pts);
            // 偏移回原图坐标
            for (auto &p : pts)
                p.y += (mask.rows - scan_rows_);
        }
        else
        {
            cv::findNonZero(mask, pts);
        }
        bool line_found = (pts.size() > 50);
        double err = 0.0;
        if (line_found)
        {
            cv::Vec4f line;
            cv::fitLine(pts, line, cv::DIST_L2, 0, 0.01, 0.01);
            float vx = line[0], vy = line[1];
            float x0 = line[2], y0 = line[3];
            float rows = float(mask.rows);
            float t = (rows - y0) / vy;
            float x_ground = x0 + t * vx;
            err = double(x_ground - mask.cols / 2.0);

            cv::Point p1(int(x0 - vx * 100), int(y0 - vy * 100));
            cv::Point p2(int(x0 + vx * 100), int(y0 + vy * 100));
            cv::line(b, p1, p2, cv::Scalar(0, 0, 255), 2);
            cv::circle(b, cv::Point(int(x_ground), int(rows)), 5, cv::Scalar(0, 255, 0), -1);

            cv::imshow("BEV or TopRegion", b);
        }

        // 自适应误差平滑: 小误差快速响应，大误差加大平滑
        double smooth_factor = (std::abs(err) < 20.0) ? 0.2 : err_alpha_;
        err = (1.0 - smooth_factor) * err + smooth_factor * ep_;
        ep_ = err;
        // 积分抗饱和: 丢线时衰减积分，防止积分饱和
        if (line_found)
            ei_ += err;
        else
            ei_ *= 0.8;
        double ang = 0.0;
        if (line_found)
        {
            double ed = err - el_;
            ang = -(kp_ * err + ki_ * ei_ + kd_ * ed);
            ang = std::max(std::min(ang, ang_lim_), -ang_lim_);
            el_ = err;
        }

        // 自适应速度: 线在中心时全速, 偏离越大速度越低
        double abs_err_px = std::abs(err);
        double speed_scale = std::max(0.3, 1.0 - abs_err_px / speed_decay_width_);

        geometry_msgs::Twist cmd;
        if (line_found)
        {
            cmd.linear.x = v_forward_ * speed_scale;
            cmd.angular.z = ang;
        }
        else
        {
            cmd.linear.x = 0.0;
            cmd.angular.z = 0.0;
        }
        pub_.publish(cmd);

        cv::waitKey(1);
    }

    image_transport::ImageTransport it_;
    image_transport::Subscriber sub_;
    ros::Subscriber cmd_sub_;
    ros::Publisher pub_;

    double v_forward_, v_min_, kp_, ki_, kd_, ang_lim_, v_thresh_;
    int scan_rows_;
    double err_alpha_, speed_decay_width_;

    bool use_bev_;
    cv::Mat M_;
    cv::Size size_;

    bool run_line_follow_;
    double ep_, ei_, el_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "advanced_line_follower");
    ros::NodeHandle nh("~");
    AdvancedLineFollower node(nh);
    ros::spin();
    return 0;
}
