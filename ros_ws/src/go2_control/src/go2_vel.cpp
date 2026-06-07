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
#include <std_msgs/Float32.h>
#include <algorithm>
#include <cmath>

class CmdVelSplitter
{
public:
    CmdVelSplitter()
        : nh_(), pnh_("~")
    {
        pnh_.param("deadband_x", deadband_x_, 0.02f);
        pnh_.param("deadband_y", deadband_y_, 0.02f);
        pnh_.param("deadband_yaw", deadband_yaw_, 0.03f);
        pnh_.param("max_x", max_x_, 1.5f);
        pnh_.param("max_y", max_y_, 1.0f);
        pnh_.param("max_yaw", max_yaw_, 1.5f);

        pub_x_ = nh_.advertise<std_msgs::Float32>("/cmd_vel_x", 10);
        pub_y_ = nh_.advertise<std_msgs::Float32>("/cmd_vel_y", 10);
        pub_yaw_ = nh_.advertise<std_msgs::Float32>("/cmd_vel_yaw", 10);
        sub_twist_ = nh_.subscribe("/cmd_vel", 10, &CmdVelSplitter::twistCallback, this);
    }

private:
    static float filterValue(double value, float deadband, float max_abs)
    {
        float out = static_cast<float>(value);
        if (std::fabs(out) < deadband) out = 0.0f;
        if (max_abs > 0.0f) out = std::clamp(out, -max_abs, max_abs);
        return out;
    }

    void twistCallback(const geometry_msgs::Twist::ConstPtr &msg)
    {
        std_msgs::Float32 fx, fy, fyaw;
        fx.data = filterValue(msg->linear.x, deadband_x_, max_x_);
        fy.data = filterValue(msg->linear.y, deadband_y_, max_y_);
        fyaw.data = filterValue(msg->angular.z, deadband_yaw_, max_yaw_);
        pub_x_.publish(fx);
        pub_y_.publish(fy);
        pub_yaw_.publish(fyaw);
    }

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber sub_twist_;
    ros::Publisher pub_x_, pub_y_, pub_yaw_;
    float deadband_x_, deadband_y_, deadband_yaw_;
    float max_x_, max_y_, max_yaw_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "go2_cmd_vel");
    CmdVelSplitter splitter;
    ros::spin();
    return 0;
}
