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
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/PoseStamped.h>
#include <cmath>
#include <mutex>
#include <thread>
#include <atomic>
#include <algorithm>

class PickAndPlaceNode
{
public:
    explicit PickAndPlaceNode(ros::NodeHandle &nh);

private:
    void endEffectorCallback(const geometry_msgs::PoseStamped::ConstPtr &msg);
    void arrayCallback(const std_msgs::Float64MultiArray::ConstPtr &msg);

    void publishPose(const geometry_msgs::PoseStamped &ps, const std::string &tag);
    void setGripper(float width);                    
    bool waitForPosition(const geometry_msgs::PoseStamped &target,
                         double tol = 0.02,
                         double timeout = 12.0,
                         double resend_hz = 2.0);
    void doPick(geometry_msgs::PoseStamped target);
    static double distance(const geometry_msgs::PoseStamped &a,
                           const geometry_msgs::PoseStamped &b);

    ros::NodeHandle nh_;
    ros::Publisher  ee_pub_, grip_pub_, result_pub_;
    ros::Subscriber ee_sub_,  obj_sub_;

    geometry_msgs::PoseStamped home_, cur_;
    std::atomic<bool> busy_{false}, got_pose_{false};
    std::atomic<float> grip_width_{0.06f};           
    const float kOpen = 0.06f;                        
    const float kClose = 0.005f;                    
    std::mutex mtx_;

    geometry_msgs::PoseStamped last_target_;
    ros::Time last_target_stamp_;
    double position_tolerance_ = 0.02;
    double wait_timeout_sec_ = 12.0;
    double goal_republish_hz_ = 2.0;
    double duplicate_target_tol_ = 0.01;
    double duplicate_holdoff_sec_ = 1.0;
    double approach_height_ = 0.10;
};

static const auto MAKE_HOME = []{
    geometry_msgs::PoseStamped p;
    p.header.frame_id = "base_link";
    p.pose.position.x = -0.02943485;
    p.pose.position.y =  0.24787514;
    p.pose.position.z =  0.27064234;
    p.pose.orientation.x = -0.1222970;
    p.pose.orientation.y =  0.1224853;
    p.pose.orientation.z = -0.6964485;
    p.pose.orientation.w = -0.6964196;
    return p;
}();

PickAndPlaceNode::PickAndPlaceNode(ros::NodeHandle &nh): nh_(nh)
{
    ros::NodeHandle pnh("~");
    pnh.param("position_tolerance", position_tolerance_, 0.02);
    pnh.param("wait_timeout_sec", wait_timeout_sec_, 12.0);
    pnh.param("goal_republish_hz", goal_republish_hz_, 2.0);
    pnh.param("duplicate_target_tol", duplicate_target_tol_, 0.01);
    pnh.param("duplicate_holdoff_sec", duplicate_holdoff_sec_, 1.0);
    pnh.param("approach_height", approach_height_, 0.10);
    ee_pub_   = nh_.advertise<geometry_msgs::PoseStamped>("/end_effector/target_pose", 10);
    grip_pub_ = nh_.advertise<std_msgs::Float32>("/gripper/target", 10);
    result_pub_=nh_.advertise<std_msgs::Bool>("/pick_place_result", 1, true);

    ee_sub_  = nh_.subscribe("/end_effector/pose", 10,
                             &PickAndPlaceNode::endEffectorCallback, this);
    obj_sub_ = nh_.subscribe("/arm_end_pose_quaternion", 10,
                             &PickAndPlaceNode::arrayCallback, this);

    ros::Duration(1.0).sleep();
    setGripper(kOpen);
    ROS_INFO("初始化完成，夹爪张开 %.2f cm", kOpen * 100);

    home_ = MAKE_HOME;
    home_.header.stamp = ros::Time::now();
    publishPose(home_, "启动 → 回到 Home");
    waitForPosition(home_, position_tolerance_, wait_timeout_sec_, goal_republish_hz_);
}

void PickAndPlaceNode::setGripper(float width)
{
    width = std::clamp(width, 0.001f, 0.06f);
    grip_width_ = width;
    std_msgs::Float32 msg;
    msg.data = width;
    grip_pub_.publish(msg);
}

void PickAndPlaceNode::endEffectorCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(mtx_);
    cur_ = *msg;
    got_pose_ = true;
}

void PickAndPlaceNode::arrayCallback(const std_msgs::Float64MultiArray::ConstPtr &msg)
{
    if (msg->data.size() < 7) return;

    geometry_msgs::PoseStamped target;
    target.header.frame_id = "base_link";
    target.header.stamp = ros::Time::now();
    target.pose.position.x = msg->data[0];
    target.pose.position.y = msg->data[1];
    target.pose.position.z = msg->data[2];
    target.pose.orientation.x = msg->data[3];
    target.pose.orientation.y = msg->data[4];
    target.pose.orientation.z = msg->data[5];
    target.pose.orientation.w = msg->data[6];

    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (busy_)
        {
            ROS_WARN_THROTTLE(1.0, "抓取流程仍在执行，忽略新的目标");
            return;
        }

        if (!last_target_stamp_.isZero())
        {
            const double dt = (ros::Time::now() - last_target_stamp_).toSec();
            if (dt < duplicate_holdoff_sec_ && distance(last_target_, target) <= duplicate_target_tol_)
            {
                ROS_INFO_THROTTLE(1.0, "检测到重复抓取目标，已忽略");
                return;
            }
        }

        last_target_ = target;
        last_target_stamp_ = ros::Time::now();
        busy_ = true;
    }

    std::thread(&PickAndPlaceNode::doPick, this, target).detach();
}

void PickAndPlaceNode::publishPose(const geometry_msgs::PoseStamped &ps, const std::string &tag)
{
    ee_pub_.publish(ps);
    ROS_INFO("%s → [x=%.3f, y=%.3f, z=%.3f]", tag.c_str(),
             ps.pose.position.x, ps.pose.position.y, ps.pose.position.z);
}

double PickAndPlaceNode::distance(const geometry_msgs::PoseStamped &a,
                                  const geometry_msgs::PoseStamped &b)
{
    double dx = a.pose.position.x - b.pose.position.x;
    double dy = a.pose.position.y - b.pose.position.y;
    double dz = a.pose.position.z - b.pose.position.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool PickAndPlaceNode::waitForPosition(const geometry_msgs::PoseStamped &target,
                                       double tol,
                                       double timeout,
                                       double resend_hz)
{
    ros::Rate rate(50);
    const ros::Time start = ros::Time::now();
    ros::Time last_pub(0);
    const double resend_period = (resend_hz > 0.0) ? (1.0 / resend_hz) : 0.5;

    while (ros::ok()) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (got_pose_ && distance(cur_, target) <= tol) return true;
        }

        const ros::Time now = ros::Time::now();
        if ((now - start).toSec() > timeout) {
            ROS_WARN("等待末端到位超时：timeout=%.2f s tol=%.3f m", timeout, tol);
            return false;
        }

        if (last_pub.isZero() || (now - last_pub).toSec() >= resend_period) {
            auto retry = target;
            retry.header.stamp = now;
            ee_pub_.publish(retry);
            last_pub = now;
        }
        rate.sleep();
    }
    return false;
}

void PickAndPlaceNode::doPick(geometry_msgs::PoseStamped target)
{
    bool ok = true;

    home_.header.stamp = ros::Time::now();
    publishPose(home_, "抓取前 → Home");
    ok = ok && waitForPosition(home_, position_tolerance_, wait_timeout_sec_, goal_republish_hz_);

    geometry_msgs::PoseStamped approach = target;
    approach.pose.position.z += approach_height_;
    approach.header.stamp = ros::Time::now();
    publishPose(approach, "Approach");
    ok = ok && waitForPosition(approach, position_tolerance_, wait_timeout_sec_, goal_republish_hz_);

    if (ok) {
        target.header.stamp = ros::Time::now();
        publishPose(target, "Descent");
        ok = ok && waitForPosition(target, position_tolerance_, wait_timeout_sec_, goal_republish_hz_);
    }

    if (ok) {
        setGripper(kClose);
        ros::Duration(0.8).sleep();

        home_.header.stamp = ros::Time::now();
        publishPose(home_, "抓取后 → Home");
        ok = ok && waitForPosition(home_, position_tolerance_, wait_timeout_sec_, goal_republish_hz_);
    }

    std_msgs::Bool res;
    res.data = ok;
    result_pub_.publish(res);
    busy_ = false;

    if (ok) {
        ROS_INFO("抓取流程完成");
    } else {
        ROS_WARN("抓取流程未完成，已提前结束");
    }
}

int main(int argc, char **argv)
{
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "pick_and_place_node");
    ros::AsyncSpinner spinner(2); spinner.start();

    ros::NodeHandle nh;
    PickAndPlaceNode node(nh);

    ros::waitForShutdown();
    return 0;
}
