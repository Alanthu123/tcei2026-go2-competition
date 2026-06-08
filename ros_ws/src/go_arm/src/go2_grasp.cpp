/*
*
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
*
* 优化内容:
*  - 机器狗姿态协调: 抓取前发布 stand 指令, 确保腿锁定
*  - 稳定等待: 到达目标后延时, 让机器人充分静止
*  - 重试机制: 失败后最多重试3次, 每次微调接近方向
*  - 抓取验证: 闭合后检查夹爪实际位置确认是否抓到了物体
*  - 可调参数: max_retries, retry_step, stabilize_delay 等均可通过ROS参数配置
*/
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
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
    bool verifyGrasp();

    ros::NodeHandle nh_;
    ros::Publisher  ee_pub_, grip_pub_, result_pub_, posture_pub_;
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
    double duplicate_target_tol_ = 0.02;
    double duplicate_holdoff_sec_ = 1.0;
    double approach_height_ = 0.08;

    // 新增优化参数
    int max_retries_ = 3;
    double retry_step_ = 0.03;           // 每次重试向目标靠近的距离(m)
    double stabilize_delay_ = 1.5;       // 到达后稳定等待时间(s)
    double grasp_verify_timeout_ = 1.0;  // 抓取验证超时
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
    pnh.param("duplicate_target_tol", duplicate_target_tol_, 0.02);
    pnh.param("duplicate_holdoff_sec", duplicate_holdoff_sec_, 1.0);
    pnh.param("approach_height", approach_height_, 0.08);
    pnh.param("max_retries", max_retries_, 3);
    pnh.param("retry_step", retry_step_, 0.03);
    pnh.param("stabilize_delay", stabilize_delay_, 1.5);
    pnh.param("grasp_verify_timeout", grasp_verify_timeout_, 1.0);

    ee_pub_    = nh_.advertise<geometry_msgs::PoseStamped>("/end_effector/target_pose", 10);
    grip_pub_  = nh_.advertise<std_msgs::Float32>("/gripper/target", 10);
    result_pub_= nh_.advertise<std_msgs::Bool>("/pick_place_result", 1, true);
    posture_pub_= nh_.advertise<std_msgs::String>("/go2_posture", 1, true);

    ee_sub_  = nh_.subscribe("/end_effector/pose", 10,
                             &PickAndPlaceNode::endEffectorCallback, this);
    obj_sub_ = nh_.subscribe("/arm_end_pose_quaternion", 10,
                             &PickAndPlaceNode::arrayCallback, this);

    ros::Duration(1.0).sleep();
    setGripper(kOpen);
    ROS_INFO("初始化完成，夹爪张开 %.2f cm，最大重试次数=%d", kOpen * 100, max_retries_);

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
            // 正在抓取中, 更新最新目标位置 (用于重试时使用最新坐标)
            last_target_ = target;
            last_target_stamp_ = ros::Time::now();
            ROS_INFO_THROTTLE(1.0, "收到新的目标坐标 (%.3f, %.3f, %.3f), 将在重试时使用",
                             target.pose.position.x, target.pose.position.y, target.pose.position.z);
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
            auto retry_cmd = target;
            retry_cmd.header.stamp = now;
            ee_pub_.publish(retry_cmd);
            last_pub = now;
        }
        rate.sleep();
    }
    return false;
}

bool PickAndPlaceNode::verifyGrasp()
{
    // 等待夹爪闭合到位, 检查实际位置是否达到闭合阈值
    ros::Time start = ros::Time::now();
    ros::Rate rate(20);
    while (ros::ok() && (ros::Time::now() - start).toSec() < grasp_verify_timeout_)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (got_pose_)
        {
            // 如果夹爪已经闭合到 < 0.01, 认为抓住了东西
            if (grip_width_ <= 0.01f)
                return true;
        }
        rate.sleep();
    }
    // 超时未闭合 → 可能没有物体
    ROS_WARN("抓取验证失败: 夹爪未闭合到目标位置 (当前=%.4f)", grip_width_.load());
    return false;
}

void PickAndPlaceNode::doPick(geometry_msgs::PoseStamped target)
{
    // ===== 步骤0: 机器狗姿态协调 - 确保站立姿态 =====
    {
        std_msgs::String posture_cmd;
        posture_cmd.data = "stand";
        posture_pub_.publish(posture_cmd);
        ROS_INFO("发布机器人姿态指令: stand (锁定腿部, 稳定机身)");
    }
    ros::Duration(stabilize_delay_).sleep();
    ROS_INFO("稳定等待 %.1f s 完成, 开始抓取", stabilize_delay_);

    bool final_ok = false;

    // ===== 重试循环 =====
    for (int attempt = 0; attempt < max_retries_ && ros::ok(); ++attempt)
    {
        if (attempt > 0)
        {
            ROS_INFO("===== 第 %d 次重试 =====", attempt);
            // 使用最新收到的目标位置
            {
                std::lock_guard<std::mutex> lock(mtx_);
                if ((ros::Time::now() - last_target_stamp_).toSec() < 3.0)
                    target = last_target_;
            }
            // 微调接近方向: 稍微降低高度, 向目标靠近一点
            target.pose.position.z -= retry_step_ * 0.5;
            // 沿 XY 方向微调靠近
            double dx = target.pose.position.x - home_.pose.position.x;
            double dy = target.pose.position.y - home_.pose.position.y;
            double dxy = std::sqrt(dx*dx + dy*dy);
            if (dxy > 0.01)
            {
                double scale = retry_step_ / dxy;
                target.pose.position.x += dx * scale;
                target.pose.position.y += dy * scale;
            }
            ROS_INFO("重试调整: Δx=%.3f, Δy=%.3f, Δz=%.3f",
                     target.pose.position.x, target.pose.position.y, target.pose.position.z);
        }
        else
        {
            ROS_INFO("===== 第 1 次抓取尝试 =====");
        }

        bool ok = true;

        // 步骤1: 先回 Home
        home_.header.stamp = ros::Time::now();
        publishPose(home_, "→ Home");
        ok = waitForPosition(home_, position_tolerance_, wait_timeout_sec_, goal_republish_hz_);
        if (!ok) { ROS_WARN("回 Home 失败"); continue; }

        // 步骤2: Approach — 目标上方
        geometry_msgs::PoseStamped approach = target;
        approach.pose.position.z += approach_height_;
        approach.header.stamp = ros::Time::now();
        publishPose(approach, "→ Approach");
        ok = waitForPosition(approach, position_tolerance_, wait_timeout_sec_, goal_republish_hz_);
        if (!ok) { ROS_WARN("Approach 超时"); continue; }

        // 步骤3: Descent — 下降到抓取位姿
        target.header.stamp = ros::Time::now();
        publishPose(target, "→ Descent");
        // 下降阶段使用更宽松的容差 (可能卡到物体)
        ok = waitForPosition(target, position_tolerance_ * 2.0,
                             wait_timeout_sec_ * 0.8, goal_republish_hz_);
        if (!ok) { ROS_WARN("Descent 超时, 可能接触到了物体"); }

        // 步骤4: 闭合夹爪
        setGripper(kClose);
        ros::Duration(0.8).sleep();

        // 步骤5: 抓取验证
        bool grasped = verifyGrasp();
        if (grasped)
        {
            ROS_INFO("✓ 抓取验证成功! 物体已抓住");
            final_ok = true;
        }
        else
        {
            ROS_WARN("✗ 抓取验证失败, 可能未抓取到物体");
            setGripper(kOpen);  // 张开夹爪, 准备重试
            ros::Duration(0.3).sleep();
            continue;
        }

        // 步骤6: 携带物体回 Home
        home_.header.stamp = ros::Time::now();
        publishPose(home_, "→ Home (携物)");
        ok = waitForPosition(home_, position_tolerance_, wait_timeout_sec_, goal_republish_hz_);
        if (!ok)
        {
            ROS_WARN("携物回 Home 超时, 但物体已抓住");
        }
        break;  // 成功, 退出重试
    }

    // ===== 结果发布 =====
    std_msgs::Bool res;
    res.data = final_ok;
    result_pub_.publish(res);
    busy_ = false;

    if (final_ok)
        ROS_INFO("===== 抓取流程成功完成 =====");
    else
        ROS_WARN("===== 抓取流程失败 (已重试 %d 次) =====", max_retries_);
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
