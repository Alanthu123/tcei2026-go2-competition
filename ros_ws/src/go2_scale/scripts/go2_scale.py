"""

* 2026无人系统具身智能算法挑战赛 专用代码
* 版权所有 (c) 2026 无人系统具身智能算法挑战赛组委会
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

"""
#!/home/q/anaconda3/envs/inference/bin/python
# -*- coding: utf-8 -*-
import rospy, sys, threading, json, traceback, numpy as np, cv2
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import OccupancyGrid
from PIL import Image

import torch
from transformers import AutoModel, AutoTokenizer


def ensure_model_loaded(func):
    def wrapper(self, *args, **kw):
        if self.model is None or self.tokenizer is None:
            rospy.logwarn("模型未加载，跳过调用。")
            return
        return func(self, *args, **kw)
    return wrapper


class ImageProcessorNode:
    def __init__(self, default_bbox_prompt="请处理图像并返回结果"):
        rospy.loginfo("Python exec: %s", sys.executable)

        self.model, self.tokenizer = None, None
        self._load_model("/root/inference/FM9G4B-V")

        self.bridge            = CvBridge()
        self.latest_cv_image   = None           
        self.current_map_image = None           
        self.frame_lock        = threading.Lock()
        self.map_lock          = threading.Lock()

        self.model_output_pub = rospy.Publisher("model_output", String, queue_size=10)
        self.nav_goal_pub     = rospy.Publisher("llm_nav_goal", String, queue_size=10)
        self.line_ctrl_pub    = rospy.Publisher("/line_follow_control", String, queue_size=10)

        rospy.Subscriber("/camera/front",        ROSImage,       self._camera_front_cb, queue_size=1)
        rospy.Subscriber("/yolo_annotated_image", ROSImage,       self._yolo_image_cb,   queue_size=1)
        rospy.Subscriber("/map",                 OccupancyGrid,  self._map_cb)

        threading.Thread(target=self._command_thread, daemon=True).start()
        rospy.on_shutdown(lambda: rospy.loginfo("Shutting down node ..."))

    def _load_model(self, model_dir):
        try:
            rospy.loginfo("加载模型 %s ...", model_dir)
            self.model = (AutoModel.from_pretrained(
                model_dir, trust_remote_code=True,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16
            ).eval().to("cuda", dtype=torch.bfloat16))
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            rospy.loginfo("模型 & tokenizer 就绪。")
        except Exception as e:
            rospy.logerr("模型加载失败: %s\n%s", e, traceback.format_exc())

    def _camera_front_cb(self, msg):
        self._update_latest_frame(msg)

    def _yolo_image_cb(self, msg):
        self._update_latest_frame(msg)

    def _update_latest_frame(self, ros_img):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")
            with self.frame_lock:
                self.latest_cv_image = cv_img
        except CvBridgeError as e:
            rospy.logerr("CvBridge 转换错误: %s", e)

    def _map_cb(self, data):
        try:
            w, h = data.info.width, data.info.height
            arr  = np.array(data.data, dtype=np.int8).reshape((h, w))
            gray = np.zeros((h, w), dtype=np.uint8)
            gray[arr == 100] = 255
            with self.map_lock:
                self.current_map_image = Image.fromarray(gray)
        except Exception as e:
            rospy.logerr("地图处理错误: %s\n%s", e, traceback.format_exc())

    @ensure_model_loaded
    def _process_line_command(self, user_cmd: str):
        if "停止" in user_cmd:
            act = "stop"
        else:
            act = "start"
        self.line_ctrl_pub.publish(String(data=act))
        rospy.loginfo("巡线控制发布: %s", act)

    # ========== 导航点知识库 (名称 → 坐标) ==========
    NAV_POINTS = {
        "通道":       (6.03, 1.62, 1.57),
        "障碍":       (6.03, 1.62, 1.57),
        "战术拐角区": (6.03, 1.62, 1.57),
        "手榴弹桌子": (0.03, 1.54, 1.57),
        "手电筒桌子": (5.28, 3.65, 1.57),
        "烟雾弹桌子": (0.05, 6.99, 1.57),
        "终点":       (2.86, 8.24, 0.0),
    }

    @classmethod
    def _lookup_nav_point(cls, name_hint: str):
        """根据名称或关键词查找导航点坐标, 返回 (x, y, yaw) 或 None"""
        name_hint = name_hint.strip()
        # 精确匹配
        for key, coords in cls.NAV_POINTS.items():
            if key == name_hint:
                return coords
        # 模糊匹配: 关键词包含在名称中, 或名称包含在关键词中
        for key, coords in cls.NAV_POINTS.items():
            if key in name_hint or name_hint in key:
                return coords
        # 关键词兜底
        kw_map = {
            "拐角": "战术拐角区", "穿越": "战术拐角区", "通道": "通道",
            "手榴弹": "手榴弹桌子", "手电": "手电筒桌子", "手电筒": "手电筒桌子",
            "烟雾弹": "烟雾弹桌子", "终点": "终点",
        }
        for kw, key in kw_map.items():
            if kw in name_hint:
                return cls.NAV_POINTS.get(key)
        return None

    @ensure_model_loaded
    def _process_nav_command(self, user_cmd: str):
        # -------- 先尝试直接用关键词匹配 (不调用LLM, 最快最可靠) --------
        result = self._lookup_nav_point(user_cmd)
        if result is not None:
            x, y, yaw = result
            goal = {"x": x, "y": y, "yaw": yaw}
            rospy.loginfo("导航决策(关键词匹配): %s → %s", user_cmd, goal)
            self.nav_goal_pub.publish(String(data=json.dumps(goal)))
            self.model_output_pub.publish(String(data=f"导航结果: {goal}"))
            return

        # -------- 关键词未命中, 调用LLM --------
        with self.map_lock:
            if self.current_map_image is None:
                rospy.logwarn("暂无地图，无法导航。")
                return
            map_img = self.current_map_image

        point_names = list(self.NAV_POINTS.keys())
        prompt = (
            f"已知导航点: {point_names}\n"
            f"用户指令: {user_cmd}\n"
            f"从已知导航点中选择最匹配的一个, 只输出该导航点的名称, "
            f"不要输出坐标, 不要输出JSON, 不要输出其他内容。"
        )

        try:
            res = self.model.chat(
                image=map_img,
                msgs=[{"role":"user", "content": prompt}],
                tokenizer=self.tokenizer
            )
            name = str(res).strip()
            # 清理常见多余格式
            for prefix in ["导航点名称:", "名称:", "导航点:", "匹配结果:"]:
                if prefix in name:
                    name = name.split(prefix)[-1].strip()
            # 去引号
            name = name.strip('"').strip("'").strip("。").strip()
            rospy.loginfo("LLM 返回导航点名称: %s", name)

            result = self._lookup_nav_point(name)
            if result is not None:
                x, y, yaw = result
                goal = {"x": x, "y": y, "yaw": yaw}
                rospy.loginfo("导航决策(LLM+查表): %s → %s", name, goal)
                self.nav_goal_pub.publish(String(data=json.dumps(goal)))
                self.model_output_pub.publish(String(data=f"导航结果: {goal}"))
            else:
                rospy.logerr("LLM 返回的名称 '%s' 无法匹配任何已知导航点", name)
        except Exception as e:
            rospy.logerr("导航解析失败: %s\n%s", e, traceback.format_exc())

    @ensure_model_loaded
    def _custom(self, prompt: str):
        with self.frame_lock:
            cv_img = None if self.latest_cv_image is None else self.latest_cv_image.copy()

        pil_img = None
        if cv_img is not None:
            pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        else:
            rospy.logwarn("当前无相机帧，改用纯文本模式。")

        try:
            res = self.model.chat(
                image=pil_img,
                msgs=[{"role": "user", "content": prompt}],
                tokenizer=self.tokenizer
            )
            out = str(res).strip()
            self.model_output_pub.publish(String(data=out))
            rospy.loginfo("模型输出: %s", out)
        except Exception as e:
            rospy.logerr("视觉/抓取处理失败: %s\n%s", e, traceback.format_exc())

    def _command_thread(self):
        grab_kw = ["抓取", "拿取", "拾取", "位置", "识别"]
        line_kw = ["巡线", "沿线", "停止巡线"]
        nav_kw  = ["导航", "穿过", "穿越", "终点", "桌子", "手榴弹", "烟雾弹", "手电", "通道", "障碍", "拐角", "物资"]

        while not rospy.is_shutdown():
            try:
                cmd = input("\n请输入指令: ").strip()
                if not cmd:
                    continue

                if any(k in cmd for k in grab_kw):
                    rospy.loginfo("→ 抓取/视觉")
                    self._custom(cmd)
                elif any(k in cmd for k in line_kw):
                    rospy.loginfo("→ 巡线")
                    self._process_line_command(cmd)
                elif any(k in cmd for k in nav_kw):
                    rospy.loginfo("→ 导航")
                    self._process_nav_command(cmd)
                else:
                    rospy.loginfo("→ 默认（视觉）")
                    self._custom(cmd)

                rospy.sleep(0.05)
            except Exception as e:
                rospy.logerr("输入线程错误: %s\n%s", e, traceback.format_exc())


if __name__ == '__main__':
    rospy.init_node("ros_image_processor", anonymous=True)
    ImageProcessorNode()
    rospy.spin()
