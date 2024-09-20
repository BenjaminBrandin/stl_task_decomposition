import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rosie_pick_and_place_interfaces.action import Pickobject  # Adjust the import based on your package name
from geometry_msgs.msg import Pose, Point
from geometry_msgs.msg import Quaternion
import ros2_numpy as rnp
from scipy.spatial.transform import Rotation


class MyCustomActionClient(Node):

    def __init__(self):
        super().__init__('my_custom_action_client')
        self._action_client = ActionClient(self, Pickobject, "/rosie1/pick_action")
        self._send_goal()

    def _send_goal(self):
        goal_msg = Pickobject.Goal()
        # goal_msg.order = 10  # Example goal value

        ## Populate goal message
        goal_msg.id = '4'
        # --
        pose1 = Pose()
        pose1.position.x = 0.0
        pose1.position.y = 0.0
        pose1.position.z = 0.0
        pose1.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        goal_msg.aruco_poses = [pose1]
        # --
        goal_msg.pregrasp_pose = Pose(
            position=Point(x=0., y=0.02, z=0.1),
            orientation=rnp.msgify(
                Quaternion, Rotation.from_euler("xyz", [180., 0., 90.], degrees=True).as_quat()
            )
        )
        
        goal_msg.grasp_pose = Pose(
            position=Point(x=0., y=0.02, z=0.08),
            orientation=rnp.msgify(
                Quaternion, Rotation.from_euler("xyz", [180., 0., 90.], degrees=True).as_quat()
            )
        )
        # --
        goal_msg.pregrasp_position_tolerance = Point(x=0.01, y=0.01, z=0.01)
        goal_msg.pregrasp_orientation_tolerance = Point(x=5., y=5., z=5.)
        goal_msg.grasp_position_tolerance = Point(x=0.02, y=0.02, z=0.02)
        goal_msg.grasp_orientation_tolerance = Point(x=5., y=5., z=5.)
        ##

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Success: {result.success}')
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.filler}')

def main(args=None):
    rclpy.init(args=args)
    action_client = PerformGraspActionClient()
    rclpy.spin(action_client)


def main(args=None):
    rclpy.init(args=args)
    action_client = MyCustomActionClient()
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
