import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from rosie_pick_and_place_interfaces.action import Placenew
from geometry_msgs.msg import Pose


class PlacenewActionClient(Node):

    def __init__(self):
        super().__init__('placenew_action_client')

        # Create an action client to call the "rosie1/placenew_service_provider" action of type Placenew
        self._action_client = ActionClient(self, Placenew, 'rosie1/placenew_action')

        # Wait for the action server to be available
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

    def send_goal(self, active_object_num, active_object_type, passive_object_num, passive_object_type, pose_list):
        # Create the goal message
        goal_msg = Placenew.Goal()
        goal_msg.active_object_num = active_object_num
        goal_msg.active_object_type = active_object_type
        goal_msg.passive_object_num = passive_object_num
        goal_msg.passive_object_type = passive_object_type
        goal_msg.passive_marker_id = 10
        goal_msg.active_marker_id = 0
        goal_msg.pose_goalactiveobject_passiveobject = pose_list

        # Send the goal to the action server
        self.get_logger().info('Sending goal to action server...')
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)

        # Define a callback to handle the response from the action server
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected by the action server.')
            return

        self.get_logger().info('Goal accepted by the action server.')

        # Now wait for the result of the action
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        # This method is called whenever there is feedback from the action server
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.filler}')

    def get_result_callback(self, future):
        result = future.result().result

        if result.success:
            self.get_logger().info('Action succeeded! Object placed successfully.')
        else:
            self.get_logger().info('Action failed. Could not place object.')

        # Shutdown the node after getting the result
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    # Create the action client node
    action_client = PlacenewActionClient()

    # Define active and passive object parameters
    active_object_num = 0  # Example value, could be cube number 1
    active_object_type = 0  # 0 for cube, 1 for plank
    passive_object_num = 1  # Example value, could be plank number 2
    passive_object_type = 0  # 0 for cube, 1 for plank

    # Define the poses (list of Pose messages) for goal
    pose_list = []

    # Example: Two poses to represent the placement goal
    pose1 = Pose()
    pose1.position.x = 0.0
    pose1.position.y = 0.0
    pose1.position.z = 0.065
    pose1.orientation.x = 0.0
    pose1.orientation.y = 0.0
    pose1.orientation.z = 0.0
    pose1.orientation.w = 1.0

    pose2 = Pose()
    pose2.position.x = 0.0
    pose2.position.y = 0.0
    pose2.position.z = 0.045
    pose2.orientation.x = 0.0
    pose2.orientation.y = 0.0
    pose2.orientation.z = 0.0
    pose2.orientation.w = 1.0

    pose_list.append(pose1)
    pose_list.append(pose2)

    # Send goal to the action server
    action_client.send_goal(
        active_object_num=active_object_num,
        active_object_type=active_object_type,
        passive_object_num=passive_object_num,
        passive_object_type=passive_object_type,
        pose_list=pose_list
    )

    # Keep the node alive until the result is received
    rclpy.spin(action_client)


if __name__ == '__main__':
    main()
