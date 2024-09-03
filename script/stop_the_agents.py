import rclpy
from rclpy.node import Node
import os
from ament_index_python.packages import get_package_share_directory
import yaml
from geometry_msgs.msg import PoseStamped, Twist


class StopAgents(Node):

    def __init__(self):
        
        
        super().__init__('stop_agents')
        
        # parameters declaration from launch file
        self.declare_parameter('num_robots', rclpy.Parameter.Type.INTEGER)
        
        # Agent Information # check if this can be above the optimization problem
        self.total_agents = self.get_parameter('num_robots').get_parameter_value().integer_value

        self.turtle_vel_pub: dict = {}
        # Setup subscribers
        for id in range(1, self.total_agents+1):
            self.turtle_vel_pub[id] = self.create_publisher(Twist, f"/turtlebot{id}/cmd_vel", 5)

        # Stop the agents
        self.stop_agents_timer = self.create_timer(0.05, self.create_stop_msg)

    def create_stop_msg(self):
        # Stop the agents
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.linear.y = 0.0
        stop_msg.linear.z = 0.0
        stop_msg.angular.x = 0.0
        stop_msg.angular.y = 0.0
        stop_msg.angular.z = 0.0

        for id in range(1, self.total_agents+1):
            self.turtle_vel_pub[id].publish(stop_msg)






def main(args=None):
    rclpy.init(args=args)
    stop_agents = StopAgents()
    rclpy.spin(stop_agents)
    stop_agents.destroy_node()
    rclpy.shutdown() 


if __name__ == "__main__":
    main()