# RL_navigation
与修改过的ETH的Rotors仿真器配合使用，Rotors仿真器中的位置控制器被覆盖了，接收trajectory消息中的速度指令进行控制。
simenv提供了一个env，将rotors封装成一个类gym的环境，算法通过这个接口控制飞机，重置环境等。
