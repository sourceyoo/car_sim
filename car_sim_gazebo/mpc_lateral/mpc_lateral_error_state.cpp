#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/float64.hpp>

#include <QuadProg++.hh>
#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

namespace
{
constexpr double kEpsilon = 1.0e-9;

double normalizeAngle(double a)
{
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

double yawFromQuat(const geometry_msgs::msg::Quaternion & q)
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

double yawFromPath(const nav_msgs::msg::Path & path, const size_t idx)
{
  if (path.poses.empty()) return 0.0;
  if (idx + 1 < path.poses.size()) {
    const auto & p0 = path.poses.at(idx).pose.position;
    const auto & p1 = path.poses.at(idx + 1).pose.position;
    return std::atan2(p1.y - p0.y, p1.x - p0.x);
  }
  if (idx > 0) {
    const auto & p0 = path.poses.at(idx - 1).pose.position;
    const auto & p1 = path.poses.at(idx).pose.position;
    return std::atan2(p1.y - p0.y, p1.x - p0.x);
  }
  return yawFromQuat(path.poses.front().pose.orientation);
}

size_t findNearestIdx(const nav_msgs::msg::Path & path, const geometry_msgs::msg::Pose & pose)
{
  double best_dist = std::numeric_limits<double>::max();
  size_t best_idx = 0;
  for (size_t i = 0; i < path.poses.size(); ++i) {
    const auto & p = path.poses.at(i).pose.position;
    const double dx = pose.position.x - p.x;
    const double dy = pose.position.y - p.y;
    const double dist = dx * dx + dy * dy;
    if (dist < best_dist) {
      best_dist = dist;
      best_idx = i;
    }
  }
  return best_idx;
}

namespace qp = quadprogpp;

qp::Matrix<double> toQPMatrix(const Eigen::MatrixXd & m)
{
  qp::Matrix<double> qp_m(m.rows(), m.cols());
  for (int r = 0; r < m.rows(); ++r) {
    for (int c = 0; c < m.cols(); ++c) {
      qp_m[r][c] = m(r, c);
    }
  }
  return qp_m;
}

qp::Vector<double> toQPVector(const Eigen::VectorXd & v)
{
  qp::Vector<double> qp_v(v.size());
  for (int i = 0; i < v.size(); ++i) {
    qp_v[i] = v(i);
  }
  return qp_v;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> buildPredictionMatrices(
  const Eigen::MatrixXd & Ad, const Eigen::MatrixXd & Bd, const int Np, const int Nc)
{
  const int n = Ad.rows();
  const int m = Bd.cols();

  Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(n * Np, n);
  Eigen::MatrixXd Gamma = Eigen::MatrixXd::Zero(n * Np, m * Nc);

  Eigen::MatrixXd A_power = Eigen::MatrixXd::Identity(n, n);
  for (int i = 0; i < Np; ++i) {
    A_power = Ad * A_power;
    Phi.block(i * n, 0, n, n) = A_power;
  }

  for (int r = 0; r < Np; ++r) {
    for (int c = 0; c < Nc; ++c) {
      if (r >= c) {
        Eigen::MatrixXd A_pc = Eigen::MatrixXd::Identity(n, n);
        for (int k = 0; k < (r - c); ++k) {
          A_pc = Ad * A_pc;
        }
        Gamma.block(r * n, c * m, n, m) = A_pc * Bd;
      }
    }
  }
  return {Phi, Gamma};
}

}  // namespace

class MPCLateralNode : public rclcpp::Node
{
public:
  MPCLateralNode()
  : Node("mpc_lateral"),
    horizon_(declare_parameter<int>("Np", 30)),
    control_horizon_(declare_parameter<int>("Nc", 10)),
    dt_(declare_parameter<double>("Ts", 0.05)),
    wheelbase_(declare_parameter<double>("wheelbase", 2.65)),
    steer_limit_(declare_parameter<double>("steer_limit", 0.7)),
    q_lat_(declare_parameter<double>("Qy", 200.0)),
    q_head_(declare_parameter<double>("Qpsi", 10.0)),
    r_steer_(declare_parameter<double>("Rdelta", 100.0)),
    dmax_deg_(declare_parameter<double>("Dmax_deg", 2.0)),
    path_frame_(declare_parameter<std::string>("path_frame", "map")),
    cmd_topic_(declare_parameter<std::string>("command_topic", "/itusct/command_cmd")),
    path_topic_(declare_parameter<std::string>("path_topic", "csv_path")),
    odom_topic_(declare_parameter<std::string>("odom_topic", "odom")),
    target_speed_(declare_parameter<double>("target_speed", 1.0)),
    dev_dist_thresh_(declare_parameter<double>("deviation_distance_thresh", 2.0)),
    dev_heading_thresh_(declare_parameter<double>("deviation_heading_thresh_rad", 0.5)),
    curvature_lookahead_dist_(declare_parameter<double>("curvature_lookahead_dist", 3.0)),
    ff_gain_(declare_parameter<double>("ff_gain", 0.5)),
    use_odom_steer_est_(declare_parameter<bool>("use_odom_steer_est", false)),
    steer_est_v_thresh_(declare_parameter<double>("steer_est_v_thresh", 0.5)),
    steer_est_lpf_alpha_(declare_parameter<double>("steer_est_lpf_alpha", 0.2))
  {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    rclcpp::QoS path_qos(rclcpp::KeepLast(10));
    path_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
    path_qos.durability(rclcpp::DurabilityPolicy::Volatile);
    path_sub_ = create_subscription<nav_msgs::msg::Path>(
      path_topic_, path_qos, std::bind(&MPCLateralNode::onPath, this, std::placeholders::_1));

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::QoS(20), std::bind(&MPCLateralNode::onOdom, this, std::placeholders::_1));

    cmd_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(cmd_topic_, 10);
    pred_path_pub_ = create_publisher<nav_msgs::msg::Path>("mpc/predicted_path", 1);
    debug_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("mpc/debug_markers", 1);
    lateral_error_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/lateral_error", 10);
    heading_error_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/heading_error", 10);
    qp_time_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/qp_time_ms", 10);
    timer_ms_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/on_timer_ms", 10);
    yaw_rate_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/yaw_rate", 10);
    steering_cmd_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/steering_cmd", 10);
    steering_est_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/steering_est", 10);
    yaw_rate_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/yaw_rate", 10);
    steering_cmd_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/steering_cmd", 10);
    steering_est_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/steering_est", 10);

    timer_ = create_wall_timer(
      std::chrono::duration<double>(dt_), std::bind(&MPCLateralNode::onTimer, this));

    RCLCPP_INFO(get_logger(), "Error-State MPC Ready. Wheelbase: %.2f", wheelbase_);
  }

private:
  void onPath(const nav_msgs::msg::Path::SharedPtr msg)
  {
    if (msg->poses.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Received empty path");
      return;
    }
    latest_path_ = *msg;
    path_ready_ = true;
  }

  void onOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    latest_odom_ = *msg;
    odom_ready_ = true;
  }

  // Helper used for FF if needed (optional in error-state, but kept)
  double computeCurvature(const nav_msgs::msg::Path & path, size_t current_idx, double lookahead_dist)
  {
    if (path.poses.size() <= current_idx + 1) return 0.0;

    double accumulated_dist = 0.0;
    size_t target_idx = current_idx;

    for (size_t i = current_idx + 1; i < path.poses.size(); ++i) {
      const auto & prev_p = path.poses.at(i - 1).pose.position;
      const auto & curr_p = path.poses.at(i).pose.position;
      accumulated_dist += std::hypot(curr_p.x - prev_p.x, curr_p.y - prev_p.y);
      target_idx = i;
      if (accumulated_dist >= lookahead_dist) break;
    }

    if (target_idx == current_idx) return 0.0;

    const double psi_curr = yawFromPath(path, current_idx);
    const double psi_target = yawFromPath(path, target_idx);
    const double dpsi = normalizeAngle(psi_target - psi_curr);
    
    if (accumulated_dist < 0.1) return 0.0;

    return dpsi / accumulated_dist; 
  }

  // [수정된 부분] Error-State MPC Logic
  void onTimer()
  {
    const auto t_cb_start = std::chrono::steady_clock::now();
    auto publish_timer_ms = [&](const std::chrono::steady_clock::time_point & start_tp) {
      if (!timer_ms_pub_) return;
      std_msgs::msg::Float64 msg;
      msg.data = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start_tp).count();
      timer_ms_pub_->publish(msg);
    };

    if (!path_ready_ || !odom_ready_) {
      publish_timer_ms(t_cb_start);
      return;
    }

    // 1. Pose & State Update
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header = latest_odom_.header;
    pose_stamped.pose = latest_odom_.pose.pose;

    if (pose_stamped.header.frame_id == "world" && path_frame_ == "map") {
      pose_stamped.header.frame_id = "map";
    }
    // TF Transform (odom -> map/path_frame)
    if (pose_stamped.header.frame_id != path_frame_) {
      if (tf_buffer_->canTransform(path_frame_, pose_stamped.header.frame_id, tf2::TimePointZero)) {
        try {
          pose_stamped = tf_buffer_->transform(
            pose_stamped, path_frame_, tf2::durationFromSec(0.05));
        } catch (const tf2::TransformException & ex) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF transform failed: %s", ex.what());
          return;
        }
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF unavailable");
        return;
      }
    }
    const auto & pose = pose_stamped.pose;
    const auto & twist = latest_odom_.twist.twist;
    const double speed = std::sqrt(twist.linear.x * twist.linear.x + twist.linear.y * twist.linear.y);

    // Find Nearest Point & Calculate Errors
    const size_t nearest_idx = findNearestIdx(latest_path_, pose);
    const double ref_yaw = yawFromPath(latest_path_, nearest_idx);
    const double heading = yawFromQuat(pose.orientation);
    
    // [ERROR STATE CALCULATION]
    // 1. Heading Error (e_psi)
    const double e_psi = normalizeAngle(heading - ref_yaw);
    
    // 2. Lateral Error (e_y) - Frenet Frame approximation
    const double dx = pose.position.x - latest_path_.poses.at(nearest_idx).pose.position.x;
    const double dy = pose.position.y - latest_path_.poses.at(nearest_idx).pose.position.y;
    // Cross product to find lateral error with sign (Left +, Right -)
    const double e_y = -std::sin(ref_yaw) * dx + std::cos(ref_yaw) * dy;

    // Publish errors for monitoring
    std_msgs::msg::Float64 lat_err_msg; lat_err_msg.data = e_y; lateral_error_pub_->publish(lat_err_msg);
    std_msgs::msg::Float64 head_err_msg; head_err_msg.data = e_psi; heading_error_pub_->publish(head_err_msg);

    // 2. MPC Setup (Error-State Formulation)
    // State Vector x: [e_y, e_psi, v, delta] (Size 4)
    // Input u: [delta_rate] (Size 1) -> delta_rate * Ts = delta_increment
    const int nx = 4; 
    const int nu = 1;
    int Np = std::max(1, horizon_);
    int Nc = std::max(1, control_horizon_);
    Nc = std::min(Nc, Np);
    const double Ts = dt_;
    const double v_ref = std::max(speed, 0.1); // Linearization velocity

    // Sync Steering (Use previous command as state)
    // If you want to use estimated steer, implement logic here, but keeping it simple/safe is better.
    double delta_curr = delta_prev_; 

    // [NEW] Error Dynamics Matrix A (4x4)
    // e_y_next   = e_y + v * sin(e_psi) * dt  ~= e_y + v * e_psi * dt
    // e_psi_next = e_psi + v/L * tan(delta) * dt ~= e_psi + v/L * delta * dt
    // v_next     = v
    // delta_next = delta + u * dt (if u is rate) OR delta + u (if u is increment)
    // We assume u is 'steering increment per step'.
    
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(nx, nx);
    A(0, 1) = v_ref * Ts;                  // e_y depends on e_psi
    A(1, 3) = (v_ref / wheelbase_) * Ts;   // e_psi depends on delta
    // v depends on v (Identity)
    // delta depends on delta (Identity)

    // [NEW] Input Matrix B (4x1)
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(nx, nu);
    B(3, 0) = 1.0; // u affects delta directly (Integrator)

    Eigen::VectorXd xk(nx);
    xk << e_y, e_psi, speed, delta_curr;

    // 3. Reference Generation (Target is all ZEROs)
    Eigen::VectorXd ref_stack = Eigen::VectorXd::Zero(Np * nx);
    for (int i = 0; i < Np; ++i) {
      ref_stack(i * nx + 0) = 0.0;           // Target e_y = 0
      ref_stack(i * nx + 1) = 0.0;           // Target e_psi = 0
      ref_stack(i * nx + 2) = target_speed_; // Target v
      ref_stack(i * nx + 3) = 0.0;           // Target delta = 0 (Neutral)
      // Note: Ideally target delta should be curvature based, but MPC feedback handles it too.
    }

    // 4. QP Formulation
    // Re-map weights to new state order: [e_y, e_psi, v, delta]
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nx, nx);
    Q(0, 0) = q_lat_;    // e_y
    Q(1, 1) = q_head_;   // e_psi
    Q(2, 2) = 1.0;       // v weight
    Q(3, 3) = 0.1;       // delta weight (keep small to allow steering)

    // R is weight on 'steering increment'
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nu, nu) * std::max(r_steer_, 1.0);

    const auto [Phi, Gamma] = buildPredictionMatrices(A, B, Np, Nc);
    
    Eigen::MatrixXd Qbar = Eigen::MatrixXd::Zero(nx * Np, nx * Np);
    for (int i = 0; i < Np; ++i) Qbar.block(i * nx, i * nx, nx, nx) = Q;
    Eigen::MatrixXd Rbar = Eigen::MatrixXd::Zero(nu * Nc, nu * Nc);
    for (int i = 0; i < Nc; ++i) Rbar.block(i * nu, i * nu, nu, nu) = R;
    
    const Eigen::VectorXd X_stack_nom = Phi * xk;
    Eigen::MatrixXd H = Gamma.transpose() * Qbar * Gamma + Rbar;
    H = 0.5 * (H + H.transpose());
    H += 1e-8 * Eigen::MatrixXd::Identity(nu*Nc, nu*Nc);
    
    const Eigen::VectorXd f = Gamma.transpose() * Qbar * (X_stack_nom - ref_stack);

    // 6. Constraints
    const int nv = nu * Nc;
    double u_max = dmax_deg_ * M_PI / 180.0; // Rate limit (rad per step)
    Eigen::VectorXd lb = Eigen::VectorXd::Constant(nv, -u_max);
    Eigen::VectorXd ub = Eigen::VectorXd::Constant(nv, u_max);
    
    // Absolute constraints on Delta
    const int num_step_constraints = 2 * nv;
    const int num_abs_constraints = 2 * Nc;
    const int total_constraints = num_step_constraints + num_abs_constraints;

    Eigen::MatrixXd CI_eig = Eigen::MatrixXd::Zero(nv, total_constraints);
    Eigen::VectorXd ci0_eig = Eigen::VectorXd::Zero(total_constraints);

    // Rate limits (Step constraints)
    for(int i=0; i<nv; ++i) {
        CI_eig(i, i) = 1.0;            ci0_eig(i) = -lb(i);
        CI_eig(i, nv+i) = -1.0;        ci0_eig(nv+i) = ub(i);
    }
    // Absolute limits (Accumulated u + delta_curr)
    for(int k=0; k<Nc; ++k) {
        for(int j=0; j<=k; ++j) {
            CI_eig(j, num_step_constraints+k) = 1.0;
            CI_eig(j, num_step_constraints+Nc+k) = -1.0;
        }
        ci0_eig(num_step_constraints+k) = steer_limit_ + delta_curr;
        ci0_eig(num_step_constraints+Nc+k) = steer_limit_ - delta_curr;
    }

    // 7. Solve QP
    qp::Matrix<double> qpG = toQPMatrix(H);
    qp::Vector<double> qpg0 = toQPVector(f);
    qp::Matrix<double> qpCI = toQPMatrix(CI_eig);
    qp::Vector<double> qpci0 = toQPVector(ci0_eig);
    qp::Matrix<double> qpCE(nv, 0); 
    qp::Vector<double> qpce0(0);
    qp::Vector<double> qp_result(nv);

    // Timer for QP
    const auto t_qp_start = std::chrono::steady_clock::now();
    double qp_status = qp::solve_quadprog(qpG, qpg0, qpCE, qpce0, qpCI, qpci0, qp_result);
    const double qp_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t_qp_start).count();
    
    if (qp_time_pub_) {
      std_msgs::msg::Float64 msg; msg.data = qp_ms; qp_time_pub_->publish(msg);
    }

    // 8. Output Calculation
    double delta_cmd = delta_curr;
    double delta_inc = 0.0;

    if (std::isfinite(qp_status)) {
        delta_inc = qp_result[0];
        delta_cmd = delta_curr + delta_inc;
    } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "QP Failed");
    }

    // Final Saturation
    const double limit = std::abs(steer_limit_);
    delta_cmd = std::clamp(delta_cmd, -limit, limit);
    
    // Save state for next step
    delta_prev_ = delta_cmd;

    // 9. Visualization & Publish
    // Predicted Path Visualization (forward simulate simple kinematic bicycle)
    if (std::isfinite(qp_status)) {
      Eigen::VectorXd u_stack = Eigen::VectorXd::Zero(nu * Nc);
      for (int i = 0; i < nv; ++i) u_stack(i) = qp_result[i];
      
      nav_msgs::msg::Path pred_path;
      pred_path.header.stamp = now();
      pred_path.header.frame_id = path_frame_;

      double x_viz = pose.position.x;
      double y_viz = pose.position.y;
      double psi_viz = heading;
      double delta_viz = delta_curr;
      const double v_viz = std::max(speed, 0.1);

      for (int i = 0; i < Np; ++i) {
        if (i < Nc && i < u_stack.size()) {
          delta_viz += u_stack(i);
        }
        delta_viz = std::clamp(delta_viz, -steer_limit_, steer_limit_);

        // Propagate kinematic bicycle
        x_viz += v_viz * std::cos(psi_viz) * Ts;
        y_viz += v_viz * std::sin(psi_viz) * Ts;
        psi_viz += (v_viz / wheelbase_) * std::tan(delta_viz) * Ts;

        geometry_msgs::msg::PoseStamped p;
        p.header = pred_path.header;
        p.pose.position.x = x_viz;
        p.pose.position.y = y_viz;
        tf2::Quaternion q;
        q.setRPY(0, 0, psi_viz);
        p.pose.orientation = tf2::toMsg(q);
        pred_path.poses.push_back(p);
      }
      pred_path_pub_->publish(pred_path);
    }

    // Debug Markers
    if(debug_marker_pub_->get_subscription_count() > 0) {
      visualization_msgs::msg::MarkerArray markers;
      visualization_msgs::msg::Marker text;
      text.header.frame_id = path_frame_;
      text.header.stamp = now();
      text.ns = "mpc_info";
      text.id = 1;
      text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text.action = visualization_msgs::msg::Marker::ADD;
      text.pose.position = pose.position;
      text.pose.position.z += 1.0;
      text.scale.z = 0.3;
      text.color.a = 1.0; text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0;
      std::ostringstream ss;
      ss.setf(std::ios::fixed); ss.precision(3);
      ss << "Cmd: " << delta_cmd << "\nInc: " << delta_inc << "\nEy: " << e_y << "\nEpsi: " << e_psi;
      text.text = ss.str();
      markers.markers.push_back(text);
      debug_marker_pub_->publish(markers);
    }

    ackermann_msgs::msg::AckermannDriveStamped cmd;
    cmd.header.stamp = now();
    cmd.header.frame_id = path_frame_;
    cmd.drive.steering_angle = delta_cmd;
    cmd.drive.speed = std::isfinite(qp_status) ? target_speed_ : 0.0;
    cmd_pub_->publish(cmd);

    // Monitoring
    std_msgs::msg::Float64 yaw_rate_msg; yaw_rate_msg.data = twist.angular.z; yaw_rate_pub_->publish(yaw_rate_msg);
    std_msgs::msg::Float64 steer_est_msg; steer_est_msg.data = delta_est_; steering_est_pub_->publish(steer_est_msg);
    std_msgs::msg::Float64 steer_cmd_msg; steer_cmd_msg.data = delta_cmd; steering_cmd_pub_->publish(steer_cmd_msg);

    publish_timer_ms(t_cb_start);
  }

  // Members
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr cmd_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pred_path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_marker_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr lateral_error_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr heading_error_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr qp_time_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr yaw_rate_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr steering_cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr steering_est_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr timer_ms_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  nav_msgs::msg::Path latest_path_;
  nav_msgs::msg::Odometry latest_odom_;
  bool path_ready_{false};
  bool odom_ready_{false};

  int horizon_;
  int control_horizon_;
  double dt_;
  double wheelbase_;
  double steer_limit_;
  double q_lat_;
  double q_head_;
  double r_steer_;
  double dmax_deg_;
  std::string path_frame_;
  std::string cmd_topic_;
  std::string path_topic_;
  std::string odom_topic_;
  
  double delta_prev_{0.0};
  double delta_est_{0.0};
  double target_speed_;
  double dev_dist_thresh_{0.0};
  double dev_heading_thresh_{0.0};#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/float64.hpp>

#include <QuadProg++.hh>
#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

namespace
{
constexpr double kEpsilon = 1.0e-9;

double normalizeAngle(double a)
{
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

double yawFromQuat(const geometry_msgs::msg::Quaternion & q)
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

double yawFromPath(const nav_msgs::msg::Path & path, const size_t idx)
{
  if (path.poses.empty()) return 0.0;
  if (idx + 1 < path.poses.size()) {
    const auto & p0 = path.poses.at(idx).pose.position;
    const auto & p1 = path.poses.at(idx + 1).pose.position;
    return std::atan2(p1.y - p0.y, p1.x - p0.x);
  }
  if (idx > 0) {
    const auto & p0 = path.poses.at(idx - 1).pose.position;
    const auto & p1 = path.poses.at(idx).pose.position;
    return std::atan2(p1.y - p0.y, p1.x - p0.x);
  }
  return yawFromQuat(path.poses.front().pose.orientation);
}

size_t findNearestIdx(const nav_msgs::msg::Path & path, const geometry_msgs::msg::Pose & pose)
{
  double best_dist = std::numeric_limits<double>::max();
  size_t best_idx = 0;
  for (size_t i = 0; i < path.poses.size(); ++i) {
    const auto & p = path.poses.at(i).pose.position;
    const double dx = pose.position.x - p.x;
    const double dy = pose.position.y - p.y;
    const double dist = dx * dx + dy * dy;
    if (dist < best_dist) {
      best_dist = dist;
      best_idx = i;
    }
  }
  return best_idx;
}

namespace qp = quadprogpp;

qp::Matrix<double> toQPMatrix(const Eigen::MatrixXd & m)
{
  qp::Matrix<double> qp_m(m.rows(), m.cols());
  for (int r = 0; r < m.rows(); ++r) {
    for (int c = 0; c < m.cols(); ++c) {
      qp_m[r][c] = m(r, c);
    }
  }
  return qp_m;
}

qp::Vector<double> toQPVector(const Eigen::VectorXd & v)
{
  qp::Vector<double> qp_v(v.size());
  for (int i = 0; i < v.size(); ++i) {
    qp_v[i] = v(i);
  }
  return qp_v;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> buildPredictionMatrices(
  const Eigen::MatrixXd & Ad, const Eigen::MatrixXd & Bd, const int Np, const int Nc)
{
  const int n = Ad.rows();
  const int m = Bd.cols();

  Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(n * Np, n);
  Eigen::MatrixXd Gamma = Eigen::MatrixXd::Zero(n * Np, m * Nc);

  Eigen::MatrixXd A_power = Eigen::MatrixXd::Identity(n, n);
  for (int i = 0; i < Np; ++i) {
    A_power = Ad * A_power;
    Phi.block(i * n, 0, n, n) = A_power;
  }

  for (int r = 0; r < Np; ++r) {
    for (int c = 0; c < Nc; ++c) {
      if (r >= c) {
        Eigen::MatrixXd A_pc = Eigen::MatrixXd::Identity(n, n);
        for (int k = 0; k < (r - c); ++k) {
          A_pc = Ad * A_pc;
        }
        Gamma.block(r * n, c * m, n, m) = A_pc * Bd;
      }
    }
  }
  return {Phi, Gamma};
}

}  // namespace

class MPCLateralNode : public rclcpp::Node
{
public:
  MPCLateralNode()
  : Node("mpc_lateral"),
    horizon_(declare_parameter<int>("Np")),
    control_horizon_(declare_parameter<int>("Nc")),
    dt_(declare_parameter<double>("Ts")),
    wheelbase_(declare_parameter<double>("wheelbase")),
    steer_limit_(declare_parameter<double>("steer_limit")),
    q_lat_(declare_parameter<double>("Qy")),
    q_head_(declare_parameter<double>("Qpsi")),
    r_steer_(declare_parameter<double>("Rdelta")),
    dmax_deg_(declare_parameter<double>("Dmax_deg")),
    path_frame_(declare_parameter<std::string>("path_frame")),
    cmd_topic_(declare_parameter<std::string>("command_topic")),
    path_topic_(declare_parameter<std::string>("path_topic")),
    odom_topic_(declare_parameter<std::string>("odom_topic")),
    target_speed_(declare_parameter<double>("target_speed")),
    dev_dist_thresh_(declare_parameter<double>("deviation_distance_thresh")),
    dev_heading_thresh_(declare_parameter<double>("deviation_heading_thresh_rad")),
    // [New Parameters]
    curvature_lookahead_dist_(declare_parameter<double>("curvature_lookahead_dist")),
    ff_gain_(declare_parameter<double>("ff_gain")),
    use_odom_steer_est_(declare_parameter<bool>("use_odom_steer_est")),
    steer_est_v_thresh_(declare_parameter<double>("steer_est_v_thresh")),
    steer_est_lpf_alpha_(declare_parameter<double>("steer_est_lpf_alpha"))
  {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    rclcpp::QoS path_qos(rclcpp::KeepLast(10));
    path_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
    path_qos.durability(rclcpp::DurabilityPolicy::Volatile);
    path_sub_ = create_subscription<nav_msgs::msg::Path>(
      path_topic_, path_qos, std::bind(&MPCLateralNode::onPath, this, std::placeholders::_1));

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::QoS(20), std::bind(&MPCLateralNode::onOdom, this, std::placeholders::_1));

    cmd_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(cmd_topic_, 10);
    pred_path_pub_ = create_publisher<nav_msgs::msg::Path>("mpc/predicted_path", 1);
    debug_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("mpc/debug_markers", 1);
    lateral_error_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/lateral_error", 10);
    heading_error_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/heading_error", 10);
    qp_time_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/qp_time_ms", 10);
    timer_ms_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/on_timer_ms", 10);
    yaw_rate_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/yaw_rate", 10);
    steering_cmd_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/steering_cmd", 10);
    steering_est_pub_ = create_publisher<std_msgs::msg::Float64>("mpc/steering_est", 10);

    timer_ = create_wall_timer(
      std::chrono::duration<double>(dt_), std::bind(&MPCLateralNode::onTimer, this));

    RCLCPP_INFO(get_logger(), "MPC (Augmented State: Rate Control) Ready.");
  }

private:
  void onPath(const nav_msgs::msg::Path::SharedPtr msg)
  {
    if (msg->poses.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Received empty path");
      return;
    }
    latest_path_ = *msg;
    path_ready_ = true;
  }

  void onOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    latest_odom_ = *msg;
    odom_ready_ = true;
  }

  double computeCurvature(const nav_msgs::msg::Path & path, size_t current_idx, double lookahead_dist)
  {
    if (path.poses.size() <= current_idx + 1) return 0.0;

    double accumulated_dist = 0.0;
    size_t target_idx = current_idx;

    for (size_t i = current_idx + 1; i < path.poses.size(); ++i) {
      const auto & prev_p = path.poses.at(i - 1).pose.position;
      const auto & curr_p = path.poses.at(i).pose.position;
      accumulated_dist += std::hypot(curr_p.x - prev_p.x, curr_p.y - prev_p.y);
      target_idx = i;
      if (accumulated_dist >= lookahead_dist) break;
    }

    if (target_idx == current_idx) return 0.0;

    const double psi_curr = yawFromPath(path, current_idx);
    const double psi_target = yawFromPath(path, target_idx);
    const double dpsi = normalizeAngle(psi_target - psi_curr);
    
    if (accumulated_dist < 0.1) return 0.0;

    return dpsi / accumulated_dist; 
  }

  void onTimer()
  {
    const auto t_cb_start = std::chrono::steady_clock::now();
    auto publish_timer_ms = [&](const std::chrono::steady_clock::time_point & start_tp) {
      if (!timer_ms_pub_) return;
      std_msgs::msg::Float64 msg;
      msg.data = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start_tp).count();
      timer_ms_pub_->publish(msg);
    };

    if (!path_ready_ || !odom_ready_) {
      publish_timer_ms(t_cb_start);
      return;
    }

    // 1. Pose & State Update
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header = latest_odom_.header;
    pose_stamped.pose = latest_odom_.pose.pose;

    if (pose_stamped.header.frame_id == "world" && path_frame_ == "map") {
      pose_stamped.header.frame_id = "map";
    }
    if (pose_stamped.header.frame_id != path_frame_) {
      if (tf_buffer_->canTransform(path_frame_, pose_stamped.header.frame_id, tf2::TimePointZero)) {
        try {
          pose_stamped = tf_buffer_->transform(
            pose_stamped, path_frame_, tf2::durationFromSec(0.05));
        } catch (const tf2::TransformException & ex) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF transform failed: %s", ex.what());
          return;
        }
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF unavailable");
        return;
      }
    }
    const auto & pose = pose_stamped.pose;
    const auto & twist = latest_odom_.twist.twist;

    const double vx = twist.linear.x;
    const double vy = twist.linear.y;
    const double speed = std::sqrt(vx * vx + vy * vy);

    const size_t nearest_idx = findNearestIdx(latest_path_, pose);
    const double ref_yaw = yawFromPath(latest_path_, nearest_idx);
    const double heading = yawFromQuat(pose.orientation);
    const double heading_err = normalizeAngle(heading - ref_yaw);
    const double dx = pose.position.x - latest_path_.poses.at(nearest_idx).pose.position.x;
    const double dy = pose.position.y - latest_path_.poses.at(nearest_idx).pose.position.y;
    const double pos_err = std::hypot(dx, dy);

    // Monitoring Publishers
    std_msgs::msg::Float64 lat_err_msg; lat_err_msg.data = pos_err; lateral_error_pub_->publish(lat_err_msg);
    std_msgs::msg::Float64 head_err_msg; head_err_msg.data = heading_err; heading_error_pub_->publish(head_err_msg);

    // 2. MPC Parameters & State Augmentation
    // [KEY CHANGE] State size is now 5: [x, y, psi, v, delta]
    const int nx = 5; 
    const int nu = 1;
    int Np = std::max(1, horizon_);
    int Nc = std::max(1, control_horizon_);
    Nc = std::min(Nc, Np);
    const double Ts = dt_;
    const double v_lin = std::max(speed, 0.1);

    // Steering Estimation (Sync logic)
    double delta_base = delta_prev_;
    if (use_odom_steer_est_) {
      const double yaw_rate = twist.angular.z;
      if (std::abs(speed) > steer_est_v_thresh_) {
        const double delta_meas = std::atan((wheelbase_ * yaw_rate) / std::max(std::abs(speed), steer_est_v_thresh_));
        delta_est_ = steer_est_lpf_alpha_ * delta_est_ + (1.0 - steer_est_lpf_alpha_) * delta_meas;
        delta_est_ = std::clamp(delta_est_, -steer_limit_, steer_limit_);
        delta_base = delta_est_;
        delta_prev_ = delta_est_; 
      }
    }

    // Kinematic Model Linearization (Standard 4x4)
    Eigen::MatrixXd A_kin(4, 4);
    A_kin << 1.0, 0.0, -v_lin * std::sin(heading) * Ts, std::cos(heading) * Ts,
             0.0, 1.0,  v_lin * std::cos(heading) * Ts, std::sin(heading) * Ts,
             0.0, 0.0, 1.0, (1.0 / wheelbase_) * std::tan(delta_base) * Ts, 
             0.0, 0.0, 0.0, 1.0; 

    // B_kin describes how 'delta' affects state (part of A in augmented model)
    const double sec_delta_sq = 1.0 / std::max(std::pow(std::cos(delta_base), 2), kEpsilon);
    Eigen::VectorXd B_kin(4);
    B_kin << 0.0, 0.0, (v_lin / wheelbase_) * sec_delta_sq * Ts, 0.0;

    // [KEY CHANGE] Augmented Matrix A (5x5)
    // State: [x, y, psi, v, delta]
    // delta_{k+1} = delta_k + u_k (integration)
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(nx, nx);
    A.block(0, 0, 4, 4) = A_kin;
    A.block(0, 4, 4, 1) = B_kin; 

    // [KEY CHANGE] Augmented Matrix B (5x1)
    // Input u is 'delta_delta' (steering rate)
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(nx, nu);
    B(4, 0) = 1.0; 

    Eigen::MatrixXd Ad = A;
    Eigen::MatrixXd Bd = B;

    // Current State Vector (5D)
    Eigen::VectorXd xk(nx);
    xk << pose.position.x, pose.position.y, heading, speed, delta_base;

    // 3. Reference Generation
    Eigen::VectorXd ref_stack = Eigen::VectorXd::Zero(Np * nx);
    
    // Dist calc
    const size_t path_size = latest_path_.poses.size();
    const size_t start_idx = nearest_idx;
    std::vector<double> cum_dist;
    cum_dist.reserve(path_size - start_idx);
    cum_dist.push_back(0.0);
    for (size_t j = start_idx + 1; j < path_size; ++j) {
      const auto & p0 = latest_path_.poses.at(j - 1).pose.position;
      const auto & p1 = latest_path_.poses.at(j).pose.position;
      const double ds = std::hypot(p1.x - p0.x, p1.y - p0.y);
      cum_dist.push_back(cum_dist.back() + ds);
    }

    for (int i = 0; i < Np; ++i) {
      const double required_dist = v_lin * Ts * i;
      size_t rel_idx = 0;
      while (rel_idx + 1 < cum_dist.size() && cum_dist[rel_idx] < required_dist) {
        ++rel_idx;
      }
      const size_t idx = std::min(start_idx + rel_idx, path_size - 1);
      const auto & ref_pose = latest_path_.poses.at(idx).pose;
      
      ref_stack(i * nx + 0) = ref_pose.position.x;
      ref_stack(i * nx + 1) = ref_pose.position.y;
      ref_stack(i * nx + 2) = yawFromPath(latest_path_, idx);
      ref_stack(i * nx + 3) = target_speed_;
      ref_stack(i * nx + 4) = 0.0; // Target steering angle is 0 (or FF, but 0 stabilizes)
    }

    // 4. QP Matrices
    const auto [Phi, Gamma] = buildPredictionMatrices(Ad, Bd, Np, Nc);

    // [KEY CHANGE] Weight Matrices for 5 States
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nx, nx);
    Q(0, 0) = std::max(q_lat_, 1e-4);
    Q(1, 1) = std::max(q_lat_, 1e-4);
    Q(2, 2) = std::max(q_head_, 1e-4);
    Q(3, 3) = 0.1; // speed error
    Q(4, 4) = 0.1; // absolute steering minimization (keep small)

    // R is now weight on STEERING RATE (Input u) -> Prevents Oscillation
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nu, nu) * std::max(r_steer_, 1.0);

    Eigen::MatrixXd Qbar = Eigen::MatrixXd::Zero(nx * Np, nx * Np);
    for (int i = 0; i < Np; ++i) Qbar.block(i * nx, i * nx, nx, nx) = Q;
    
    Eigen::MatrixXd Rbar = Eigen::MatrixXd::Zero(nu * Nc, nu * Nc);
    for (int i = 0; i < Nc; ++i) Rbar.block(i * nu, i * nu, nu, nu) = R;

    // FF (Curvature)
    Eigen::VectorXd delta_ff_vec = Eigen::VectorXd::Zero(Np);
    for (int i = 0; i < Np; ++i) {
      const double required_dist_k = v_lin * Ts * i;
      size_t rel_idx_k = 0;
      while (rel_idx_k + 1 < cum_dist.size() && cum_dist[rel_idx_k] < required_dist_k) {
        ++rel_idx_k;
      }
      const size_t idx_k = std::min(start_idx + rel_idx_k, path_size - 1);
      const double k_i = computeCurvature(latest_path_, idx_k, curvature_lookahead_dist_);
      delta_ff_vec(i) = ff_gain_ * std::atan(k_i * wheelbase_);
    }

    const Eigen::VectorXd X_stack_nom = Phi * xk;
    Eigen::MatrixXd H = Gamma.transpose() * Qbar * Gamma + Rbar;
    H = 0.5 * (H + H.transpose());
    H += 1e-8 * Eigen::MatrixXd::Identity(H.rows(), H.cols());
    
    const Eigen::VectorXd f = Gamma.transpose() * Qbar * (X_stack_nom - ref_stack);

    // 6. Constraints
    const int nv = nu * Nc;
    // dmax_deg_ controls the RATE (e.g. 1 deg per step = 20 deg/s)
    double u_max = dmax_deg_ * M_PI / 180.0; 
    
    Eigen::VectorXd lb = Eigen::VectorXd::Constant(nv, -u_max);
    Eigen::VectorXd ub = Eigen::VectorXd::Constant(nv, u_max);

    // Constraints Matrix Construction
    // 1. Step constraints (Rate limits): -u_max <= u <= u_max
    // 2. Absolute constraints (Steer limits): -limit <= delta_base + sum(u) <= limit
    const int num_step_constraints = 2 * nv;
    const int num_abs_constraints = 2 * Nc;
    const int total_constraints = num_step_constraints + num_abs_constraints;
    Eigen::MatrixXd CI_eig = Eigen::MatrixXd::Zero(nv, total_constraints);
    Eigen::VectorXd ci0_eig = Eigen::VectorXd::Zero(total_constraints);

    // Rate Constraints
    for (int i = 0; i < nv; ++i) {
      CI_eig(i, i) = 1.0;            
      ci0_eig(i) = -lb(i);
      CI_eig(i, nv + i) = -1.0;      
      ci0_eig(nv + i) = ub(i);
    }
    // Absolute Constraints (Accumulated u)
    for (int k = 0; k < Nc; ++k) {
      for (int j = 0; j <= k; ++j) {
        CI_eig(j, num_step_constraints + k) = 1.0;            
        CI_eig(j, num_step_constraints + Nc + k) = -1.0;      
      }
      ci0_eig(num_step_constraints + k) = steer_limit_ + delta_base; // Lower bound logic for QP
      ci0_eig(num_step_constraints + Nc + k) = steer_limit_ - delta_base;
    }

    // 7. Solve QP
    qp::Matrix<double> qpG = toQPMatrix(H);
    qp::Vector<double> qpg0 = toQPVector(f);
    qp::Matrix<double> qpCE(nv, 0);
    qp::Vector<double> qpce0(0);
    qp::Matrix<double> qpCI = toQPMatrix(CI_eig);
    qp::Vector<double> qpci0 = toQPVector(ci0_eig);
    qp::Vector<double> qp_result(nv);

    const auto t_qp_start = std::chrono::steady_clock::now();
    const double qp_status = qp::solve_quadprog(qpG, qpg0, qpCE, qpce0, qpCI, qpci0, qp_result);
    const double qp_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t_qp_start).count();
    
    if (qp_time_pub_) {
      std_msgs::msg::Float64 msg; msg.data = qp_ms; qp_time_pub_->publish(msg);
    }

    // 8. Command Calculation
    double delta_cmd = delta_prev_; 
    double delta_inc = 0.0;
    double current_ff_val = delta_ff_vec(0); 

    if (std::isfinite(qp_status)) {
      delta_inc = qp_result[0]; // Optimal Change
      
      // Final Output = Current State + Optimal Change + FeedForward
      delta_cmd = delta_base + delta_inc + current_ff_val;
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "QP Failed");
    }

    const double delta_limit = std::abs(steer_limit_);
    delta_cmd = std::clamp(delta_cmd, -delta_limit, delta_limit);
    
    // Update previous state for next loop sync
    delta_prev_ = delta_cmd; 

    // 9. Visualization
    if (pred_path_pub_->get_subscription_count() > 0 && std::isfinite(qp_status)) {
      Eigen::VectorXd u_stack = Eigen::VectorXd::Zero(nu * Nc);
      for (int i = 0; i < nv; ++i) u_stack(i) = qp_result[i];
      
      // Reconstruct state trajectory from augmented model
      const Eigen::VectorXd x_pred_stack = X_stack_nom + Gamma * u_stack;

      nav_msgs::msg::Path pred_path;
      pred_path.header.stamp = now();
      pred_path.header.frame_id = path_frame_;
      
      for (int i = 0; i < Np; ++i) {
        geometry_msgs::msg::PoseStamped p;
        p.header = pred_path.header;
        // x_pred_stack has structure: [x0, y0, psi0, v0, d0, x1, y1, ...]
        const auto x_seg = x_pred_stack.segment(nx * i, nx);
        p.pose.position.x = x_seg(0);
        p.pose.position.y = x_seg(1);
        tf2::Quaternion q;
        q.setRPY(0, 0, x_seg(2));
        p.pose.orientation = tf2::toMsg(q);
        pred_path.poses.push_back(p);
      }
      pred_path_pub_->publish(pred_path);

      // Debug Markers
      visualization_msgs::msg::MarkerArray markers;
      visualization_msgs::msg::Marker text;
      text.header = pred_path.header;
      text.ns = "mpc_info";
      text.id = 1;
      text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text.action = visualization_msgs::msg::Marker::ADD;
      text.pose.position = pose.position;
      text.pose.position.z += 1.0;
      text.scale.z = 0.3;
      text.color.a = 1.0; text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0;
      std::ostringstream ss;
      ss.setf(std::ios::fixed); ss.precision(3);
      ss << "Cmd: " << delta_cmd << "\nInc: " << delta_inc << "\nErr: " << pos_err;
      text.text = ss.str();
      markers.markers.push_back(text);
      debug_marker_pub_->publish(markers);
    }

    ackermann_msgs::msg::AckermannDriveStamped cmd;
    cmd.header.stamp = now();
    cmd.header.frame_id = path_frame_;
    cmd.drive.steering_angle = delta_cmd;
    cmd.drive.speed = std::isfinite(qp_status) ? target_speed_ : 0.0;
    cmd_pub_->publish(cmd);

    // Monitoring
    std_msgs::msg::Float64 yaw_rate_msg; yaw_rate_msg.data = twist.angular.z; yaw_rate_pub_->publish(yaw_rate_msg);
    std_msgs::msg::Float64 steer_est_msg; steer_est_msg.data = delta_est_; steering_est_pub_->publish(steer_est_msg);

    publish_timer_ms(t_cb_start);
  }

  // Members
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr cmd_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pred_path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_marker_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr lateral_error_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr heading_error_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr qp_time_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr yaw_rate_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr steering_cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr steering_est_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr timer_ms_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  nav_msgs::msg::Path latest_path_;
  nav_msgs::msg::Odometry latest_odom_;
  bool path_ready_{false};
  bool odom_ready_{false};

  int horizon_;
  int control_horizon_;
  double dt_;
  double wheelbase_;
  double steer_limit_;
  double q_lat_;
  double q_head_;
  double r_steer_;
  double dmax_deg_;
  std::string path_frame_;
  std::string cmd_topic_;
  std::string path_topic_;
  std::string odom_topic_;
  
  double delta_prev_{0.0};
  double delta_est_{0.0};
  double target_speed_;
  double dev_dist_thresh_{0.0};
  double dev_heading_thresh_{0.0};

  double curvature_lookahead_dist_;
  double ff_gain_;
  bool use_odom_steer_est_;
  double steer_est_v_thresh_;
  double steer_est_lpf_alpha_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MPCLateralNode>());
  rclcpp::shutdown();
  return 0;
}


  double curvature_lookahead_dist_;
  double ff_gain_;
  bool use_odom_steer_est_;
  double steer_est_v_thresh_;
  double steer_est_lpf_alpha_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MPCLateralNode>());
  rclcpp::shutdown();
  return 0;
}
