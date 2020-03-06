#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;
  
  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;
  
  // State dimension
  n_x_ = 5;
  
  // Augmented state dimension
  n_aug_ = 7;
  
  // Number of sigma points
  n_sig_ = 2 * n_aug_ + 1;
  
  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  
  // predicted sigma points matrix
  Xsig_pred_ = Eigen::MatrixXd::Zero(n_x_, n_sig_);
  
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.7;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  
  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;
  
  // Weights of sigma points
  weights_ = Eigen::VectorXd::Zero(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < n_sig_; i++) {
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }
  
  // The current NIS for radar
  NIS_radar_ = 0.0;
  
  // The current NIS for laser
  NIS_laser_ = 0.0;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if(!is_initialized_){
    if(meas_package.sensor_type_ == MeasurementPackage::LASER){
      float px = meas_package.raw_measurements_[0];
      float py = meas_package.raw_measurements_[1];
     
      x_ << px, py, 0, 0, 0;
      P_ << std_laspx_*std_laspx_, 0, 0, 0, 0,
            0, std_laspy_*std_laspy_, 0, 0, 0,
            0, 0, 1, 0, 0, 
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
    }
    else if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rhod = meas_package.raw_measurements_[2];
      
      float px = rho * std::cos(phi);
      float py = rho * std::sin(phi);
      float vx = rhod * std::cos(phi);
      float vy = rhod * std::sin(phi);
      float v = sqrt(vx*vx + vy*vy);
      
      x_ << px, py, v, 0, 0;
      P_ <<  std_radr_*std_radr_, 0, 0, 0, 0,
              0, std_radr_*std_radr_, 0, 0, 0,
              0, 0, std_radrd_*std_radrd_, 0, 0,
              0, 0, 0, std_radphi_*std_radphi_, 0,
              0, 0, 0, 0, 1;
    }
    
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    return;
  }
  else{
    // Define dt to be the change in time expressed in seconds
    float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;
    
    //     Prediction
    UKF::Prediction(dt);
    
    //     Update
    if(meas_package.sensor_type_ == MeasurementPackage::LASER){
      // Laser updates
      UKF::UpdateLidar(meas_package);
    }
    else if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      // RADAR ubdates
      UKF::UpdateRadar(meas_package);
    }
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  
  //   Step One: Generate Sigma Points
  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  
  // create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  
  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;
  
  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }
  
  //   Step Two: Predict Sigma Points
  // predict sigma points
  for (int i = 0; i < n_sig_; ++i) {
    // extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  
  //   Step Three: Predict Mean and Covariance
  //create vector for predicted state
  VectorXd x = VectorXd(n_x_);
  
  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);
  
  // predicted state mean
  x.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {  // iterate over sigma points
    x = x + weights_(i) * Xsig_pred_.col(i);
  }

  // predicted state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {  // iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    // angle normalization
    while (x_diff(3)> M_PI) {
        x_diff(3)-=2.*M_PI;
    }
    while (x_diff(3)<-M_PI){
        x_diff(3)+=2.*M_PI;
    } 

    P = P + weights_(i) * x_diff * x_diff.transpose() ;
  }
  x_ = x;
  P_ = P;
}


void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  
  //   Step One: convert predicted sigma points into measurement space
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_zl, n_sig_);
  
  // convert predicted sigma points into measurement space to create the measurement vector
  for (int i = 0; i < n_sig_; ++i){  // 2n+1 simga points
    // measurement model
    Zsig(0,i) = Xsig_pred_(0,i);                  
    Zsig(1,i) = Xsig_pred_(1,i);                             
  }
  
  //   Step Two: calculate the mean and covariance of the measurement
  // Predicted Measurement Mean
  VectorXd z_pred = VectorXd(n_zl);
  
  // Predicted Covariance
  MatrixXd S = MatrixXd(n_zl, n_zl);
  
  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < n_sig_; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI){ 
      z_diff(1)-=2.*M_PI;
    }
    while (z_diff(1)<-M_PI){
      z_diff(1)+=2.*M_PI;
    }

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  
  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_zl, n_zl);
  R <<  std_laspx_*std_laspx_, 0,
        0, std_laspy_*std_laspy_;

  S = S + R;
  
  //   Step Three: update
  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_zl);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    // angle normalization
    while (z_diff(1)> M_PI){ 
      z_diff(1)-=2.*M_PI;
    }
    while (z_diff(1)<-M_PI){
      z_diff(1)+=2.*M_PI;
    }

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    // angle normalization
    while (x_diff(3)> M_PI){
      x_diff(3)-=2.*M_PI;
    }
    while (x_diff(3)<-M_PI){
      x_diff(3)+=2.*M_PI;
    }

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // lidar measurement
  VectorXd z = meas_package.raw_measurements_;
  
  // residual
  VectorXd z_diff = z - z_pred;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  //calculate NIS
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  
  //   Step One: convert predicted sigma points into measurement space
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_zr, n_sig_);
  
  // convert predicted sigma points into measurement space to create the measurement vector
  for (int i = 0; i < n_sig_; ++i){  // 2n+1 simga points
    // measurement model
    double px = Xsig_pred_(0,i);                  
    double py = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double psi = Xsig_pred_(3,i);
    
    // measurement model
    Zsig(0,i) = sqrt(px*px + py*py);// r
    Zsig(1,i) = atan2(py, px);// phi
    Zsig(2,i) = (px*cos(psi)*v + py*sin(psi)*v) / sqrt(px*px + py*py);// r_dot
  }
  
  //   Step Two: calculate the mean and covariance of the measurement
  // Predicted Measurement Mean
  VectorXd z_pred = VectorXd(n_zr);
  
  // Predicted Covariance
  MatrixXd S = MatrixXd(n_zr,n_zr);
  
  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < n_sig_; ++i) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI){ 
      z_diff(1)-=2.*M_PI;
    }
    while (z_diff(1)<-M_PI){
      z_diff(1)+=2.*M_PI;
    }

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  
  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_zr, n_zr);
  R <<  std_radr_*std_radr_, 0, 0,
        0, std_radphi_*std_radphi_, 0,
        0, 0, std_radrd_*std_radrd_;

  S = S + R;
  
  //   Step Three: update
  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_zr);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    // angle normalization
    while (z_diff(1)> M_PI){ 
      z_diff(1)-=2.*M_PI;
    }
    while (z_diff(1)<-M_PI){
      z_diff(1)+=2.*M_PI;
    }

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    // angle normalization
    while (x_diff(3)> M_PI){
      x_diff(3)-=2.*M_PI;
    }
    while (x_diff(3)<-M_PI){
      x_diff(3)+=2.*M_PI;
    }

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // radar measurement
  VectorXd z = meas_package.raw_measurements_;
  
  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization
  while (z_diff(1)> M_PI){ 
    z_diff(1)-=2.*M_PI;
  }
  while (z_diff(1)<-M_PI){
    z_diff(1)+=2.*M_PI;
  }

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  //calculate NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}/**/