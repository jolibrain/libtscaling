
#ifdef LBFGSPP
#include <LBFGS.h>
#else
#include <lbfgs.h>
#endif
#include <vector>

#include <string>

class TempScaler
{
  // cross entropyloss is negative log likelihood applied on logits
  // or said differently : first softmax then negative log likelihood
 private:

#ifdef LBFGSPP
  Eigen::VectorXd t_;
  LBFGSpp::LBFGSParam<double>* params_;
  LBFGSpp::LBFGSSolver<double>* solver_;
#else
  lbfgsfloatval_t t_;
  lbfgs_parameter_t* params_;
#endif

  const std::vector<std::vector<double> >* logits_;
  const std::vector<int>* labels_;


 public:
  TempScaler(const std::vector<std::vector<double> >&logits,
               const std::vector<int>& labels);
  TempScaler();

  void setData(const std::vector<std::vector<double> >& logits, const std::vector<int>& labels);

  void getPredConfBatch(std::vector<std::vector<double> > logitbatch, std::vector<int>& preds, std::vector<double>&confs);
  void getPredConf(std::vector<double>& logits, int &pred, double&conf);
  void getPredConfWithTemperature(std::vector<double>& logits, double temperature, int &pred, double&conf);


  double getTemperature();
  double calibrate();


 private:

  double crossEntropyLossWithTemperature(const double& x, double& gx);



#ifdef LBFGSPP
 public:
  double operator()(const Eigen::VectorXd &x , Eigen::VectorXd& grad);
#else
 private:
  static lbfgsfloatval_t _evaluate(
                                   void *instance,
                                   const lbfgsfloatval_t *x,
                                   lbfgsfloatval_t *g,
                                   const int n,
                                   const lbfgsfloatval_t step
                                   );
  lbfgsfloatval_t evaluate(
                           const lbfgsfloatval_t *x,
                           lbfgsfloatval_t *g,
                           const int n,
                           const lbfgsfloatval_t step
                           );
#endif

};


class CalibrationError{
 private:
  std::vector<double>* confs_; // confidence of prediction i
  std::vector<int>* predictions_;
  std::vector<int>* targets_;
  std::vector<std::vector<int> > bins_;
  std::vector<double> bin_acc_cache_;
  std::vector<double> bin_conf_cache_;
  int nbins_;
  bool cached_;
 public:
  CalibrationError(std::vector<double>& confidences, std::vector<int>& predictions, std::vector<int>& targets, int nbins);
  CalibrationError(int nbins);
  void setData(std::vector<double>& confidences, std::vector<int>& predictions,
               std::vector<int>& targets);
  double ECE();
  double bin_acc(int bi);
  double bin_conf(int bi);
  double MCE();
  void display();
  void accuracies(std::vector<double>& accuracies);
  void percents(std::vector<double>& percents);
  void edges(std::vector<double>& edges);
  void to_py(std::string fname);

 private:
  void fill_bins();
  void clear_bins();

};
