
#ifdef LBFGSPP
#include <LBFGS.h>
#else
#include <lbfgs.h>
#endif
#include <vector>



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
