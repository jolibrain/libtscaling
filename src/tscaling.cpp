#include <stdio.h>
#include <math.h>
#include <iostream>


#include "tscaling.h"

TempScaler::TempScaler(const std::vector<std::vector<double> >&logits,
                           const std::vector<int>& labels) :
  logits_(&logits), labels_(&labels)
{
#ifdef LBFGSPP
  t_ =  Eigen::VectorXd::Ones(1);
  params_ = new LBFGSpp::LBFGSParam<double>();
#else
  t_ = (lbfgsfloatval_t) 1.0;
  params_ = new lbfgs_parameter_t();
  lbfgs_parameter_init(params_);
#endif

  params_->epsilon = 1E-6; // default anyway
  params_->max_iterations = 100; // set a limit in case of problem

#ifdef LBFGSPP
  solver_ = new LBFGSpp::LBFGSSolver<double>(*params_);
#endif
}

TempScaler::TempScaler() : logits_(NULL), labels_(NULL)
{
#ifdef LBFGSPP
    t_ =  Eigen::VectorXd::Ones(1);
    params_ = new LBFGSpp::LBFGSParam<double>();
#else
    t_ = (lbfgsfloatval_t) 1.0;
    params_ = new lbfgs_parameter_t();
    lbfgs_parameter_init(params_);
#endif

    params_->epsilon = 1E-6; // default anyway
    params_->max_iterations = 100; // set a limit in case of problem


#ifdef LBFGSPP
    solver_ = new LBFGSpp::LBFGSSolver<double>(*params_);
#endif
  }

void TempScaler::setData(const std::vector<std::vector<double> >& logits, const std::vector<int>& labels)
{
  logits_ = &logits;
  labels_ = &labels;
}



double TempScaler::getTemperature()
{
#ifdef LBFGSPP
  return t_(0);
#else
  return (double)t_;
#endif
}

double TempScaler::calibrate()
{
  if (logits_ == NULL || labels_ == NULL)
    {
      std::cerr << "no labels/logits given to calibration";
      return -1;
    }
#ifdef LBFGSPP
  double fx;
  solver_->minimize(*this, t_, fx);
#else
  lbfgsfloatval_t fx;
  lbfgs(1, &t_, &fx, _evaluate, NULL, this, params_);
#endif
  return getTemperature();
}


#ifdef LBFGSPP
double TempScaler::operator()(const Eigen::VectorXd &x , Eigen::VectorXd& grad)
// x is the temperature
// grad the value of the gradient at x
// return the value of the function
{

  double gx;
  double fx = crossEntropyLossWithTemperature(x(0),  gx);
  grad(0) = gx;
  return fx;
}
#endif


double TempScaler::crossEntropyLossWithTemperature(const double& x, double& gx)
{
  // return  value at x
  // gx is derivation of f wrt t, evaluated at x
  // now the computation of crossentropyloss and its derivative, averaged over batch size

  double temperature = x;
  int nlogits = (*logits_)[0].size();
  double sum_fx = 0;
  double sum_gfx = 0;

  for (int bi = 0; bi < logits_->size(); ++bi)
    {
      const std::vector<double>& logits = (*logits_)[bi];
      const int label = (*labels_)[bi];

      double sum1 = 0;
      double sum2 = 0;
      for (int j = 0; j<nlogits; ++j)
        {
          sum1 += exp(logits[j]/temperature);
          sum2 += exp(logits[j]/temperature) * logits[j] / temperature / temperature;
        }

      sum_fx += log(sum1) - logits[label]/temperature;
      sum_gfx += logits[label]/temperature/temperature - sum2/sum1;


    }

  gx = sum_gfx / (double)logits_->size();
  return sum_fx / (double)logits_->size();
}

#ifndef LBFGSPP
lbfgsfloatval_t TempScaler::_evaluate(
                                        void *instance,
                                        const lbfgsfloatval_t *x,
                                        lbfgsfloatval_t *g,
                                        const int n,
                                        const lbfgsfloatval_t step
                                        )
{
  return reinterpret_cast<TempScaler*>(instance)->evaluate(x, g, n, step);
}

lbfgsfloatval_t TempScaler::evaluate(
                                       const lbfgsfloatval_t *x,
                                       lbfgsfloatval_t *g,
                                       const int n,
                                       const lbfgsfloatval_t step
                                       )
  {


    double fx = crossEntropyLossWithTemperature(*x, *g);
    return  (lbfgsfloatval_t)fx;
  }
#endif





int main(int argc, char **argv)
{
  TempScaler celwt;
  std::vector<int> labels {0, 3 , 5 ,3};
  std::vector<std::vector<double> > logits{
    {50, 10, 3, 25, 10, 2},
      {-10, 20, 10, 20, 20, 1},
        {10, 50, 5, 30, 30, 40},
          {1, 20, 5, 15, 40, 4}};

  celwt.setData(logits,labels);

  std::cout<< "final temperature : " << celwt.calibrate() << std::endl;
}
