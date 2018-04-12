#include <Eigen/Core>
#include <iostream>
#include <LBFGS.h>
#include <vector>

using Eigen::VectorXi;
using Eigen::VectorXd;
using namespace LBFGSpp;


class CrossEntropyLossWithTemperature
// cross entropyloss is negative log likelihood applied on logits
// or said differently : first softmax then negative log likelihood
{
private:
  const std::vector<std::vector<double> >* logits_;
  const std::vector<int>* labels_;

public:
  CrossEntropyLossWithTemperature(const std::vector<std::vector<double> >& logits,
                                  const std::vector<int>& labels) :
    logits_(&logits), labels_(&labels) {}

  CrossEntropyLossWithTemperature() :  logits_(NULL), labels_(NULL) {}

  void setData(const std::vector<std::vector<double> >& logits,
               const std::vector<int>& labels)
  {
    logits_ = &logits;
    labels_ = &labels;
  }

  double operator()(const VectorXd &x , VectorXd& grad)
  // x is the temperature
  // grad the value of the gradient at x
  // return the value of the function
  {

    double temperature = x(0);
    double sum_fx = 0;
    double sum_gfx = 0;

    for (int bi = 0; bi < labels_->size(); ++bi)
      {
        const std::vector<double>& logits = logits_->at(bi);
        const int label = labels_->at(bi);

        double sum1 = 0;
        double sum2 = 0;
        for (int j = 0; j<logits.size(); ++j)
          {
            sum1 += exp(logits[j]/temperature);
            sum2 += exp(logits[j]/temperature) * logits[j] / temperature / temperature;
          }

        sum_fx += log(sum1) - logits[label]/temperature;
        sum_gfx += logits[label]/temperature/temperature - sum2/sum1;
      }


    //std::cout << "x= " << temperature<< "    fx= " << sum_fx/ double (labels_->size()) << "    gfx = " << sum_gfx / double (labels_->size()) << std::endl;
    grad(0) = sum_gfx / double (labels_->size());
    return sum_fx / double (labels_->size());

  }
};


int main()
{
  // const int n = 10;
  // Set up parameters
  LBFGSParam<double> param;
  param.epsilon = 1e-6;
  param.max_iterations = 100;

  // Create solver and function object
  LBFGSSolver<double> solver(param);
  CrossEntropyLossWithTemperature celwt;

  // example with 6 classes, batch_size = 4
  std::vector<int> labels {0, 3 , 5 ,3};
  std::vector<std::vector<double> > logits{
      {50, 10, 3, 25, 10, 2},
      {-10, 20, 10, 20, 20, 1},
      {10, 50, 5, 30, 30, 40},
        {1, 20, 5, 15, 40, 4}};

  celwt.setData(logits,labels);

  // Initial guess
  VectorXd x = VectorXd::Ones(1);
  // x will be overwritten to be the best point found
  double fx;
  int niter = solver.minimize(celwt, x, fx);

  //std::cout << niter << " iterations" << std::endl;
  //std::cout << "x = \n" << x.transpose() << std::endl;
  //std::cout << "f(x) = " << fx << std::endl;
  std::cout << "final temperature : " << x(0) << std::endl;

  return 0;
}
