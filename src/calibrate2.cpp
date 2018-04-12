#include <lbfgs.h>
#include <stdio.h>
#include <cstddef>
#include <vector>
#include <math.h>
#include <iostream>

using std::vector;

class CrossEntropyLossWithTemperature
{
  // cross entropyloss is negative log likelihood applied on logits
  // or said differently : first softmax then negative log likelihood
private:
  lbfgsfloatval_t t_;

  const vector<vector<double> >* logits_;
  const vector<int>* labels_;

  lbfgs_parameter_t params_;

public:
  CrossEntropyLossWithTemperature(const vector<vector<double> >&logits, const vector<int>& labels) :
    t_(1), logits_(&logits), labels_(&labels)
  {
    lbfgs_parameter_init(&params_);
    params_.epsilon = 1E-6; // default anyway
    params_.max_iterations = 1E4; // set a limit in case of problem
  }

  CrossEntropyLossWithTemperature() : t_(1), logits_(NULL), labels_(NULL)
  {
    lbfgs_parameter_init(&params_);
    params_.epsilon = 1E-6; // default anyway
    params_.max_iterations = 1E4; // set a limit in case of problem
  }

  void setData(const vector<vector<double> >& logits, const vector<int>& labels)
  {
    logits_ = &logits;
    labels_ = &labels;
  }

  double getTemperature()
  {
    return (double)t_;
  }

  int calibrate()
  {
    lbfgsfloatval_t fx;
    if (logits_ == NULL || labels_ == NULL)
      {
        std::cerr << "no labels/logits given to calibration";
        return -1;
      }
    //int ret = lbfgs(1, &t_, &fx, _evaluate, _progress, this, NULL);
    return  lbfgs(1, &t_, &fx, _evaluate, NULL, this, &params_);
    //printf("L-BFGS optimization terminated with status code = %d\n", ret);
    //printf("  fx = %f, t = %f\n", fx, t_);
    //return ret;
  }

private:


  static lbfgsfloatval_t _evaluate(
                                   void *instance,
                                   const lbfgsfloatval_t *x,
                                   lbfgsfloatval_t *g,
                                   const int n,
                                   const lbfgsfloatval_t step
                                   )
  {
    return reinterpret_cast<CrossEntropyLossWithTemperature*>(instance)->evaluate(x, g, n, step);
  }

  lbfgsfloatval_t evaluate(
                           const lbfgsfloatval_t *x,
                           lbfgsfloatval_t *g,
                           const int n,
                           const lbfgsfloatval_t step
                           )
  {
    lbfgsfloatval_t fx = 0.0;

    // fx is value at x[0]
    // g[0] is derivation of f wrt t, evaluated at x[0]
    // now the computation of crossentropyloss and its derivative, averaged over batch size

    double temperature = *x;
    int nlogits = (*logits_)[0].size();
    double sum_fx = 0;
    double sum_gfx = 0;

    for (int bi = 0; bi < logits_->size(); ++bi)
      {
        const vector<double>& logits = (*logits_)[bi];
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
    fx = (lbfgsfloatval_t)(sum_fx / (double)logits_->size());
    *g = (lbfgsfloatval_t)(sum_gfx / (double)logits_->size());
    return fx;
  }

  static int _progress(
                       void *instance,
                       const lbfgsfloatval_t *x,
                       const lbfgsfloatval_t *g,
                       const lbfgsfloatval_t fx,
                       const lbfgsfloatval_t xnorm,
                       const lbfgsfloatval_t gnorm,
                       const lbfgsfloatval_t step,
                       int n,
                       int k,
                       int ls
                       )
  {
    return reinterpret_cast<CrossEntropyLossWithTemperature*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
  }

  int progress(
               const lbfgsfloatval_t *x,
               const lbfgsfloatval_t *g,
               const lbfgsfloatval_t fx,
               const lbfgsfloatval_t xnorm,
               const lbfgsfloatval_t gnorm,
               const lbfgsfloatval_t step,
               int n,
               int k,
               int ls
               )
  {
    printf("Iteration %d:", k);
    printf("  fx = %f, x[0] = %f\n", fx, x[0]);
    //printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    //printf("\n");
    return 0;
  }


};




int main(int argc, char **argv)
{
  CrossEntropyLossWithTemperature celwt;
  std::vector<int> labels {0, 3 , 5 ,3};
  std::vector<std::vector<double> > logits{
    {50, 10, 3, 25, 10, 2},
      {-10, 20, 10, 20, 20, 1},
        {10, 50, 5, 30, 30, 40},
          {1, 20, 5, 15, 40, 4}};

  celwt.setData(logits,labels);

  celwt.calibrate();
  std::cout<< "final temperature : " << celwt.getTemperature() << std::endl;
}
