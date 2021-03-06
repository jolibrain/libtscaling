#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "tscaling.h"

TempScaler::TempScaler(const std::vector<std::vector<double> >&logits,
                           const std::vector<int>& labels) :
  logits_(logits), labels_(labels)
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

TempScaler::TempScaler()
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
  logits_ = logits;
  for (std::vector<double> l : logits_)
    {
      for (double ll : l)
        std::cout << ll << " ";
      std::cout << std::endl;
    }

  labels_ = labels;
  for (int l : labels_)
    std::cout << l << " ";
  std::cout << std::endl;

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
  if (logits_.size()  == 0 || labels_.size() == 0)
    {
      std::cerr << "no labels/logits given to calibration";
      return -1;
    }

  for (std::vector<double> l : logits_)
    {
      for (double ll : l)
        std::cout << ll << " ";
      std::cout << std::endl;
    }

  for (int l : labels_)
    std::cout << l << " ";
  std::cout << std::endl;



#ifdef LBFGSPP
  double fx;
  std::cout << "about to minimize LBFGSPP" << std::endl;
  solver_->minimize(*this, t_, fx);
#else
  std::cout << "about to minimize NOLBFGSPP" << std::endl;
  lbfgsfloatval_t fx;
  lbfgs(1, &t_, &fx, _evaluate, NULL, this, params_);
#endif
  std::cout << "minimized" << std::endl;
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


void TempScaler::getPredConf(std::vector<double>& logits, int &pred, double&conf)
{
  double sum = exp(logits[0]);
  double max_exp = sum;
  pred = 0;
  for (unsigned int i =1; i<logits.size(); ++i)
    {
      double cur_exp = exp(logits[i]);
      sum += cur_exp;
      if (cur_exp > max_exp)
        {
          max_exp = cur_exp;
          pred = i;
        }
    }
  conf = max_exp / sum;
}

void TempScaler::getPredConfWithTemperature(std::vector<double>& logits, double temperature, int &pred, double&conf)
{
  double sum = exp(logits[0]/temperature);
  double max_exp = sum;
  pred = 0;
  for (unsigned int i =1; i<logits.size(); ++i)
    {
      double cur_exp = exp(logits[i]/temperature);
      sum += cur_exp;
      if (cur_exp > max_exp)
        {
          max_exp = cur_exp;
          pred = i;
        }
    }
  conf = max_exp / sum;
}


void TempScaler::getPredConfBatch(std::vector<std::vector<double> > logitbatch, std::vector<int>& preds, std::vector<double>&confs)
{
  for (unsigned int i = 0; i< logitbatch.size(); ++i)
    {
      getPredConf(logitbatch[i], preds[i], confs[i]);
    }
}

double TempScaler::crossEntropyLossWithTemperature(const double& x, double& gx)
{
  // return  value at x
  // gx is derivation of f wrt t, evaluated at x
  // now the computation of crossentropyloss and its derivative, averaged over batch size

  double temperature = x;
  int nlogits = logits_[0].size();
  double sum_fx = 0;
  double sum_gfx = 0;

  for (unsigned int bi = 0; bi < logits_.size(); ++bi)
    {
      const std::vector<double>& logits = logits_[bi];
      const int label = labels_[bi];

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

  gx = sum_gfx / (double)logits_.size();
  return sum_fx / (double)logits_.size();
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






CalibrationError::CalibrationError(int nbins): confs_(NULL), predictions_(NULL), targets_(NULL), nbins_(nbins), cached_(false)
{
  bins_ = std::vector<std::vector<int> >(nbins);
  bin_acc_cache_ = std::vector<double>(nbins);
  bin_conf_cache_ = std::vector<double>(nbins);
}


CalibrationError::CalibrationError(std::vector<double>& confidences,
                                   std::vector<int>& predictions,
                                   std::vector<int>& targets, int nbins) :
  confs_(&confidences), predictions_(&predictions), targets_(&targets), nbins_(nbins), cached_(false)
{
  bins_ = std::vector<std::vector<int> >(nbins);
  bin_acc_cache_ = std::vector<double>(nbins);
  bin_conf_cache_ = std::vector<double>(nbins);
  fill_bins();
}

void CalibrationError::setData(std::vector<double>& confidences, std::vector<int>& predictions,
                         std::vector<int>& targets)
{
  confs_ = &confidences;
  predictions_ = &predictions;
  targets_ = &targets;
  clear_bins();
  fill_bins();
}

void CalibrationError::fill_bins()
{
  for (unsigned int i =0; i< confs_->size(); ++i)
    {
      int bi = static_cast<int>(confs_->at(i)*nbins_);
      if (bi == confs_->at(i)*nbins_ && bi != 0)
        bi--;
      bins_[bi].push_back(i);
    }
  for (int bi = 0; bi< nbins_; ++bi)
    {
      bin_acc(bi);
      bin_conf(bi);
    }
  cached_ = true;
}

void CalibrationError::clear_bins()
{
  for (unsigned int i =0; i< bins_.size(); ++i)
    bins_[i].clear();
  cached_ = false;
}

double CalibrationError::bin_acc(int bi)
{
  if (cached_)
    return bin_acc_cache_[bi];
  if (bins_[bi].size() == 0)
    {
      bin_acc_cache_[bi] = 0;
      return 0;
    }
  int correct = 0;
  for (int i:bins_[bi])
    if (targets_->at(i) == predictions_->at(i))
      correct += 1;
  double acc = (double) correct / double(bins_[bi].size());
  bin_acc_cache_[bi] = acc;
  return acc;
}

double CalibrationError::bin_conf(int bi)
{
  if (cached_)
    return bin_conf_cache_[bi];
  if (bins_[bi].size() == 0)
    {
      bin_conf_cache_[bi] = 0;
      return 0;
    }
  double sum = 0;
  for (int i:bins_[bi])
    sum += confs_->at(i);
  double conf = sum / (double) bins_[bi].size();
  bin_conf_cache_[bi] = conf;
  return conf;
}

double CalibrationError::ECE()
{
  double ece = 0;
  double ntot = 0;
  for (int bi = 0; bi< nbins_; ++bi)
    {
      int bm = bins_[bi].size();
      if (bm == 0)
        continue;
      ntot += bm;
      ece += fabs(bin_acc(bi) - bin_conf(bi)) * bm;
    }
  return ece / (double) ntot;
}

double CalibrationError::MCE()
{
  double mce = 0;
  for (int bi = 0; bi < nbins_; ++bi)
    {
      if (bins_[bi].size() == 0)
        continue;
      double v = fabs(bin_acc(bi) - bin_conf(bi));
      if (v> mce)
        mce = v;
    }
  return mce;
}

void CalibrationError::display()
{
  for (int i =0; i<nbins_; ++i)
    {
      std::cout << "bin " << i << " :]" << (double)i/(double)nbins_<<":"<<(double)(i+1)/(double)nbins_<<"]";
      std::cout << " : {";
      for (unsigned int j =0; j<bins_[i].size(); ++j)
        std::cout << bins_[i][j] << ",";
      std::cout << "}   acc = "<< bin_acc(i) << "    conf = " << bin_conf(i) << std::endl;
    }
}

void CalibrationError::edges(std::vector<double>& edges)
{
  for (int i=0; i<nbins_; ++i)
    edges.push_back((double)i/(double)nbins_);
}


void CalibrationError::percents(std::vector<double>& percents)
{
  int tot = 0;

  for (int i=0; i<nbins_; ++i)
    {
      int n = bins_[i].size();
      percents.push_back(n);
      tot += n;
    }
  for (int i=0; i<nbins_; ++i)
    percents[i] /= (double)tot;
}

void CalibrationError::accuracies(std::vector<double>& accuracies)
{
  for (int i=0; i<nbins_; ++i)
    accuracies.push_back(bin_acc(i));
}

void CalibrationError::to_py(std::string fname)
{
  std::ofstream file;
  file.open (fname);
  std::vector<double> redges;
  edges(redges);
  std::vector<double> rpercents;
  percents(rpercents);
  std::vector<double> raccuracies;
  accuracies(raccuracies);

  file << "import matplotlib.pyplot" << std::endl;

  double width = 1.0/(double)nbins_;
  file << "width = " << width << std::endl;
  file << "centers = [";
  for (int i=0; i<nbins_-1; ++i)
    file << redges[i] + width/2 << ",";
  file << redges[nbins_-1] + width/2 << "]\n";

  file << "percents = [";
  for (int i=0; i<nbins_-1; ++i)
    file << rpercents[i] << ",";
  file << rpercents[nbins_-1] << "]\n";

  file << "accuracies = [";
  for (int i=0; i<nbins_-1; ++i)
    file << raccuracies[i] << ",";
  file << raccuracies[nbins_-1] << "]\n";

  file << "matplotlib.pyplot.bar(centers,percents,width)" << std::endl;
  file << "matplotlib.pyplot.xlabel('confidence')" << std::endl;
  file << "matplotlib.pyplot.ylabel('% samples')" << std::endl;
  file << "matplotlib.pyplot.savefig('conf_repartition.pdf')" << std::endl;
  file << "matplotlib.pyplot.clf()" << std::endl;
  file << "matplotlib.pyplot.bar(centers,accuracies,width)" << std::endl;
  file << "matplotlib.pyplot.xlabel('confidence')" << std::endl;
  file << "matplotlib.pyplot.ylabel('accuracy')" << std::endl;
  file << "matplotlib.pyplot.savefig('conf_accuracy.pdf')" << std::endl;

  file.close();

}
