#include "tscaling.h"
#include <boost/python.hpp>

void setDataPy(TempScaler& ts, boost::python::list& llogits, boost::python::list& llabels)
{
  std::vector<std::vector<double>> vlogits;
  for (int i=0; i< len(llogits); ++i)
    {
      std::vector<double> logits;
      boost::python::list l = boost::python::extract<boost::python::list>(llogits[i]);
      for (int j = 0; j<len(l); ++j)
        {
          logits.push_back(boost::python::extract<double>(l[j]));
        }
      vlogits.push_back(logits);
    }

  std::vector<int> labels;
  for (int i=0; i< len(llabels); ++i)
    {
      labels.push_back(boost::python::extract<double>(llabels[i]));
    }
  ts.setData(vlogits,labels);
}

boost::python::tuple getPredConfPy(TempScaler& ts, boost::python::list& llogits)
{
  std::vector<double> logits;
  for (int i=0; i< len(llogits); ++i)
    {
      logits.push_back(boost::python::extract<double>(llogits[i]));
    }
  int pred;
  double conf;
  ts.getPredConf(logits, pred,conf);
  return boost::python::make_tuple(pred,conf);
}

BOOST_PYTHON_MODULE(pytscaling)
{
  using namespace boost::python;




  class_<TempScaler>("TempScaler")
    .def("setData", &setDataPy)
    .def("getPredConf", &getPredConfPy)
    .def("calibrate", &TempScaler::calibrate)
    .def("getTemperature", &TempScaler::getTemperature)
    ;
};
