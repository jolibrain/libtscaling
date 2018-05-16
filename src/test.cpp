#include "tscaling.h"
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>

int main(int argc, char **argv)
{
  TempScaler celwt;
  std::vector<int> targetb {0, 3 , 5 ,3}; // 4 predictions over classes 0..5
  std::vector<std::vector<double> > logitbatch{
    {50, 10, 3, 25, 10, 2},
      {-10, 20, 10, 20, 20, 1},
        {10, 50, 5, 30, 30, 40},
          {1, 20, 5, 15, 40, 4}};

  std::vector<double> confb(logitbatch.size());
  std::vector<int> predb(logitbatch.size());

  celwt.getPredConfBatch(logitbatch,predb,confb);


  CalibrationError ce(3);
  ce.setData(confb,predb, targetb);

  std::cout << "mce: " << ce.MCE() << "    ece: " << ce.ECE() << std::endl;
  ce.display();
  ce.to_py("graphs_calibration.py");


  celwt.setData(logitbatch,targetb);
  double temperature = celwt.calibrate();
  std::cout<< "final temperature : " << temperature << std::endl;

  for_each(logitbatch.begin(), logitbatch.end(), [temperature](std::vector<double>& v){ std::for_each(v.begin(), v.end(), [temperature](double&l){l/=temperature;});});


  celwt.getPredConfBatch(logitbatch,predb,confb);
  ce.setData(confb,predb,targetb);

  std::cout << "mce: " << ce.MCE() << "    ece: " << ce.ECE() << std::endl;
  ce.display();
  ce.to_py("graphs.py");
}
