## libtscaling Temperature scaling for neural network calibration

This library allows calibration of neural networks. See  (https://arxiv.org/abs/1706.04599) for detailed problem description and solutions. 

#### Description
Modern deep architectures have good prediction power, but are generally over confident. The idea is to use a validation set after initial training in order to add a distillation phase between training and production (predictions) allowing to calibrate confidence over this unseen set. The output of this library is a distillation temperature to be used for scaling logits before softmax a prediction time (logits should be divided by final temperature before softmax). 


#### Dependencies
Distillation temperature factor is computed via lbfgs, using either bundled liblbfgspp from (https://github.com/yixuan/LBFGSpp) that needs eigen, or liblbfgs available on unbuntu for instance (liblbfgs-dev).

#### Build
Cmake based.  If USE_LBFGSPP is turned off, then LBFGS_LIB is mandatory (auto detected in standard pathes); inversely, if LBFGS_LIB is not found, USE_LBFGSPP is forced to on.
in details:
- `mkdir build && cd build`
- `cmake .. -DUSE_LBFGSPP`    or `cmake ..`
- `make`

gives libtscaling.so

#### Example usage
API is going to change !!!

important methods: 
- TempScaler::setData(logits, validation_true_labels)
- TempScaler::calibrate() gives the final temperature

- CalibrationError(number_of_bins)
- CalibrationError::setData(confidences, predictions, targets)
- CalibrationError::ECE() gives expected calibration error
- CalibrationError::MCE() gives maximum calibration error
- CalibrationError::percents() gives percent of samples per confidence bin
- CalibrationError::accuracies() gives accuracies per confidence bin
- CalibrationError::to_py() output a matploblib'ed python file showing graphs as in  (https://arxiv.org/abs/1706.04599)

