/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#ifndef __OPENCV_ONLINEBOOSTING_HPP__
#define __OPENCV_ONLINEBOOSTING_HPP__

#include "opencv2/core.hpp"

namespace cv
{
//TODO based on the original implementation
//http://vision.ucsd.edu/~bbabenko/project_miltrack.shtml

class BaseClassifier;
class WeakClassifier;

class StrongClassifierDirectSelection
{
 public:

  StrongClassifierDirectSelection( int numBaseClf, int numWeakClf, Size patchSz, bool useFeatureEx = false, int iterationInit = 0 );

  virtual
  ~StrongClassifierDirectSelection();

  bool update( Mat response, Rect ROI, int target, float importance = 1.0 );

 private:

  //StrongClassifier
  int numBaseClassifier;
  int numAllWeakClassifier;
  BaseClassifier** baseClassifier;
  std::vector<float> alpha;
  cv::Size patchSize;

  bool useFeatureExchange;

  //StrongClassifierDirectSelection
  bool * m_errorMask;
  std::vector<float> m_errors;
  std::vector<float> m_sumErrors;
};

class BaseClassifier
{
 public:

  BaseClassifier( int numWeakClassifier, int iterationInit, Size patchSize );
  BaseClassifier( int numWeakClassifier, int iterationInit, WeakClassifier** weakClassifier );

  WeakClassifier** getReferenceWeakClassifier();
  void trainClassifier( Mat response, Rect ROI, int target, float importance, bool* errorMask );
  int selectBestClassifier( bool* errorMask, float importance, std::vector<float> & errors );
  int replaceWeakestClassifier( const std::vector<float> & errors, Size patchSize );
  void replaceClassifierStatistic( int sourceIndex, int targetIndex );
  int getIdxOfNewWeakClassifier();

  virtual ~BaseClassifier();

 protected:

  WeakClassifier** weakClassifier;
  bool m_referenceWeakClassifier;
  int m_numWeakClassifier;
  int m_selectedClassifier;
  int m_idxOfNewWeakClassifier;
  std::vector<float> m_wCorrect;
  std::vector<float> m_wWrong;
  int m_iterationInit;

};

class WeakClassifier
{

 public:

  WeakClassifier();
  virtual
  ~WeakClassifier();

  virtual bool
  update( Mat image, Rect ROI, int target );

  virtual int
  eval( Mat image, Rect ROI );

  virtual float
  getValue( Mat image, Rect ROI );

  virtual int
  getType();

};

} /* namespace cv */

#endif
