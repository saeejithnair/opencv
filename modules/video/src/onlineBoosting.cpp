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

#include "precomp.hpp"
#include "opencv2/video/onlineBoosting.hpp"

namespace cv
{

StrongClassifierDirectSelection::StrongClassifierDirectSelection( int numBaseClf, int numWeakClf, Size patchSz, bool useFeatureEx, int iterationInit )
{
  //StrongClassifier
  numBaseClassifier = numBaseClf;
  numAllWeakClassifier = numWeakClf + iterationInit;

  alpha.assign( numBaseClf, 0 );

  patchSize = patchSz;
  useFeatureExchange = useFeatureEx;

  //StrongClassifierDirectSelection
  baseClassifier = new BaseClassifier*[numBaseClassifier];
  baseClassifier[0] = new BaseClassifier( numWeakClf, iterationInit, patchSize );

  for ( int curBaseClassifier = 1; curBaseClassifier < numBaseClassifier; curBaseClassifier++ )
    baseClassifier[curBaseClassifier] = new BaseClassifier( numWeakClf, iterationInit, baseClassifier[0]->getReferenceWeakClassifier() );

  m_errorMask = new bool[numAllWeakClassifier];
  m_errors.resize( numAllWeakClassifier );
  m_sumErrors.resize( numAllWeakClassifier );
}

StrongClassifierDirectSelection::~StrongClassifierDirectSelection()
{
  for ( int curBaseClassifier = 0; curBaseClassifier < numBaseClassifier; curBaseClassifier++ )
    delete baseClassifier[curBaseClassifier];
  delete[] baseClassifier;
  alpha.clear();
}

bool StrongClassifierDirectSelection::update( Mat response, Rect ROI, int target, float importance )
{
  memset( m_errorMask, 0, numAllWeakClassifier * sizeof(bool) );
  m_errors.assign( numAllWeakClassifier, 0 );
  m_sumErrors.assign( numAllWeakClassifier, 0 );

  baseClassifier[0]->trainClassifier( response, ROI, target, importance, m_errorMask );
  for ( int curBaseClassifier = 0; curBaseClassifier < numBaseClassifier; curBaseClassifier++ )
  {
    int selectedClassifier = baseClassifier[curBaseClassifier]->selectBestClassifier( m_errorMask, importance, m_errors );

    if( m_errors[selectedClassifier] >= 0.5 )
      alpha[curBaseClassifier] = 0;
    else
      alpha[curBaseClassifier] = logf( ( 1.0f - m_errors[selectedClassifier] ) / m_errors[selectedClassifier] );

    if( m_errorMask[selectedClassifier] )
      importance *= (float) sqrt( ( 1.0f - m_errors[selectedClassifier] ) / m_errors[selectedClassifier] );
    else
      importance *= (float) sqrt( m_errors[selectedClassifier] / ( 1.0f - m_errors[selectedClassifier] ) );

    //weight limitation
    //if (importance > 100) importance = 100;

    //sum up errors
    for ( int curWeakClassifier = 0; curWeakClassifier < numAllWeakClassifier; curWeakClassifier++ )
    {
      if( m_errors[curWeakClassifier] != FLT_MAX && m_sumErrors[curWeakClassifier] >= 0 )
        m_sumErrors[curWeakClassifier] += m_errors[curWeakClassifier];
    }

    //mark feature as used
    m_sumErrors[selectedClassifier] = -1;
    m_errors[selectedClassifier] = FLT_MAX;
  }

  if( useFeatureExchange )
  {
    int replacedClassifier = baseClassifier[0]->replaceWeakestClassifier( m_sumErrors, patchSize );
    if( replacedClassifier > 0 )
      for ( int curBaseClassifier = 1; curBaseClassifier < numBaseClassifier; curBaseClassifier++ )
        baseClassifier[curBaseClassifier]->replaceClassifierStatistic( baseClassifier[0]->getIdxOfNewWeakClassifier(), replacedClassifier );
  }

  return true;
}

float StrongClassifierDirectSelection::eval( Mat response, Rect ROI )
{
  float value = 0.0f;
  int curBaseClassifier = 0;

  for ( curBaseClassifier = 0; curBaseClassifier < numBaseClassifier; curBaseClassifier++ )
    value += baseClassifier[curBaseClassifier]->eval( response, ROI ) * alpha[curBaseClassifier];

  return value;
}

BaseClassifier::BaseClassifier( int numWeakClassifier, int iterationInit, Size patchSize )
{
  this->m_numWeakClassifier = numWeakClassifier;
  this->m_iterationInit = iterationInit;

  weakClassifier = new WeakClassifierHaarFeature*[numWeakClassifier + iterationInit];
  m_idxOfNewWeakClassifier = numWeakClassifier;

  generateRandomClassifier( patchSize );

  m_referenceWeakClassifier = false;
  m_selectedClassifier = 0;

  m_wCorrect.assign( numWeakClassifier + iterationInit, 0 );

  m_wWrong.assign( numWeakClassifier + iterationInit, 0 );

  for ( int curWeakClassifier = 0; curWeakClassifier < numWeakClassifier + iterationInit; curWeakClassifier++ )
    m_wWrong[curWeakClassifier] = m_wCorrect[curWeakClassifier] = 1;
}

BaseClassifier::BaseClassifier( int numWeakClassifier, int iterationInit, WeakClassifierHaarFeature** weakClassifier )
{
  this->m_numWeakClassifier = numWeakClassifier;
  this->m_iterationInit = iterationInit;
  this->weakClassifier = weakClassifier;
  m_referenceWeakClassifier = true;
  m_selectedClassifier = 0;
  m_idxOfNewWeakClassifier = numWeakClassifier;

  m_wCorrect.assign( numWeakClassifier + iterationInit, 0 );
  m_wWrong.assign( numWeakClassifier + iterationInit, 0 );

  for ( int curWeakClassifier = 0; curWeakClassifier < numWeakClassifier + iterationInit; curWeakClassifier++ )
    m_wWrong[curWeakClassifier] = m_wCorrect[curWeakClassifier] = 1;
}

void BaseClassifier::generateRandomClassifier( Size patchSize )
{
  for ( int curWeakClassifier = 0; curWeakClassifier < m_numWeakClassifier + m_iterationInit; curWeakClassifier++ )
  {
    weakClassifier[curWeakClassifier] = new WeakClassifierHaarFeature( patchSize );
  }
}

void BaseClassifier::trainClassifier( Mat response, Rect ROI, int target, float importance, bool* errorMask )
{
  //get poisson value
  double A = 1;
  int K = 0;
  int K_max = 10;
  while ( 1 )
  {
    double U_k = (double) rand() / RAND_MAX;
    A *= U_k;
    if( K > K_max || A < exp( -importance ) )
      break;
    K++;
  }

  for ( int curK = 0; curK <= K; curK++ )
    for ( int curWeakClassifier = 0; curWeakClassifier < m_numWeakClassifier + m_iterationInit; curWeakClassifier++ )
      errorMask[curWeakClassifier] = weakClassifier[curWeakClassifier]->update( response, ROI, target );

}

int BaseClassifier::selectBestClassifier( bool* errorMask, float importance, std::vector<float> & errors )
{
  float minError = FLT_MAX;
  int tmp_selectedClassifier = m_selectedClassifier;

  for ( int curWeakClassifier = 0; curWeakClassifier < m_numWeakClassifier + m_iterationInit; curWeakClassifier++ )
  {
    if( errorMask[curWeakClassifier] )
    {
      m_wWrong[curWeakClassifier] += importance;
    }
    else
    {
      m_wCorrect[curWeakClassifier] += importance;
    }

    if( errors[curWeakClassifier] == FLT_MAX )
      continue;

    errors[curWeakClassifier] = m_wWrong[curWeakClassifier] / ( m_wWrong[curWeakClassifier] + m_wCorrect[curWeakClassifier] );

    /*if(errors[curWeakClassifier] < 0.001 || !(errors[curWeakClassifier]>0.0))
     {
     errors[curWeakClassifier] = 0.001;
     }

     if(errors[curWeakClassifier] >= 1.0)
     errors[curWeakClassifier] = 0.999;

     assert (errors[curWeakClassifier] > 0.0);
     assert (errors[curWeakClassifier] < 1.0);*/

    if( curWeakClassifier < m_numWeakClassifier )
    {
      if( errors[curWeakClassifier] < minError )
      {
        minError = errors[curWeakClassifier];
        tmp_selectedClassifier = curWeakClassifier;
      }
    }
  }

  m_selectedClassifier = tmp_selectedClassifier;
  return m_selectedClassifier;
}

int BaseClassifier::replaceWeakestClassifier( const std::vector<float> & errors, Size patchSize )
{
  float maxError = 0.0f;
  int index = -1;

  //search the classifier with the largest error
  for ( int curWeakClassifier = m_numWeakClassifier - 1; curWeakClassifier >= 0; curWeakClassifier-- )
  {
    if( errors[curWeakClassifier] > maxError )
    {
      maxError = errors[curWeakClassifier];
      index = curWeakClassifier;
    }
  }

  CV_Assert( index > -1 );
  CV_Assert( index != m_selectedClassifier );

  //replace
  m_idxOfNewWeakClassifier++;
  if( m_idxOfNewWeakClassifier == m_numWeakClassifier + m_iterationInit )
    m_idxOfNewWeakClassifier = m_numWeakClassifier;

  if( maxError > errors[m_idxOfNewWeakClassifier] )
  {
    delete weakClassifier[index];
    weakClassifier[index] = weakClassifier[m_idxOfNewWeakClassifier];
    m_wWrong[index] = m_wWrong[m_idxOfNewWeakClassifier];
    m_wWrong[m_idxOfNewWeakClassifier] = 1;
    m_wCorrect[index] = m_wCorrect[m_idxOfNewWeakClassifier];
    m_wCorrect[m_idxOfNewWeakClassifier] = 1;

    weakClassifier[m_idxOfNewWeakClassifier] = new WeakClassifierHaarFeature( patchSize );

    return index;
  }
  else
    return -1;
}
void BaseClassifier::replaceClassifierStatistic( int sourceIndex, int targetIndex )
{
  CV_Assert( targetIndex >= 0 );
  CV_Assert( targetIndex != m_selectedClassifier );
  CV_Assert( targetIndex < m_numWeakClassifier );

  //replace
  m_wWrong[targetIndex] = m_wWrong[sourceIndex];
  m_wWrong[sourceIndex] = 1.0f;
  m_wCorrect[targetIndex] = m_wCorrect[sourceIndex];
  m_wCorrect[sourceIndex] = 1.0f;
}

WeakClassifierHaarFeature** BaseClassifier::getReferenceWeakClassifier()
{
  return weakClassifier;
}

int BaseClassifier::getIdxOfNewWeakClassifier()
{
  return m_idxOfNewWeakClassifier;
}

int BaseClassifier::eval( Mat response, Rect ROI )
{
  return weakClassifier[m_selectedClassifier]->eval( response, ROI );
}

BaseClassifier::~BaseClassifier()
{
  if( !m_referenceWeakClassifier )
  {
    for ( int curWeakClassifier = 0; curWeakClassifier < m_numWeakClassifier + m_iterationInit; curWeakClassifier++ )
      delete weakClassifier[curWeakClassifier];

    delete[] weakClassifier;
  }
  m_wCorrect.clear();
  m_wWrong.clear();
}

WeakClassifierHaarFeature::WeakClassifierHaarFeature( Size patchSize )
{
  //TODO
  /*m_feature = new FeatureHaar( patchSize );
   generateRandomClassifier();
   m_feature->getInitialDistribution( (EstimatedGaussDistribution*) m_classifier->getDistribution( -1 ) );
   m_feature->getInitialDistribution( (EstimatedGaussDistribution*) m_classifier->getDistribution( 1 ) );*/
}

int WeakClassifierHaarFeature::eval( Mat response, Rect ROI )
{
  //TODO
  return response.at<float>( 0, 0 );
}

WeakClassifierHaarFeature::~WeakClassifierHaarFeature()
{
  //TODO
}

bool WeakClassifierHaarFeature::update( Mat response, Rect ROI, int target )
{
  //TODO
  return true;
}

Detector::Detector( StrongClassifierDirectSelection* classifier )
{
  m_classifier = classifier;

  m_sizeConfidences = 0;
  m_maxConfidence = -FLT_MAX;
  m_numDetections = 0;
  m_idxBestDetection = -1;
  m_sizeDetections = 0;
}

Detector::~Detector()
{

}
} /* namespace cv */
