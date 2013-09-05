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

StrongClassifierDirectSelection::StrongClassifierDirectSelection( int numBaseClf, int numWeakClf, Size patchSz, const Rect& sampleROI,
                                                                  bool useFeatureEx, int iterationInit )
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

  ROI = sampleROI;
  detector = new Detector( this );
}

StrongClassifierDirectSelection::~StrongClassifierDirectSelection()
{
  for ( int curBaseClassifier = 0; curBaseClassifier < numBaseClassifier; curBaseClassifier++ )
    delete baseClassifier[curBaseClassifier];
  delete[] baseClassifier;
  alpha.clear();
  delete detector;
}

Size StrongClassifierDirectSelection::getPatchSize() const
{
  return patchSize;
}

Rect StrongClassifierDirectSelection::getROI() const
{
  return ROI;
}

float StrongClassifierDirectSelection::classifySmooth( const std::vector<Mat>& images, const Rect& sampleROI, int& idx )
{
  //TODO
  ROI = sampleROI;
  idx = 0;
  float confidence = 0;
  //detector->classify (image, patches);
  detector->classifySmooth( images );

  //move to best detection
  if( detector->getNumDetections() <= 0 )
  {
    confidence = 0;
    return confidence;
  }
  idx = detector->getPatchIdxOfBestDetection();
  confidence = detector->getConfidenceOfBestDetection();
  /*
   classifier->update( image, patches->getSpecialRect( "UpperLeft" ), -1 );
   classifier->update( image, trackedPatch, 1 );
   classifier->update( image, patches->getSpecialRect( "UpperRight" ), -1 );
   classifier->update( image, trackedPatch, 1 );
   classifier->update( image, patches->getSpecialRect( "LowerLeft" ), -1 );
   classifier->update( image, trackedPatch, 1 );
   classifier->update( image, patches->getSpecialRect( "LowerRight" ), -1 );
   classifier->update( image, trackedPatch, 1 );

   return true;*/

  return confidence;
}

bool StrongClassifierDirectSelection::update( const Mat& image, Rect ROI, int target, float importance )
{
  memset( m_errorMask, 0, numAllWeakClassifier * sizeof(bool) );
  m_errors.assign( numAllWeakClassifier, 0 );
  m_sumErrors.assign( numAllWeakClassifier, 0 );

  baseClassifier[0]->trainClassifier( image, ROI, target, importance, m_errorMask );
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

float StrongClassifierDirectSelection::eval( const Mat& response, Rect ROI )
{
  float value = 0.0f;
  int curBaseClassifier = 0;

  for ( curBaseClassifier = 0; curBaseClassifier < numBaseClassifier; curBaseClassifier++ )
    value += baseClassifier[curBaseClassifier]->eval( response, ROI ) * alpha[curBaseClassifier];

  return value;
}

int StrongClassifierDirectSelection::getNumBaseClassifier()
{
  return numBaseClassifier;
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

void BaseClassifier::generateRandomClassifier( Size patchSize )
{
  for ( int curWeakClassifier = 0; curWeakClassifier < m_numWeakClassifier + m_iterationInit; curWeakClassifier++ )
  {
    weakClassifier[curWeakClassifier] = new WeakClassifierHaarFeature( patchSize );
  }
}

int BaseClassifier::eval( const Mat& image, Rect ROI )
{
  return weakClassifier[m_selectedClassifier]->eval( image, ROI );
}

float BaseClassifier::getValue( const Mat& image, Rect ROI, int weakClassifierIdx )
{
  if( weakClassifierIdx < 0 || weakClassifierIdx >= m_numWeakClassifier )
    return weakClassifier[m_selectedClassifier]->getValue( image, ROI );
  return weakClassifier[weakClassifierIdx]->getValue( image, ROI );
}

void BaseClassifier::trainClassifier( const Mat& image, Rect ROI, int target, float importance, bool* errorMask )
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
      errorMask[curWeakClassifier] = weakClassifier[curWeakClassifier]->update( image, ROI, target );

}

void BaseClassifier::getErrorMask( const Mat& image, Rect ROI, int target, bool* errorMask )
{
  for ( int curWeakClassifier = 0; curWeakClassifier < m_numWeakClassifier + m_iterationInit; curWeakClassifier++ )
    errorMask[curWeakClassifier] = ( weakClassifier[curWeakClassifier]->eval( image, ROI ) != target );
}

float BaseClassifier::getError( int curWeakClassifier )
{
  if( curWeakClassifier == -1 )
    curWeakClassifier = m_selectedClassifier;
  return m_wWrong[curWeakClassifier] / ( m_wWrong[curWeakClassifier] + m_wCorrect[curWeakClassifier] );
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

void BaseClassifier::getErrors( float* errors )
{
  for ( int curWeakClassifier = 0; curWeakClassifier < m_numWeakClassifier + m_iterationInit; curWeakClassifier++ )
  {
    if( errors[curWeakClassifier] == FLT_MAX )
      continue;

    errors[curWeakClassifier] = m_wWrong[curWeakClassifier] / ( m_wWrong[curWeakClassifier] + m_wCorrect[curWeakClassifier] );

    CV_Assert( errors[curWeakClassifier] > 0 );
  }
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

#define SQROOTHALF 0.7071
#define INITSIGMA( numAreas ) ( static_cast<float>( sqrt( 256.0f*256.0f / 12.0f * (numAreas) ) ) );

FeatureHaar::FeatureHaar( Size patchSize )
{
  try
  {
    generateRandomFeature( patchSize );
  }
  catch ( ... )
  {
    throw;
  }
}

void FeatureHaar::generateRandomFeature( Size patchSize )
{
  cv::Point2i position;
  Size baseDim;
  Size sizeFactor;
  int area;

  Size minSize = Size( 3, 3 );
  int minArea = 9;

  bool valid = false;
  while ( !valid )
  {
    //chosse position and scale
    position.y = rand() % ( patchSize.height );
    position.x = rand() % ( patchSize.width );

    baseDim.width = (int) ( ( 1 - sqrt( 1 - (float) rand() / RAND_MAX ) ) * patchSize.width );
    baseDim.height = (int) ( ( 1 - sqrt( 1 - (float) rand() / RAND_MAX ) ) * patchSize.height );

    //select types
    //float probType[11] = {0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0909f, 0.0950f};
    float probType[11] =
    { 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float prob = (float) rand() / RAND_MAX;

    if( prob < probType[0] )
    {
      //check if feature is valid
      sizeFactor.height = 2;
      sizeFactor.width = 1;
      if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
        continue;
      area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
      if( area < minArea )
        continue;

      m_type = 1;
      m_numAreas = 2;
      m_weights.resize( m_numAreas );
      m_weights[0] = 1;
      m_weights[1] = -1;
      m_areas.resize( m_numAreas );
      m_areas[0].x = position.x;
      m_areas[0].y = position.y;
      m_areas[0].height = baseDim.height;
      m_areas[0].width = baseDim.width;
      m_areas[1].x = position.x;
      m_areas[1].y = position.y + baseDim.height;
      m_areas[1].height = baseDim.height;
      m_areas[1].width = baseDim.width;
      m_initMean = 0;
      m_initSigma = INITSIGMA( m_numAreas );

      valid = true;

    }
    else if( prob < probType[0] + probType[1] )
    {
      //check if feature is valid
      sizeFactor.height = 1;
      sizeFactor.width = 2;
      if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
        continue;
      area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
      if( area < minArea )
        continue;

      m_type = 2;
      m_numAreas = 2;
      m_weights.resize( m_numAreas );
      m_weights[0] = 1;
      m_weights[1] = -1;
      m_areas.resize( m_numAreas );
      m_areas[0].x = position.x;
      m_areas[0].y = position.y;
      m_areas[0].height = baseDim.height;
      m_areas[0].width = baseDim.width;
      m_areas[1].x = position.x + baseDim.width;
      m_areas[1].y = position.y;
      m_areas[1].height = baseDim.height;
      m_areas[1].width = baseDim.width;
      m_initMean = 0;
      m_initSigma = INITSIGMA( m_numAreas );
      valid = true;

    }
    else if( prob < probType[0] + probType[1] + probType[2] )
    {
      //check if feature is valid
      sizeFactor.height = 4;
      sizeFactor.width = 1;
      if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
        continue;
      area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
      if( area < minArea )
        continue;

      m_type = 3;
      m_numAreas = 3;
      m_weights.resize( m_numAreas );
      m_weights[0] = 1;
      m_weights[1] = -2;
      m_weights[2] = 1;
      m_areas.resize( m_numAreas );
      m_areas[0].x = position.x;
      m_areas[0].y = position.y;
      m_areas[0].height = baseDim.height;
      m_areas[0].width = baseDim.width;
      m_areas[1].x = position.x;
      m_areas[1].y = position.y + baseDim.height;
      m_areas[1].height = 2 * baseDim.height;
      m_areas[1].width = baseDim.width;
      m_areas[2].y = position.y + 3 * baseDim.height;
      m_areas[2].x = position.x;
      m_areas[2].height = baseDim.height;
      m_areas[2].width = baseDim.width;
      m_initMean = 0;
      m_initSigma = INITSIGMA( m_numAreas );
      valid = true;
    }
    else if( prob < probType[0] + probType[1] + probType[2] + probType[3] )
    {
      //check if feature is valid
      sizeFactor.height = 1;
      sizeFactor.width = 4;
      if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
        continue;
      area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
      if( area < minArea )
        continue;

      m_type = 3;
      m_numAreas = 3;
      m_weights.resize( m_numAreas );
      m_weights[0] = 1;
      m_weights[1] = -2;
      m_weights[2] = 1;
      m_areas.resize( m_numAreas );
      m_areas[0].x = position.x;
      m_areas[0].y = position.y;
      m_areas[0].height = baseDim.height;
      m_areas[0].width = baseDim.width;
      m_areas[1].x = position.x + baseDim.width;
      m_areas[1].y = position.y;
      m_areas[1].height = baseDim.height;
      m_areas[1].width = 2 * baseDim.width;
      m_areas[2].y = position.y;
      m_areas[2].x = position.x + 3 * baseDim.width;
      m_areas[2].height = baseDim.height;
      m_areas[2].width = baseDim.width;
      m_initMean = 0;
      m_initSigma = INITSIGMA( m_numAreas );
      valid = true;
    }
    else if( prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] )
    {
      //check if feature is valid
      sizeFactor.height = 2;
      sizeFactor.width = 2;
      if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
        continue;
      area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
      if( area < minArea )
        continue;

      m_type = 5;
      m_numAreas = 4;
      m_weights.resize( m_numAreas );
      m_weights[0] = 1;
      m_weights[1] = -1;
      m_weights[2] = -1;
      m_weights[3] = 1;
      m_areas.resize( m_numAreas );
      m_areas[0].x = position.x;
      m_areas[0].y = position.y;
      m_areas[0].height = baseDim.height;
      m_areas[0].width = baseDim.width;
      m_areas[1].x = position.x + baseDim.width;
      m_areas[1].y = position.y;
      m_areas[1].height = baseDim.height;
      m_areas[1].width = baseDim.width;
      m_areas[2].y = position.y + baseDim.height;
      m_areas[2].x = position.x;
      m_areas[2].height = baseDim.height;
      m_areas[2].width = baseDim.width;
      m_areas[3].y = position.y + baseDim.height;
      m_areas[3].x = position.x + baseDim.width;
      m_areas[3].height = baseDim.height;
      m_areas[3].width = baseDim.width;
      m_initMean = 0;
      m_initSigma = INITSIGMA( m_numAreas );
      valid = true;
    }
    else if( prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] )
    {
      //check if feature is valid
      sizeFactor.height = 3;
      sizeFactor.width = 3;
      if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
        continue;
      area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
      if( area < minArea )
        continue;

      m_type = 6;
      m_numAreas = 2;
      m_weights.resize( m_numAreas );
      m_weights[0] = 1;
      m_weights[1] = -9;
      m_areas.resize( m_numAreas );
      m_areas[0].x = position.x;
      m_areas[0].y = position.y;
      m_areas[0].height = 3 * baseDim.height;
      m_areas[0].width = 3 * baseDim.width;
      m_areas[1].x = position.x + baseDim.width;
      m_areas[1].y = position.y + baseDim.height;
      m_areas[1].height = baseDim.height;
      m_areas[1].width = baseDim.width;
      m_initMean = -8 * 128;
      m_initSigma = INITSIGMA( m_numAreas );
      valid = true;
    }
    else if( prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] + probType[6] )
    {
      //check if feature is valid
      sizeFactor.height = 3;
      sizeFactor.width = 1;
      if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
        continue;
      area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
      if( area < minArea )
        continue;

      m_type = 7;
      m_numAreas = 3;
      m_weights.resize( m_numAreas );
      m_weights[0] = 1;
      m_weights[1] = -2;
      m_weights[2] = 1;
      m_areas.resize( m_numAreas );
      m_areas[0].x = position.x;
      m_areas[0].y = position.y;
      m_areas[0].height = baseDim.height;
      m_areas[0].width = baseDim.width;
      m_areas[1].x = position.x;
      m_areas[1].y = position.y + baseDim.height;
      m_areas[1].height = baseDim.height;
      m_areas[1].width = baseDim.width;
      m_areas[2].y = position.y + baseDim.height * 2;
      m_areas[2].x = position.x;
      m_areas[2].height = baseDim.height;
      m_areas[2].width = baseDim.width;
      m_initMean = 0;
      m_initSigma = INITSIGMA( m_numAreas );
      valid = true;
    }
    else if( prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] + probType[6] + probType[7] )
    {
      //check if feature is valid
      sizeFactor.height = 1;
      sizeFactor.width = 3;
      if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
        continue;

      area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;

      if( area < minArea )
        continue;

      m_type = 8;
      m_numAreas = 3;
      m_weights.resize( m_numAreas );
      m_weights[0] = 1;
      m_weights[1] = -2;
      m_weights[2] = 1;
      m_areas.resize( m_numAreas );
      m_areas[0].x = position.x;
      m_areas[0].y = position.y;
      m_areas[0].height = baseDim.height;
      m_areas[0].width = baseDim.width;
      m_areas[1].x = position.x + baseDim.width;
      m_areas[1].y = position.y;
      m_areas[1].height = baseDim.height;
      m_areas[1].width = baseDim.width;
      m_areas[2].y = position.y;
      m_areas[2].x = position.x + 2 * baseDim.width;
      m_areas[2].height = baseDim.height;
      m_areas[2].width = baseDim.width;
      m_initMean = 0;
      m_initSigma = INITSIGMA( m_numAreas );
      valid = true;
    }
    else if( prob < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] + probType[6] + probType[7] + probType[8] )
    {
      //check if feature is valid
      sizeFactor.height = 3;
      sizeFactor.width = 3;
      if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
        continue;
      area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
      if( area < minArea )
        continue;

      m_type = 9;
      m_numAreas = 2;
      m_weights.resize( m_numAreas );
      m_weights[0] = 1;
      m_weights[1] = -2;
      m_areas.resize( m_numAreas );
      m_areas[0].x = position.x;
      m_areas[0].y = position.y;
      m_areas[0].height = 3 * baseDim.height;
      m_areas[0].width = 3 * baseDim.width;
      m_areas[1].x = position.x + baseDim.width;
      m_areas[1].y = position.y + baseDim.height;
      m_areas[1].height = baseDim.height;
      m_areas[1].width = baseDim.width;
      m_initMean = 0;
      m_initSigma = INITSIGMA( m_numAreas );
      valid = true;
    }
    else if( prob
        < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] + probType[6] + probType[7] + probType[8] + probType[9] )
    {
      //check if feature is valid
      sizeFactor.height = 3;
      sizeFactor.width = 1;
      if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
        continue;
      area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
      if( area < minArea )
        continue;

      m_type = 10;
      m_numAreas = 3;
      m_weights.resize( m_numAreas );
      m_weights[0] = 1;
      m_weights[1] = -1;
      m_weights[2] = 1;
      m_areas.resize( m_numAreas );
      m_areas[0].x = position.x;
      m_areas[0].y = position.y;
      m_areas[0].height = baseDim.height;
      m_areas[0].width = baseDim.width;
      m_areas[1].x = position.x;
      m_areas[1].y = position.y + baseDim.height;
      m_areas[1].height = baseDim.height;
      m_areas[1].width = baseDim.width;
      m_areas[2].y = position.y + baseDim.height * 2;
      m_areas[2].x = position.x;
      m_areas[2].height = baseDim.height;
      m_areas[2].width = baseDim.width;
      m_initMean = 128;
      m_initSigma = INITSIGMA( m_numAreas );
      valid = true;
    }
    else if( prob
        < probType[0] + probType[1] + probType[2] + probType[3] + probType[4] + probType[5] + probType[6] + probType[7] + probType[8] + probType[9]
            + probType[10] )
    {
      //check if feature is valid
      sizeFactor.height = 1;
      sizeFactor.width = 3;
      if( position.y + baseDim.height * sizeFactor.height >= patchSize.height || position.x + baseDim.width * sizeFactor.width >= patchSize.width )
        continue;
      area = baseDim.height * sizeFactor.height * baseDim.width * sizeFactor.width;
      if( area < minArea )
        continue;

      m_type = 11;
      m_numAreas = 3;
      m_weights.resize( m_numAreas );
      m_weights[0] = 1;
      m_weights[1] = -1;
      m_weights[2] = 1;
      m_areas.resize( m_numAreas );
      m_areas[0].x = position.x;
      m_areas[0].y = position.y;
      m_areas[0].height = baseDim.height;
      m_areas[0].width = baseDim.width;
      m_areas[1].x = position.x + baseDim.width;
      m_areas[1].y = position.y;
      m_areas[1].height = baseDim.height;
      m_areas[1].width = baseDim.width;
      m_areas[2].y = position.y;
      m_areas[2].x = position.x + 2 * baseDim.width;
      m_areas[2].height = baseDim.height;
      m_areas[2].width = baseDim.width;
      m_initMean = 128;
      m_initSigma = INITSIGMA( m_numAreas );
      valid = true;
    }
    else
    CV_Assert( false );
  }

  m_initSize = patchSize;
  m_curSize = m_initSize;
  m_scaleFactorWidth = m_scaleFactorHeight = 1.0f;
  m_scaleAreas.resize( m_numAreas );
  m_scaleWeights.resize( m_numAreas );
  for ( int curArea = 0; curArea < m_numAreas; curArea++ )
  {
    m_scaleAreas[curArea] = m_areas[curArea];
    m_scaleWeights[curArea] = (float) m_weights[curArea] / (float) ( m_areas[curArea].width * m_areas[curArea].height );
  }
}

bool FeatureHaar::eval( const Mat& image, Rect ROI, float* result )
{
  *result = 0.0f;

  // define the minimum size
  Size minSize = Size( 3, 3 );

  // printf("in eval %d = %d\n",curSize.width,ROI.width );

  if( m_curSize.width != ROI.width || m_curSize.height != ROI.height )
  {
    m_curSize = cv::Size( ROI.width, ROI.height );
    if( ! ( m_initSize == m_curSize ) )
    {
      m_scaleFactorHeight = (float) m_curSize.height / m_initSize.height;
      m_scaleFactorWidth = (float) m_curSize.width / m_initSize.width;

      for ( int curArea = 0; curArea < m_numAreas; curArea++ )
      {
        m_scaleAreas[curArea].height = (int) floor( (float) m_areas[curArea].height * m_scaleFactorHeight + 0.5f );
        m_scaleAreas[curArea].width = (int) floor( (float) m_areas[curArea].width * m_scaleFactorWidth + 0.5f );

        if( m_scaleAreas[curArea].height < minSize.height || m_scaleAreas[curArea].width < minSize.width )
        {
          m_scaleFactorWidth = 0.0f;
          return false;
        }

        m_scaleAreas[curArea].x = (int) floor( (float) m_areas[curArea].x * m_scaleFactorWidth + 0.5f );
        m_scaleAreas[curArea].y = (int) floor( (float) m_areas[curArea].y * m_scaleFactorHeight + 0.5f );
        m_scaleWeights[curArea] = (float) m_weights[curArea] / (float) ( ( m_scaleAreas[curArea].width ) * ( m_scaleAreas[curArea].height ) );
      }
    }
    else
    {
      m_scaleFactorWidth = m_scaleFactorHeight = 1.0f;
      for ( int curArea = 0; curArea < m_numAreas; curArea++ )
      {
        m_scaleAreas[curArea] = m_areas[curArea];
        m_scaleWeights[curArea] = (float) m_weights[curArea] / (float) ( ( m_areas[curArea].width ) * ( m_areas[curArea].height ) );
      }
    }
  }

  if( m_scaleFactorWidth == 0.0f )
    return false;

  for ( int curArea = 0; curArea < m_numAreas; curArea++ )
  {
    *result += (float) getSum( image,
                               Rect( m_scaleAreas[curArea].x, m_scaleAreas[curArea].y, m_scaleAreas[curArea].width, m_scaleAreas[curArea].height ) )
        * m_scaleWeights[curArea];
  }

  /*
   if( image->getUseVariance() )
   {
   float variance = (float) image->getVariance( ROI );
   *result /= variance;
   }
   */

  m_response = *result;

  return true;
}

float FeatureHaar::getSum( const Mat& image, Rect imageROI )
{
  // left upper Origin
  int OriginX = imageROI.x;
  int OriginY = imageROI.y;

  // Check and fix width and height
  int Width = imageROI.width;
  int Height = imageROI.height;

  if( OriginX + Width >= image.cols - 1 )
    Width = ( image.cols - 1 ) - OriginX;
  if( OriginY + Height >= image.rows - 1 )
    Height = ( image.rows - 1 ) - OriginY;

  int value = image.at<int>( OriginY + Height, OriginX + Width ) + image.at<int>( OriginY, OriginX ) - image.at<int>( OriginY, OriginX + Width )
      - image.at<int>( OriginY + Height, OriginX );

  return value;
}

void FeatureHaar::getInitialDistribution( EstimatedGaussDistribution* distribution )
{
  distribution->setValues( m_initMean, m_initSigma );
}
WeakClassifierHaarFeature::WeakClassifierHaarFeature( Size patchSize )
{
  m_feature = new FeatureHaar( patchSize );
  generateRandomClassifier();
  m_feature->getInitialDistribution( (EstimatedGaussDistribution*) m_classifier->getDistribution( -1 ) );
  m_feature->getInitialDistribution( (EstimatedGaussDistribution*) m_classifier->getDistribution( 1 ) );
}

void WeakClassifierHaarFeature::resetPosDist()
{
  m_feature->getInitialDistribution( (EstimatedGaussDistribution*) m_classifier->getDistribution( 1 ) );
  m_feature->getInitialDistribution( (EstimatedGaussDistribution*) m_classifier->getDistribution( -1 ) );
}

WeakClassifierHaarFeature::~WeakClassifierHaarFeature()
{
  delete m_classifier;
  delete m_feature;

}

void WeakClassifierHaarFeature::generateRandomClassifier()
{
  m_classifier = new ClassifierThreshold();
}

bool WeakClassifierHaarFeature::update( const Mat& image, Rect ROI, int target )
{
  float value;

  bool valid = m_feature->eval( image, ROI, &value );
  if( !valid )
    return true;

  m_classifier->update( value, target );
  return ( m_classifier->eval( value ) != target );
}

int WeakClassifierHaarFeature::eval( const Mat& image, Rect ROI )
{
  float value;
  bool valid = m_feature->eval( image, ROI, &value );
  if( !valid )
    return 0;

  return m_classifier->eval( value );
}

float WeakClassifierHaarFeature::getValue( const Mat& image, Rect ROI )
{
  float value;
  bool valid = m_feature->eval( image, ROI, &value );
  if( !valid )
    return 0;

  return value;
}

EstimatedGaussDistribution*
WeakClassifierHaarFeature::getPosDistribution()
{
  return static_cast<EstimatedGaussDistribution*>( m_classifier->getDistribution( 1 ) );
}

EstimatedGaussDistribution*
WeakClassifierHaarFeature::getNegDistribution()
{
  return static_cast<EstimatedGaussDistribution*>( m_classifier->getDistribution( -1 ) );
}

Detector::Detector( StrongClassifierDirectSelection* classifier ) :
    m_sizeDetections( 0 )
{
  this->m_classifier = classifier;

  m_sizeConfidences = 0;
  m_maxConfidence = -FLT_MAX;
  m_numDetections = 0;
  m_idxBestDetection = -1;
}

Detector::~Detector()
{
}

void Detector::prepareConfidencesMemory( int numPatches )
{
  if( numPatches <= m_sizeConfidences )
    return;

  m_sizeConfidences = numPatches;
  m_confidences.resize( numPatches );
}

void Detector::prepareDetectionsMemory( int numDetections )
{
  if( numDetections <= m_sizeDetections )
    return;

  m_sizeDetections = numDetections;
  m_idxDetections.resize( numDetections );
}

void Detector::classifySmooth( const std::vector<Mat>& images, float minMargin )
{
  int numPatches = images.size();

  prepareConfidencesMemory( numPatches );

  m_numDetections = 0;
  m_idxBestDetection = -1;
  m_maxConfidence = -FLT_MAX;
  int numBaseClassifiers = m_classifier->getNumBaseClassifier();

  //compute grid
  //TODO 0.99 overlap from params
  Size patchSz = m_classifier->getPatchSize();
  int stepCol = (int) floor( ( 1.0f - 0.99f ) * (float) patchSz.width + 0.5f );
  int stepRow = (int) floor( ( 1.0f - 0.99f ) * (float) patchSz.height + 0.5f );
  if( stepCol <= 0 )
    stepCol = 1;
  if( stepRow <= 0 )
    stepRow = 1;

  Size patchGrid;
  Rect ROI = m_classifier->getROI();
  patchGrid.height = ( (int) ( (float) ( ROI.height - patchSz.height ) / stepRow ) + 1 );
  patchGrid.width = ( (int) ( (float) ( ROI.width - patchSz.width ) / stepCol ) + 1 );

  if( ( patchGrid.width != m_confMatrix.cols ) || ( patchGrid.height != m_confMatrix.rows ) )
  {
    m_confMatrix.create( patchGrid.height, patchGrid.width );
    m_confMatrixSmooth.create( patchGrid.height, patchGrid.width );
    m_confImageDisplay.create( patchGrid.height, patchGrid.width );
  }

  int curPatch = 0;
  // Eval and filter
  for ( int row = 0; row < patchGrid.height; row++ )
  {
    for ( int col = 0; col < patchGrid.width; col++ )
    {
      //int returnedInLayer;
      Size sz;
      Point offset;
      images[curPatch].locateROI( sz, offset );
      sz.width = images[curPatch].cols;
      sz.height = images[curPatch].rows;
      m_confidences[curPatch] = m_classifier->eval( images[curPatch], Rect( offset.x, offset.y, sz.width, sz.height ) );

      // fill matrix
      m_confMatrix( row, col ) = m_confidences[curPatch];
      curPatch++;
    }
  }

  // Filter
  //cv::GaussianBlur(m_confMatrix,m_confMatrixSmooth,cv::Size(3,3),0.8);
  cv::GaussianBlur( m_confMatrix, m_confMatrixSmooth, cv::Size( 3, 3 ), 0 );

  // Make display friendly
  double min_val, max_val;
  cv::minMaxLoc( m_confMatrixSmooth, &min_val, &max_val );
  for ( int y = 0; y < m_confImageDisplay.rows; y++ )
  {
    unsigned char* pConfImg = m_confImageDisplay[y];
    const float* pConfData = m_confMatrixSmooth[y];
    for ( int x = 0; x < m_confImageDisplay.cols; x++, pConfImg++, pConfData++ )
    {
      *pConfImg = static_cast<unsigned char>( 255.0 * ( *pConfData - min_val ) / ( max_val - min_val ) );
    }
  }

  // Get best detection
  curPatch = 0;
  for ( int row = 0; row < patchGrid.height; row++ )
  {
    for ( int col = 0; col < patchGrid.width; col++ )
    {
      // fill matrix
      m_confidences[curPatch] = m_confMatrixSmooth( row, col );

      if( m_confidences[curPatch] > m_maxConfidence )
      {
        m_maxConfidence = m_confidences[curPatch];
        m_idxBestDetection = curPatch;
      }
      if( m_confidences[curPatch] > minMargin )
      {
        m_numDetections++;
      }
      curPatch++;
    }
  }

  prepareDetectionsMemory( m_numDetections );
  int curDetection = -1;
  for ( int curPatch = 0; curPatch < numPatches; curPatch++ )
  {
    if( m_confidences[curPatch] > minMargin )
      m_idxDetections[++curDetection] = curPatch;
  }
}

int Detector::getNumDetections()
{
  return m_numDetections;
}

float Detector::getConfidence( int patchIdx )
{
  return m_confidences[patchIdx];
}

float Detector::getConfidenceOfDetection( int detectionIdx )
{
  return m_confidences[getPatchIdxOfDetection( detectionIdx )];
}

int Detector::getPatchIdxOfBestDetection()
{
  return m_idxBestDetection;
}

int Detector::getPatchIdxOfDetection( int detectionIdx )
{
  return m_idxDetections[detectionIdx];
}

ClassifierThreshold::ClassifierThreshold()
{
  m_posSamples = new EstimatedGaussDistribution();
  m_negSamples = new EstimatedGaussDistribution();
  m_threshold = 0.0f;
  m_parity = 0;
}

ClassifierThreshold::~ClassifierThreshold()
{
  if( m_posSamples != NULL )
    delete m_posSamples;
  if( m_negSamples != NULL )
    delete m_negSamples;
}

void*
ClassifierThreshold::getDistribution( int target )
{
  if( target == 1 )
    return m_posSamples;
  else
    return m_negSamples;
}

void ClassifierThreshold::update( float value, int target )
{
  //update distribution
  if( target == 1 )
    m_posSamples->update( value );
  else
    m_negSamples->update( value );

  //adapt threshold and parity
  m_threshold = ( m_posSamples->getMean() + m_negSamples->getMean() ) / 2.0f;
  m_parity = ( m_posSamples->getMean() > m_negSamples->getMean() ) ? 1 : -1;
}

int ClassifierThreshold::eval( float value )
{
  return ( ( ( m_parity * ( value - m_threshold ) ) > 0 ) ? 1 : -1 );
}

EstimatedGaussDistribution::EstimatedGaussDistribution()
{
  m_mean = 0;
  m_sigma = 1;
  this->m_P_mean = 1000;
  this->m_R_mean = 0.01f;
  this->m_P_sigma = 1000;
  this->m_R_sigma = 0.01f;
}

EstimatedGaussDistribution::EstimatedGaussDistribution( float P_mean, float R_mean, float P_sigma, float R_sigma )
{
  m_mean = 0;
  m_sigma = 1;
  this->m_P_mean = P_mean;
  this->m_R_mean = R_mean;
  this->m_P_sigma = P_sigma;
  this->m_R_sigma = R_sigma;
}

EstimatedGaussDistribution::~EstimatedGaussDistribution()
{
}

void EstimatedGaussDistribution::update( float value )
{
  //update distribution (mean and sigma) using a kalman filter for each

  float K;
  float minFactor = 0.001f;

  //mean

  K = m_P_mean / ( m_P_mean + m_R_mean );
  if( K < minFactor )
    K = minFactor;

  m_mean = K * value + ( 1.0f - K ) * m_mean;
  m_P_mean = m_P_mean * m_R_mean / ( m_P_mean + m_R_mean );

  K = m_P_sigma / ( m_P_sigma + m_R_sigma );
  if( K < minFactor )
    K = minFactor;

  float tmp_sigma = K * ( m_mean - value ) * ( m_mean - value ) + ( 1.0f - K ) * m_sigma * m_sigma;
  m_P_sigma = m_P_sigma * m_R_mean / ( m_P_sigma + m_R_sigma );

  m_sigma = static_cast<float>( sqrt( tmp_sigma ) );
  if( m_sigma <= 1.0f )
    m_sigma = 1.0f;

}

void EstimatedGaussDistribution::setValues( float mean, float sigma )
{
  this->m_mean = mean;
  this->m_sigma = sigma;
}

float EstimatedGaussDistribution::getMean()
{
  return m_mean;
}

float EstimatedGaussDistribution::getSigma()
{
  return m_sigma;
}

} /* namespace cv */
