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

BaseClassifier::BaseClassifier( int numWeakClassifier, int iterationInit, Size patchSize )
{

}

BaseClassifier::BaseClassifier( int numWeakClassifier, int iterationInit, WeakClassifier** weakClassifier )
{

}

void BaseClassifier::trainClassifier( Mat response, Rect ROI, int target, float importance, bool* errorMask )
{

}

int BaseClassifier::selectBestClassifier( bool* errorMask, float importance, std::vector<float> & errors )
{
  return 0;
}

int BaseClassifier::replaceWeakestClassifier( const std::vector<float> & errors, Size patchSize )
{
  return 0;
}
void BaseClassifier::replaceClassifierStatistic( int sourceIndex, int targetIndex )
{

}

WeakClassifier** BaseClassifier::getReferenceWeakClassifier()
{
  return weakClassifier;
}

int BaseClassifier::getIdxOfNewWeakClassifier()
{
  return m_idxOfNewWeakClassifier;
}

BaseClassifier::~BaseClassifier()
{

}

} /* namespace cv */
