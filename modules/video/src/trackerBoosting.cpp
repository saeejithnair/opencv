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
#include "trackerBoostingModel.hpp"

namespace cv
{

/*
 *  TrackerBoosting
 */

/*
 * Parameters
 */
TrackerBoosting::Params::Params()
{
  numClassifiers = 100;
  samplerOverlap = 0.99f;
  samplerSearchFactor = 2;
  featureSetNumFeatures = 250;
}

void TrackerBoosting::Params::read( const cv::FileNode& fn )
{
  numClassifiers = fn["numClassifiers"];
  samplerOverlap = fn["overlap"];
  samplerSearchFactor = fn["searchFactor"];
}

void TrackerBoosting::Params::write( cv::FileStorage& fs ) const
{
  fs << "numClassifiers" << numClassifiers;
  fs << "overlap" << samplerOverlap;
  fs << "searchFactor" << samplerSearchFactor;
}

/*
 * Constructor
 */
TrackerBoosting::TrackerBoosting( const TrackerBoosting::Params &parameters ) :
    params( parameters )
{
  initialized = false;
}

/*
 * Destructor
 */
TrackerBoosting::~TrackerBoosting()
{

}

void TrackerBoosting::read( const cv::FileNode& fn )
{
  params.read( fn );
}

void TrackerBoosting::write( cv::FileStorage& fs ) const
{
  params.write( fs );
}

bool TrackerBoosting::initImpl( const Mat& image, const Rect& boundingBox )
{
  //sampling
  TrackerSamplerCS::Params CSparameters;
  CSparameters.overlap = params.samplerOverlap;
  CSparameters.searchFactor = params.samplerSearchFactor;

  Ptr<TrackerSamplerAlgorithm> CSSampler = new TrackerSamplerCS( CSparameters );

  if( !sampler->addTrackerSamplerAlgorithm( CSSampler ) )
    return false;

  Ptr<TrackerSamplerCS>( CSSampler )->setMode( TrackerSamplerCS::MODE_INIT );
  sampler->sampling( image, boundingBox );
  std::vector<Mat> samples = sampler->getSamples();

  if( samples.empty() )
    return false;

  //compute HAAR features
  TrackerFeatureHAAR::Params HAARparameters;
  HAARparameters.numFeatures = params.featureSetNumFeatures;
  HAARparameters.rectSize = Size( boundingBox.width, boundingBox.height );
  Ptr<TrackerFeature> trackerFeature = new TrackerFeatureHAAR( HAARparameters );

  if( !featureSet->addTrackerFeature( trackerFeature ) )
    return false;
  featureSet->extraction( samples );
  const std::vector<Mat> response = featureSet->getResponses();

  //Model
  model = new TrackerBoostingModel( boundingBox );
  Ptr<TrackerStateEstimatorAdaBoosting> stateEstimator = new TrackerStateEstimatorAdaBoosting( params.numClassifiers,
                                                                                               Size( boundingBox.width, boundingBox.height ) );
  model->setTrackerStateEstimator( stateEstimator );

  //TODO Run model estimation and update
  model->modelEstimation( response );
  model->modelUpdate();
  return true;
}

bool TrackerBoosting::updateImpl( const Mat& image, Rect& /*boundingBox*/)
{
  //get the last location [AAM] X(k-1)
  Ptr<TrackerTargetState> lastLocation = model->getLastTargetState();
  Rect lastBoundingBox( lastLocation->getTargetPosition().x, lastLocation->getTargetPosition().y, lastLocation->getTargetWidth(),
                        lastLocation->getTargetHeight() );

  //sampling new frame based on last location
  ( (Ptr<TrackerSamplerCS> ) sampler->getSamplers().at( 0 ).second )->setMode( TrackerSamplerCS::MODE_TRACK );
  sampler->sampling( image, lastBoundingBox );
  std::vector<Mat> detectSamples = sampler->getSamples();
  if( detectSamples.empty() )
    return false;

  //extract features from new samples
  featureSet->extraction( detectSamples );
  std::vector<Mat> response = featureSet->getResponses();


  return true;

}

} /* namespace cv */
