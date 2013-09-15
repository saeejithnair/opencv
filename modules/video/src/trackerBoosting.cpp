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
  iterationInit = 50;
  featureSetNumFeatures = ( numClassifiers * 10 ) + iterationInit;
}

void TrackerBoosting::Params::read( const cv::FileNode& fn )
{
  numClassifiers = fn["numClassifiers"];
  samplerOverlap = fn["overlap"];
  samplerSearchFactor = fn["samplerSearchFactor"];
  iterationInit = fn["iterationInit"];
  samplerSearchFactor = fn["searchFactor"];
}

void TrackerBoosting::Params::write( cv::FileStorage& fs ) const
{
  fs << "numClassifiers" << numClassifiers;
  fs << "overlap" << samplerOverlap;
  fs << "searchFactor" << samplerSearchFactor;
  fs << "iterationInit" << iterationInit;
  fs << "samplerSearchFactor" << samplerSearchFactor;
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
  Mat_<int> intImage;
  Mat_<double> intSqImage;
  Mat image_;
  cvtColor( image, image_, CV_RGB2GRAY );
  integral( image_, intImage, intSqImage, CV_32S );
  TrackerSamplerCS::Params CSparameters;
  CSparameters.overlap = params.samplerOverlap;
  CSparameters.searchFactor = params.samplerSearchFactor;

  Ptr<TrackerSamplerAlgorithm> CSSampler = new TrackerSamplerCS( CSparameters );

  if( !sampler->addTrackerSamplerAlgorithm( CSSampler ) )
    return false;

  Ptr<TrackerSamplerCS>( CSSampler )->setMode( TrackerSamplerCS::MODE_POSITIVE );
  sampler->sampling( intImage, boundingBox );
  const std::vector<Mat> posSamples = sampler->getSamples();

  Ptr<TrackerSamplerCS>( CSSampler )->setMode( TrackerSamplerCS::MODE_NEGATIVE );
  sampler->sampling( intImage, boundingBox );
  const std::vector<Mat> negSamples = sampler->getSamples();

  if( posSamples.empty() || negSamples.empty() )
    return false;

  Rect ROI = Ptr<TrackerSamplerCS>( CSSampler )->getROI();

  //compute HAAR features
  TrackerFeatureHAAR::Params HAARparameters;
  HAARparameters.numFeatures = params.featureSetNumFeatures;
  HAARparameters.isIntegral = true;
  HAARparameters.rectSize = Size( boundingBox.width, boundingBox.height );
  Ptr<TrackerFeature> trackerFeature = new TrackerFeatureHAAR( HAARparameters );
  const std::vector<std::pair<float, float> > meanSigmaPair = ( (Ptr<TrackerFeatureHAAR> ) trackerFeature )->getMeanSigmaPairs();
  if( !featureSet->addTrackerFeature( trackerFeature ) )
    return false;

  featureSet->extraction( posSamples );
  const std::vector<Mat> posResponse = featureSet->getResponses();
  featureSet->extraction( negSamples );
  const std::vector<Mat> negResponse = featureSet->getResponses();

  //Model
  model = new TrackerBoostingModel( boundingBox );
  Ptr<TrackerStateEstimatorAdaBoosting> stateEstimator = new TrackerStateEstimatorAdaBoosting( params.numClassifiers, params.iterationInit,
                                                                                               params.featureSetNumFeatures,
                                                                                               Size( boundingBox.width, boundingBox.height ), ROI,
                                                                                               meanSigmaPair );
  model->setTrackerStateEstimator( stateEstimator );

  //Run model estimation and update
  for ( int i = 0; i < params.iterationInit; i++ )
  {
    //compute temp features
    TrackerFeatureHAAR::Params HAARparameters2;
    HAARparameters2.numFeatures = ( posSamples.size() + negSamples.size() );
    HAARparameters2.isIntegral = true;
    HAARparameters2.rectSize = Size( boundingBox.width, boundingBox.height );
    Ptr<TrackerFeatureHAAR> trackerFeature2 = new TrackerFeatureHAAR( HAARparameters2 );
    const std::vector<std::pair<float, float> > meanSigmaPair2 = ( (Ptr<TrackerFeatureHAAR> ) trackerFeature2 )->getMeanSigmaPairs();

    stateEstimator->setMeanSigmaPair( meanSigmaPair2 );
    ( (Ptr<TrackerBoostingModel> ) model )->setMode( TrackerBoostingModel::MODE_NEGATIVE, negSamples );
    model->modelEstimation( negResponse );
    ( (Ptr<TrackerBoostingModel> ) model )->setMode( TrackerBoostingModel::MODE_POSITIVE, posSamples );
    model->modelEstimation( posResponse );
    model->modelUpdate();

    //TODO get replaced classifier and change the features
    std::vector<int> replacedClassifier = stateEstimator->computeReplacedClassifier();
    std::vector<int> swappedClassified = stateEstimator->computeSwappedClassifier();
    for ( size_t j = 0; j < replacedClassifier.size(); j++ )
    {
      if( replacedClassifier[j] != -1 && swappedClassified[j] != -1 )
      {
        ( Ptr<TrackerFeatureHAAR>( trackerFeature ) )->swapFeature( replacedClassifier[j], swappedClassified[j] );
        ( Ptr<TrackerFeatureHAAR>( trackerFeature ) )->swapFeature( swappedClassified[j], trackerFeature2->getFeatureAt( j ) );
      }
    }
  }

  return true;
}

bool TrackerBoosting::updateImpl( const Mat& image, Rect& boundingBox )
{
  Mat_<int> intImage;
  Mat_<double> intSqImage;
  Mat image_;
  cvtColor( image, image_, CV_RGB2GRAY );
  integral( image_, intImage, intSqImage, CV_32S );
  //get the last location [AAM] X(k-1)
  Ptr<TrackerTargetState> lastLocation = model->getLastTargetState();
  Rect lastBoundingBox( lastLocation->getTargetPosition().x, lastLocation->getTargetPosition().y, lastLocation->getTargetWidth(),
                        lastLocation->getTargetHeight() );

  //sampling new frame based on last location
  ( (Ptr<TrackerSamplerCS> ) sampler->getSamplers().at( 0 ).second )->setMode( TrackerSamplerCS::MODE_CLASSIFY );
  sampler->sampling( intImage, lastBoundingBox );
  const std::vector<Mat> detectSamples = sampler->getSamples();
  Rect ROI = ( (Ptr<TrackerSamplerCS> ) sampler->getSamplers().at( 0 ).second )->getROI();

  if( detectSamples.empty() )
    return false;

  /*//TODO debug samples
   Mat f;
   image.copyTo( f );

   for ( size_t i = 0; i < detectSamples.size(); i = i + 10 )
   {
   Size sz;
   Point off;
   detectSamples.at( i ).locateROI( sz, off );
   rectangle( f, Rect( off.x, off.y, detectSamples.at( i ).cols, detectSamples.at( i ).rows ), Scalar( 255, 0, 0 ), 1 );
   }*/

  //extract features from new samples
  //featureSet->extraction( detectSamples );
  //const std::vector<Mat> response = featureSet->getResponses();
  std::vector<Mat> responses;
  Mat response;

  std::vector<int> classifiers = ( (Ptr<TrackerStateEstimatorAdaBoosting> ) model->getTrackerStateEstimator() )->computeSelectedWeakClassifier();
  Ptr<TrackerFeatureHAAR> extractor = featureSet->getTrackerFeature()[0].second;
  extractor->extractSelected( classifiers, detectSamples, response );
  responses.push_back( response );

  //predict new location
  ConfidenceMap cmap;
  ( (Ptr<TrackerBoostingModel> ) model )->setMode( TrackerBoostingModel::MODE_CLASSIFY, detectSamples );
  ( (Ptr<TrackerBoostingModel> ) model )->responseToConfidenceMap( responses, cmap );
  ( (Ptr<TrackerStateEstimatorAdaBoosting> ) model->getTrackerStateEstimator() )->setCurrentConfidenceMap( cmap );
  ( (Ptr<TrackerStateEstimatorAdaBoosting> ) model->getTrackerStateEstimator() )->setSampleROI( ROI );

  if( !model->runStateEstimator() )
  {
    return false;
  }

  Ptr<TrackerTargetState> currentState = model->getLastTargetState();
  boundingBox = Rect( currentState->getTargetPosition().x, currentState->getTargetPosition().y, currentState->getTargetWidth(),
                      currentState->getTargetHeight() );

  /*//TODO debug
   rectangle( f, lastBoundingBox, Scalar( 0, 255, 0 ), 1 );
   rectangle( f, boundingBox, Scalar( 0, 0, 255 ), 1 );
   imshow( "f", f );
   //waitKey( 0 );*/

  //sampling new frame based on new location
  //Positive sampling
  ( (Ptr<TrackerSamplerCS> ) sampler->getSamplers().at( 0 ).second )->setMode( TrackerSamplerCS::MODE_POSITIVE );
  sampler->sampling( intImage, boundingBox );
  const std::vector<Mat> posSamples = sampler->getSamples();

  //Negative sampling
  ( (Ptr<TrackerSamplerCS> ) sampler->getSamplers().at( 0 ).second )->setMode( TrackerSamplerCS::MODE_NEGATIVE );
  sampler->sampling( intImage, lastBoundingBox );
  const std::vector<Mat> negSamples = sampler->getSamples();

  if( posSamples.empty() || negSamples.empty() )
    return false;

  //extract features
  featureSet->extraction( posSamples );
  const std::vector<Mat> posResponse = featureSet->getResponses();

  featureSet->extraction( negSamples );
  const std::vector<Mat> negResponse = featureSet->getResponses();

  //compute temp features
  TrackerFeatureHAAR::Params HAARparameters2;
  HAARparameters2.numFeatures = ( posSamples.size() + negSamples.size() );
  HAARparameters2.isIntegral = true;
  HAARparameters2.rectSize = Size( boundingBox.width, boundingBox.height );
  Ptr<TrackerFeatureHAAR> trackerFeature2 = new TrackerFeatureHAAR( HAARparameters2 );
  const std::vector<std::pair<float, float> > meanSigmaPair2 = trackerFeature2->getMeanSigmaPairs();
  ( (Ptr<TrackerStateEstimatorAdaBoosting> ) model->getTrackerStateEstimator() )->setMeanSigmaPair( meanSigmaPair2 );

  //model estimate
  ( (Ptr<TrackerBoostingModel> ) model )->setMode( TrackerBoostingModel::MODE_NEGATIVE, negSamples );
  model->modelEstimation( negResponse );
  ( (Ptr<TrackerBoostingModel> ) model )->setMode( TrackerBoostingModel::MODE_POSITIVE, posSamples );
  model->modelEstimation( posResponse );

  //model update
  model->modelUpdate();

  //TODO get replaced classifier and change the features
  std::vector<int> replacedClassifier = ( (Ptr<TrackerStateEstimatorAdaBoosting> ) model->getTrackerStateEstimator() )->computeReplacedClassifier();
  std::vector<int> swappedClassified = ( (Ptr<TrackerStateEstimatorAdaBoosting> ) model->getTrackerStateEstimator() )->computeSwappedClassifier();
  for ( size_t j = 0; j < replacedClassifier.size(); j++ )
  {
    if( replacedClassifier[j] != -1 && swappedClassified[j] != -1 )
    {
      ( Ptr<TrackerFeatureHAAR>( featureSet->getTrackerFeature().at(0).second ) )->swapFeature( replacedClassifier[j], swappedClassified[j] );
      ( Ptr<TrackerFeatureHAAR>( featureSet->getTrackerFeature().at(0).second ) )->swapFeature( swappedClassified[j], trackerFeature2->getFeatureAt( j ) );
    }
  }

  return true;

}

} /* namespace cv */
