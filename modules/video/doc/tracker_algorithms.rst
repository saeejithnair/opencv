Tracker Algorithms
==================

.. highlight:: cpp

Two algorithms will be implemented soon, the first is MIL (Multiple Instance Learning) [MIL]_ and second is Online Boosting [OLB]_.

.. [MIL] B Babenko, M-H Yang, and S Belongie, Visual Tracking with Online Multiple Instance Learning, In CVPR, 2009

.. [OLB] H Grabner, M Grabner, and H Bischof, Real-time tracking via on-line boosting, In Proc. BMVC, volume 1, pages 47– 56, 2006

TrackerBoosting
---------------

This is a real-time object tracking based on a novel on-line version of the AdaBoost algorithm.
The classifier uses the surrounding background as negative examples in update step to avoid the drifting problem.

.. ocv:class:: TrackerBoosting

Implementation of TrackerBoosting from :ocv:class:`Tracker`::

   class CV_EXPORTS_W TrackerBoosting : public Tracker
   {
    public:

     TrackerBoosting( const TrackerBoosting::Params &parameters = TrackerBoosting::Params() );

     virtual ~TrackerBoosting();

     void read( const FileNode& fn );
     void write( FileStorage& fs ) const;


   };

TrackerBoosting::Params
------------------

.. ocv:struct:: TrackerBoosting::Params

List of BOOSTING parameters::

   struct CV_EXPORTS Params
   {
    Params();
    int numClassifiers;  //the number of classifiers to use in a OnlineBoosting algorithm
    float samplerOverlap;  //search region parameters to use in a OnlineBoosting algorithm
    float samplerSearchFactor;  // search region parameters to use in a OnlineBoosting algorithm
    int iterationInit;  //the initial iterations
    int featureSetNumFeatures;  // #features

    void read( const FileNode& fn );
    void write( FileStorage& fs ) const;
   };

TrackerBoosting::TrackerBoosting
----------------------

Constructor

.. ocv:function:: bool TrackerBoosting::TrackerBoosting( const TrackerBoosting::Params &parameters = TrackerBoosting::Params() )

    :param parameters: BOOSTING parameters :ocv:struct:`TrackerBoosting::Params`
