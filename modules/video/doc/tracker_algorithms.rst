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
   
   
TODO
----

* TrackerBoosting
* porting of boosting method from original MIL
