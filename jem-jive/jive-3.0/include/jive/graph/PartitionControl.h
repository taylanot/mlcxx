
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
 *
 *  Commercial License Usage
 *
 *  This file may be used under the terms of a commercial license
 *  provided with the software, or under the terms contained in a written
 *  agreement between you and DRG. For more information contact DRG at
 *  http://www.dynaflow.com.
 *
 *  GNU Lesser General Public License Usage
 *
 *  Alternatively, this file may be used under the terms of the GNU
 *  Lesser General Public License version 2.1 or version 3 as published
 *  by the Free Software Foundation and appearing in the file
 *  LICENSE.LGPLv21 and LICENSE.LGPLv3 included in the packaging of this
 *  file. Please review the following information to ensure the GNU
 *  Lesser General Public License requirements will be met:
 *  https://www.gnu.org/licenses/lgpl.html and
 *  http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_GRAPH_PARTITIONCONTROL_H
#define JIVE_GRAPH_PARTITIONCONTROL_H

#include <jem/base/Object.h>
#include <jive/Array.h>


JIVE_BEGIN_PACKAGE( graph )


//-----------------------------------------------------------------------
//   class PartitionControl
//-----------------------------------------------------------------------


class PartitionControl : public Object
{
 public:

  typedef PartitionControl  Self;
  typedef Object            Super;

  static const double       DEFAULT_UB;


  explicit                  PartitionControl

    ( const IdxVector&        partWeights,
      double                  ub = DEFAULT_UB );

                            PartitionControl

    ( idx_t                   partCount,
      idx_t                   totalWeight,
      double                  ub = DEFAULT_UB );

                            PartitionControl

    ( const Vector&           partWeights,
      idx_t                   totalWeight,
      double                  ub = DEFAULT_UB );

  inline idx_t              partCount           ()  const;

  inline idx_t              targetWeight

    ( idx_t                   ipart )               const;

  inline idx_t              minWeight

    ( idx_t                   ipart )               const;

  inline idx_t              maxWeight

    ( idx_t                   ipart )               const;

  inline double             unbalance           ()  const;


 protected:

  virtual                  ~PartitionControl    ();


 private:

  void                      initMinMaxWeights_  ();


 private:

  IdxVector                 targetWeights_;
  IdxVector                 minWeights_;
  IdxVector                 maxWeights_;
  double                    unbalance_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   partCount
//-----------------------------------------------------------------------


inline idx_t PartitionControl::partCount () const
{
  return targetWeights_.size ();
}


//-----------------------------------------------------------------------
//   targetWeight
//-----------------------------------------------------------------------


inline idx_t PartitionControl::targetWeight ( idx_t ipart ) const
{
  return targetWeights_[ipart];
}


//-----------------------------------------------------------------------
//   minWeight
//-----------------------------------------------------------------------


inline idx_t PartitionControl::minWeight ( idx_t ipart ) const
{
  return minWeights_[ipart];
}


//-----------------------------------------------------------------------
//   maxWeight
//-----------------------------------------------------------------------


inline idx_t PartitionControl::maxWeight ( idx_t ipart ) const
{
  return maxWeights_[ipart];
}


//-----------------------------------------------------------------------
//   unbalance
//-----------------------------------------------------------------------


inline double PartitionControl::unbalance () const
{
  return unbalance_;
}


JIVE_END_PACKAGE( graph )

#endif
