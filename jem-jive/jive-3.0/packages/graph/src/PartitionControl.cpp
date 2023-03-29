
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


#include <jem/base/assert.h>
#include <jem/base/IllegalArgumentException.h>
#include <jive/graph/PartitionControl.h>


JIVE_BEGIN_PACKAGE( graph )


using jem::IllegalArgumentException;


//=======================================================================
//   class PartitionControl
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const double  PartitionControl::DEFAULT_UB = 0.05;


//-----------------------------------------------------------------------
//   constructors & destructors
//-----------------------------------------------------------------------


PartitionControl::PartitionControl

  ( const IdxVector&  partWeights,
    double            ub ) :

    targetWeights_ ( partWeights ),
    unbalance_     (          ub )

{
  JEM_PRECHECK ( ub >= 0.0 );

  const idx_t  partCount = partWeights.size ();


  for ( idx_t ipart = 0; ipart < partCount; ipart++ )
  {
    if ( partWeights[ipart] < 0 )
    {
      throw IllegalArgumentException (
        JEM_FUNC,
        String::format (
          "invalid partition weight: %d", partWeights[ipart]
        )
      );
    }
  }

  initMinMaxWeights_ ();
}


PartitionControl::PartitionControl

  ( idx_t   partCount,
    idx_t   totalWeight,
    double  ub ) :

    unbalance_ ( ub )

{
  JEM_PRECHECK ( partCount   >= 0 &&
                 totalWeight >= 0 &&
                 ub          >= 0.0 );

  targetWeights_.resize ( partCount );

  for ( idx_t ipart = 0; ipart < partCount; ipart++ )
  {
    idx_t  pwgt  = totalWeight / (partCount - ipart);
    totalWeight -= pwgt;

    targetWeights_[ipart] = pwgt;
  }

  initMinMaxWeights_ ();
}


PartitionControl::PartitionControl

  ( const Vector&  partWeights,
    idx_t          totalWeight,
    double         ub ) :

    unbalance_ ( ub )

{
  JEM_PRECHECK ( totalWeight >= 0 && ub >= 0.0 );

  const idx_t  partCount = partWeights.size ();

  double       scale;
  double       fwgt;
  idx_t        pwgt;
  idx_t        ipart;


  targetWeights_.resize ( partCount );

  scale = 1.0;

  for ( ipart = 0; ipart < partCount && totalWeight > 0; ipart++ )
  {
    fwgt = partWeights[ipart];

    if ( fwgt < 0.0 || fwgt > 1.0 )
    {
      throw IllegalArgumentException (
        JEM_FUNC,
        String::format (
          "invalid partition weight: %g", fwgt
        )
      );
    }

    if ( ipart == (partCount - 1) )
    {
      pwgt = totalWeight;
    }
    else
    {
      if ( scale <= 0.0 )
      {
        throw IllegalArgumentException (
          JEM_FUNC,
          "invalid partition weights (sum larger than 1.0)"
        );
      }

      pwgt = (idx_t) ((fwgt / scale) * (double) totalWeight + 0.5);
    }

    targetWeights_[ipart] = pwgt;

    totalWeight -= pwgt;
    scale       -= fwgt;
  }

  for ( ; ipart < partCount; ipart++ )
  {
    targetWeights_[ipart] = 0;
  }

  initMinMaxWeights_ ();
}


PartitionControl::~PartitionControl ()
{}


//-----------------------------------------------------------------------
//   initMinMaxWeights_
//-----------------------------------------------------------------------


void PartitionControl::initMinMaxWeights_ ()
{
  const idx_t  partCount = this->partCount ();

  double       fmin, fmax;
  idx_t        wtarget, wmin, wmax;


  minWeights_.resize ( partCount );
  maxWeights_.resize ( partCount );

  fmax = (1.0 + unbalance_);
  fmin = 1.0 / (1.0 + jem::max( 0.2, 2.0 * unbalance_ ));

  for ( idx_t ipart = 0; ipart < partCount; ipart++ )
  {
    wtarget = targetWeights_[ipart];

    wmin = (idx_t) ((double) wtarget * fmin + 0.5);
    wmax = (idx_t) ((double) wtarget * fmax + 0.5);

    if ( wmin == 0 )
    {
      wmin = 1;
    }

    if ( wmin > wtarget)
    {
      wmin = wtarget;
    }

    if ( wmax < wtarget)
    {
      wmax = wtarget;
    }

    minWeights_[ipart] = wmin;
    maxWeights_[ipart] = wmax;
  }
}


JIVE_END_PACKAGE( graph )
