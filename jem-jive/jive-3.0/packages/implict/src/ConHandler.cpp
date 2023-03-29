
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


#include <jem/base/array/select.h>
#include <jem/base/array/operators.h>
#include <jive/util/Constraints.h>
#include <jive/implict/ConHandler.h>


JIVE_BEGIN_PACKAGE( implict )


//=======================================================================
//   class ConHandler
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


ConHandler::ConHandler ()
{}


//-----------------------------------------------------------------------
//   saveConstraints
//-----------------------------------------------------------------------


void ConHandler::saveConstraints ( const Constraints& cons )
{
  const idx_t  slaveCount = cons.slaveDofCount ();

  BoolVector   rmask ( slaveCount );

  idx_t        i, j;


  slaveDofs_.resize ( slaveCount );

  cons.getSlaveDofs  (        slaveDofs_ );
  cons.getRvalueMask ( rmask, slaveDofs_ );

  // Re-order slave dofs so that the ones with a zero rvalue
  // come last.

  i = 0;
  j = slaveCount;

  while ( i < j )
  {
    if ( rmask[i] )
    {
      i++;
    }
    else
    {
      j--;

      jem::swap ( rmask[i],      rmask[j]      );
      jem::swap ( slaveDofs_[i], slaveDofs_[j] );
    }
  }

  // Get the non-zero rvalues

  rvalues_.resize ( j );

  cons.getRvalues ( rvalues_, slaveDofs_[slice(BEGIN,j)] );
}


//-----------------------------------------------------------------------
//   adjustConstraints
//-----------------------------------------------------------------------


void ConHandler::adjustConstraints

  ( Constraints&   cons,
    const Vector&  state ) const

{
  const idx_t  conCount = rvalues_.size ();

  IdxVector    idofs    = slaveDofs_[slice(BEGIN,conCount)];

  Vector       rvals    ( conCount );


  if ( cons.masterDofCount() == 0 )
  {
    rvals = rvalues_ - state[idofs];
  }
  else
  {
    IdxVector  jdofs  ( 10 );
    Vector     coeffs ( 10 );

    double     rval;
    idx_t      icon;
    idx_t      idof;
    idx_t      j, n;

    for ( icon = 0; icon < conCount; icon++ )
    {
      idof = idofs[icon];
      n    = cons.masterDofCount ( idof );

      if ( n > jdofs.size() )
      {
        jdofs .resize ( n );
        coeffs.resize ( n );
      }

      cons.getConstraint ( rval, jdofs, coeffs, idof );

      rval -= state[idof];

      for ( j = 0; j < n; j++ )
      {
        rval += coeffs[j] * state[jdofs[j]];
      }

      rvals[icon] = rval;
    }
  }

  cons.setRvalues ( idofs, rvals );
}


//-----------------------------------------------------------------------
//   zeroConstraints
//-----------------------------------------------------------------------


void ConHandler::zeroConstraints ( Constraints& cons ) const
{
  const idx_t  n = rvalues_.size ();

  Vector       zeroes ( n );

  zeroes = 0.0;

  // Only touch the slave dofs with a non-zero rvalue

  cons.setRvalues ( slaveDofs_[slice(BEGIN,n)], zeroes );
}


//-----------------------------------------------------------------------
//   restoreConstraints
//-----------------------------------------------------------------------


void ConHandler::restoreConstraints ( Constraints& cons ) const
{
  const idx_t  n = rvalues_.size ();

  // Only touch the slave dofs with a non-zero rvalue

  cons.setRvalues ( slaveDofs_[slice(BEGIN,n)], rvalues_ );
}


JIVE_END_PACKAGE( implict )
