
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


#include <jem/base/IllegalInputException.h>
#include <jive/util/error.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/util/private/ConstraintsImp.h>


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   class ConstraintsImp
//=======================================================================

/*
  Implementation details.

  The protected member dofCount_ is always less than or equal to the
  actual number of DOFs as returned by dofs_->dofCount(). It is
  updated by the Constraints class and by the non-const methods
  checkDofIndex and checkDofIndices.
*/

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  ConstraintsImp::ERR_CONTEXT = "jive::util::Constraints";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ConstraintsImp::ConstraintsImp ( const Ref<DofSpace>& dofs ) :

  dofs_ ( dofs )

{
  JEM_ASSERT2 ( dofs, "NULL DofSpace" );

  dofCount_ = dofs_->dofCount ();
}


ConstraintsImp::ConstraintsImp ( const Self& rhs ) :

  dofs_ ( rhs.dofs_ )

{
  dofCount_ = rhs.dofCount_;
}


ConstraintsImp::~ConstraintsImp ()
{}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void ConstraintsImp::reserve ( idx_t slaveCount )
{}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void ConstraintsImp::trimToSize ()
{}


//-----------------------------------------------------------------------
//   checkDofIndices
//-----------------------------------------------------------------------


void ConstraintsImp::checkDofIndices ( const IdxVector& idofs )
{
  idx_t  n  = idofs.size ();
  idx_t  lo = 0;
  idx_t  hi = dofCount_ - 1;

  idx_t  idof;
  idx_t  i;


  for ( i = 0; i < n; i++ )
  {
    idof = idofs[i];

    if      ( idof < lo )
    {
      lo = idof;
    }
    else if ( idof > hi )
    {
      hi = idof;
    }
  }

  if ( lo < 0 )
  {
    dofIndexError ( lo );
  }

  if ( hi >= dofCount_ )
  {
    n = dofs_->dofCount ();

    if ( hi >= n )
    {
      dofIndexError ( hi );
    }

    if ( n != dofCount_ )
    {
      dofCount_ = n;

      newDofCount ();
    }
  }
}


//-----------------------------------------------------------------------
//   sizeError
//-----------------------------------------------------------------------


void ConstraintsImp::sizeError ( const Vector& vec ) const
{
  const idx_t  dofCount = dofs_->dofCount ();

  jive::util::sizeError ( ERR_CONTEXT, "vector",
                          vec.size(),  dofCount );
}


//-----------------------------------------------------------------------
//   resolveError
//-----------------------------------------------------------------------


void ConstraintsImp::resolveError ( idx_t idof ) const
{
  String  dofType;

  idx_t   itemID;
  idx_t   iitem;
  idx_t   itype;


  dofs_->decodeDofIndex ( iitem, itype, idof );

  itemID  = dofs_->getItems()->getItemID ( iitem );
  dofType = dofs_->getTypeName           ( itype );

  throw jem::IllegalInputException (
    ERR_CONTEXT,
    String::format (
      "unable to resolve constraint for dof `%s[%d]\' "
      "(circular dependency detected)",
      dofType,
      itemID
    )
  );
}


//-----------------------------------------------------------------------
//   dofIndexError
//-----------------------------------------------------------------------


void ConstraintsImp::dofIndexError ( idx_t idof ) const
{
  indexError ( ERR_CONTEXT, "dof", idof, dofs_->dofCount() );
}


//-----------------------------------------------------------------------
//   checkDofIndex_
//-----------------------------------------------------------------------


void ConstraintsImp::checkDofIndex_ ( idx_t idof )
{
  const idx_t  n = dofs_->dofCount ();

  if ( idof < 0 || idof >= n )
  {
    dofIndexError ( idof );
  }

  if ( n != dofCount_ )
  {
    dofCount_ = n;

    newDofCount ();
  }
}


JIVE_END_PACKAGE( util )
