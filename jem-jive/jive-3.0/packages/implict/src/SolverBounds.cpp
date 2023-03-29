
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


#include <utility>
#include <jem/base/rtti.h>
#include <jem/base/limits.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/select.h>
#include <jem/base/array/utilities.h>
#include <jem/util/None.h>
#include <jem/util/Properties.h>
#include <jive/util/DofSpace.h>
#include <jive/implict/Names.h>
#include <jive/implict/SolverBounds.h>


JIVE_BEGIN_PACKAGE( implict)


//=======================================================================
//   class SolverBounds
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  SolverBounds::LOWER_BOUND_ = 1 << 0;
const int  SolverBounds::UPPER_BOUND_ = 1 << 1;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


SolverBounds::SolverBounds ( const String& context ) :

  context_ ( context )

{
  status_ = 0;
}


SolverBounds::~SolverBounds ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void SolverBounds::configure ( const Properties& props )
{
  using jem::maxOf;
  using jem::util::isNone;
  using jem::util::Flex;

  StringVector  list;

  if ( props.find( list, PropNames::BOUNDS ) )
  {
    Flex<Bnd_>  bounds;

    for ( idx_t i = 0; i < list.size(); i++ )
    {
      Properties   data = props[list[i]];
      Ref<Object>  lb;
      Ref<Object>  ub;
      Bnd_         bnd;

      bnd.name       =  list[i];
      bnd.dofType    =  data[PropNames::DOF_TYPE];
      bnd.status     =  0;
      bnd.lowerBound = -maxOf<double> ();
      bnd.upperBound =  maxOf<double> ();

      if ( data.find( lb, PropNames::LOWER_BOUND ) )
      {
        if ( ! isNone( lb ) )
        {
          bnd.lowerBound = data[PropNames::LOWER_BOUND];
          bnd.status    |= LOWER_BOUND_;
        }
      }

      if ( data.find( ub, PropNames::UPPER_BOUND ) )
      {
        if ( ! isNone( ub ) )
        {
          bnd.upperBound = data[PropNames::UPPER_BOUND];
          bnd.status    |= UPPER_BOUND_;
        }
      }

      if ( bnd.upperBound < bnd.lowerBound )
      {
        props.propertyError ( list[i], "invalid bounds" );
      }

      bounds.pushBack ( std::move( bnd ) );
    }

    bounds_.swap ( bounds );
  }
  else
  {
    for ( idx_t i = 0; i < bounds_.size(); i++ )
    {
      if ( props.contains( bounds_[i].name ) )
      {
        Bnd_         bnd  = bounds_[i];
        Properties   data = props  [bnd.name];
        Ref<Object>  lb;
        Ref<Object>  ub;

        data.find ( bnd.dofType, PropNames::DOF_TYPE );

        if ( data.find( lb, PropNames::LOWER_BOUND ) )
        {
          if ( isNone( lb ) )
          {
            bnd.lowerBound = -maxOf<double> ();
            bnd.status    &= ~LOWER_BOUND_;
          }
          else
          {
            bnd.lowerBound =  data[PropNames::LOWER_BOUND];
            bnd.status    |=  LOWER_BOUND_;
          }
        }

        if ( data.find( ub, PropNames::UPPER_BOUND ) )
        {
          if ( isNone( ub ) )
          {
            bnd.upperBound =  maxOf<double> ();
            bnd.status    &= ~UPPER_BOUND_;
          }
          else
          {
            bnd.upperBound =  data[PropNames::UPPER_BOUND];
            bnd.status    |=  UPPER_BOUND_;
          }
        }

        if ( bnd.upperBound < bnd.lowerBound )
        {
          props.propertyError ( list[i], "invalid bounds" );
        }

        bounds_[i] = std::move ( bnd );
      }
    }
  }

  status_ = 0;

  for ( idx_t i = 0; i < bounds_.size(); i++ )
  {
    status_ |= bounds_[i].status;
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void SolverBounds::getConfig ( const Properties& conf ) const
{
  using jem::util::None;

  const idx_t   bndCount = bounds_.size ();

  StringVector  list     ( bndCount );


  for ( idx_t i = 0; i < bndCount; i++ )
  {
    const Bnd_&  bnd  = bounds_[i];
    Properties   data = conf.makeProps ( bnd.name );

    list[i] = bnd.name;

    data.set ( PropNames::DOF_TYPE, bnd.dofType );

    if ( bnd.status & LOWER_BOUND_ )
    {
      data.set ( PropNames::LOWER_BOUND, bnd.lowerBound );
    }
    else
    {
      data.set ( PropNames::LOWER_BOUND, None::getInstance() );
    }

    if ( bnd.status & UPPER_BOUND_ )
    {
      data.set ( PropNames::UPPER_BOUND, bnd.upperBound );
    }
    else
    {
      data.set ( PropNames::UPPER_BOUND, None::getInstance() );
    }
  }

  conf.set ( PropNames::BOUNDS, list );
}


//-----------------------------------------------------------------------
//   hasBounds
//-----------------------------------------------------------------------


bool SolverBounds::hasBounds () const noexcept
{
  return (status_ != 0);
}


//-----------------------------------------------------------------------
//   getBounds
//-----------------------------------------------------------------------


bool SolverBounds::getBounds

  ( const Vector&    lb,
    const Vector&    ub,
    const DofSpace&  dofs ) const

{
  JEM_PRECHECK ( lb.size() == dofs.dofCount() );
  JEM_PRECHECK ( ub.size() == dofs.dofCount() );

  using jem::maxOf;

  lb = -maxOf<double> ();
  ub =  maxOf<double> ();

  if ( status_ )
  {
    const idx_t  itemCount = dofs.itemCount ();
    const idx_t  bndCount  = bounds_  .size ();

    IdxMatrix    ibuffer   ( itemCount, 2 );

    IdxVector    iitems    = ibuffer[0];
    IdxVector    idofs     = ibuffer[1];

    iitems = jem::iarray ( itemCount );

    for ( idx_t i = 0; i < bndCount; i++ )
    {
      const Bnd_&  bnd = bounds_[i];

      if ( bnd.status )
      {
        idx_t  jtype = dofs.findType ( bnd.dofType );
        idx_t  count = 0;

        if ( jtype < 0 )
        {
          throw jem::IllegalInputException (
            context_,
            "undefined DOF type: " + bnd.dofType
          );
        }

        count = dofs.getDofsForType ( idofs, iitems, jtype );

        if ( bnd.status & LOWER_BOUND_ )
        {
          lb[idofs[slice(BEGIN,count)]] = bnd.lowerBound;
        }

        if ( bnd.status & UPPER_BOUND_ )
        {
          ub[idofs[slice(BEGIN,count)]] = bnd.upperBound;
        }
      }
    }
  }

  return (status_ != 0);
}


JIVE_END_PACKAGE( implict )

