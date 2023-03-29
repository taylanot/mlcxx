
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


#include <cmath>
#include <jem/base/assert.h>
#include <jem/base/limits.h>
#include <jem/base/System.h>
#include <jem/base/IllegalInputException.h>
#include <jive/mbody/utilities.h>
#include <jive/mbody/BodyData.h>


JIVE_BEGIN_PACKAGE( mbody )


//=======================================================================
//   class BodyDataChecker
//=======================================================================


class BodyDataChecker
{
 public:

  static const int        MAX_WARNINGS = 100;


  explicit                BodyDataChecker

    ( const BodyData&       bdata );

  void                    checkData

    ( const String&         context );


 private:

  void                    checkMass_

    ( idx_t                 ibody );

  void                    checkCenter_

    ( idx_t                 ibody );

  void                    checkOffset_

    ( idx_t                 ibody );

  void                    checkInertia2D_

    ( idx_t                 ibody );

  void                    checkInertia3D_

    ( idx_t                 ibody );

  void                    checkRotation2D_

    ( idx_t                 ibody );

  void                    checkRotation3D_

    ( idx_t                 ibody );

  void                    dataError_

    ( idx_t                 ibody,
      const String&         what  )   const;

  void                    dataWarning_

    ( idx_t                 ibody,
      const String&         what  );


 private:

  BodyData                bdata_;
  BodyData::StdColumns    cols_;

  BodySet                 bodies_;
  idx_t                   rank_;

  String                  context_;
  Vector                  buffer_;

  int                     warnCount_;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


BodyDataChecker::BodyDataChecker ( const BodyData& bdata ) :

  bdata_  ( bdata ),
  bodies_ ( bdata.getBodies() )

{
  cols_      = bdata  .getStdColumns ();
  rank_      = bodies_.rank          ();
  warnCount_ = 0;

  buffer_.resize ( 6 );
}


//-----------------------------------------------------------------------
//   checkData
//-----------------------------------------------------------------------


void BodyDataChecker::checkData ( const String& context )
{
  const idx_t  bodyCount = bodies_.size ();

  idx_t        ibody;


  if ( rank_ < 2 || rank_ > 3 )
  {
    throw jem::IllegalInputException (
      context_,
      "only 2-D and 3-D bodies are supported"
    );
  }

  context_ = context;

  for ( ibody = 0; ibody < bodyCount; ibody++ )
  {
    checkMass_   ( ibody );
    checkCenter_ ( ibody );
    checkOffset_ ( ibody );

    if ( rank_ == 2 )
    {
      checkInertia2D_  ( ibody );
      checkRotation2D_ ( ibody );
    }
    else
    {
      checkInertia3D_  ( ibody );
      checkRotation3D_ ( ibody );
    }
  }
}


//-----------------------------------------------------------------------
//   checkMass_
//-----------------------------------------------------------------------


void BodyDataChecker::checkMass_ ( idx_t ibody )
{
  bool   hasMass;
  idx_t  n;


  n       = cols_.mass->findValues ( buffer_, ibody );
  hasMass = (n == 1);

  if ( hasMass )
  {
    if ( buffer_[0] <= 0.0 )
    {
      dataError_ ( ibody, "non-positive mass" );
    }
  }
  else if ( n > 1 )
  {
    dataError_ ( ibody, "invalid mass" );
  }

  if ( ! hasMass )
  {
    n = cols_.density->findValues ( buffer_, ibody );

    if ( n == 0 )
    {
      dataError_ ( ibody, "no mass and no density specified" );
    }
    else if ( n == 1 )
    {
      if ( buffer_[0] <= 0.0 )
      {
        dataError_ ( ibody, "non-positive density" );
      }
    }
    else
    {
      dataError_ ( ibody, "invalid density" );
    }
  }
}


//-----------------------------------------------------------------------
//   checkCenter_
//-----------------------------------------------------------------------


void BodyDataChecker::checkCenter_ ( idx_t ibody )
{
  idx_t  n = cols_.center->findValues ( buffer_, ibody );

  if ( n == rank_ )
  {
    if ( cols_.rmass->rowSize( ibody ) == 0 )
    {
      dataWarning_ ( ibody, "no inertia specified" );
    }
  }
  else if ( n != rank_ )
  {
    dataError_ ( ibody, "invalid center" );
  }
}


//-----------------------------------------------------------------------
//   checkOffset_
//-----------------------------------------------------------------------


void BodyDataChecker::checkOffset_ ( idx_t ibody )
{
  idx_t  n = cols_.offset->findValues ( buffer_, ibody );

  if ( n > 0 && n != rank_ )
  {
    dataError_ ( ibody, "invalid offset" );
  }
}


//-----------------------------------------------------------------------
//   checkInertia2D_
//-----------------------------------------------------------------------


void BodyDataChecker::checkInertia2D_ ( idx_t ibody )
{
  idx_t  n = cols_.rmass->findValues ( buffer_, ibody );

  if ( n == 1 )
  {
    if ( buffer_[0] <= 0.0 )
    {
      dataError_ ( ibody, "non-positive inertia" );
    }
  }
  else if ( n > 1 )
  {
    dataError_ ( ibody, "invalid inertia" );
  }
}


//-----------------------------------------------------------------------
//   checkInertia3D_
//-----------------------------------------------------------------------


void BodyDataChecker::checkInertia3D_ ( idx_t ibody )
{
  idx_t  n = cols_.rmass->findValues ( buffer_, ibody );

  if ( n == 3 || n == 6 )
  {
    for ( int i = 0; i < 3; i++ )
    {
      if ( buffer_[i] <= 0.0 )
      {
        dataError_ ( ibody, "non-positive inertia" );
      }
    }
  }
  else if ( n > 0 )
  {
    dataError_ ( ibody, "invalid inertia" );
  }
}


//-----------------------------------------------------------------------
//   checkRotation2D_
//-----------------------------------------------------------------------


void BodyDataChecker::checkRotation2D_ ( idx_t ibody )
{
  idx_t  n = cols_.rotation->findValues ( buffer_, ibody );

  if ( n > 1 )
  {
    dataError_ ( ibody, "invalid rotation" );
  }
}


//-----------------------------------------------------------------------
//   checkRotation3D_
//-----------------------------------------------------------------------


void BodyDataChecker::checkRotation3D_ ( idx_t ibody )
{
  idx_t  n = cols_.rotation->findValues ( buffer_, ibody );

  if ( n == 4 )
  {
    double  s = 0.0;

    for ( int i = 0; i < 3; i++ )
    {
      s += std::fabs ( buffer_[i] );
    }

    if ( jem::isTiny( s ) )
    {
      dataError_ ( ibody, "zero rotation axis" );
    }
  }
  else if ( n > 0 )
  {
    dataError_ ( ibody, "invalid rotation" );
  }
}


//-----------------------------------------------------------------------
//   dataError_
//-----------------------------------------------------------------------


void BodyDataChecker::dataError_

  ( idx_t          ibody,
    const String&  what ) const

{
  const idx_t  bodyID = bodies_.getBodyID ( ibody );

  throw jem::IllegalInputException (
    context_,
    String::format (
      "invalid data for body %d: %s", bodyID, what
    )
  );
}


//-----------------------------------------------------------------------
//   dataWarning_
//-----------------------------------------------------------------------


void BodyDataChecker::dataWarning_

  ( idx_t          ibody,
    const String&  what )

{
  using jem::System;

  const idx_t  bodyID = bodies_.getBodyID ( ibody );

  if ( warnCount_ < MAX_WARNINGS )
  {
    print ( System::warn(), context_,
            " : suspicious data for body ", bodyID,
            " (", what, ")\n" );
  }
  else if ( warnCount_ == MAX_WARNINGS )
  {
    print ( System::warn(), context_,
            " : more warnings not printed\n" );
  }

  if ( warnCount_ <= MAX_WARNINGS )
  {
    warnCount_++;
  }
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   checkBodyData
//-----------------------------------------------------------------------


void               checkBodyData

  ( const BodyData&  bdata,
    const String&    context )

{
  JEM_PRECHECK ( bdata );

  BodyDataChecker  chk ( bdata );

  chk.checkData ( context );
}


JIVE_END_PACKAGE( mbody )
