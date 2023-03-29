
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

#ifndef JIVE_MBODY_BODYDATA_H
#define JIVE_MBODY_BODYDATA_H

#include <jive/util/DatabaseSkin.h>
#include <jive/mbody/BodySet.h>


JIVE_BEGIN_PACKAGE( mbody )


//-----------------------------------------------------------------------
//   BodyData
//-----------------------------------------------------------------------


class BodyData : public DatabaseSkin
{
 public:

  typedef BodyData          Self;
  typedef DatabaseSkin      Super;

  static const char*        MASS_COL;
  static const char*        RMASS_COL;
  static const char*        CENTER_COL;
  static const char*        OFFSET_COL;
  static const char*        DENSITY_COL;
  static const char*        ROTATION_COL;

  struct                    StdColumns
  {
    FloatColumn*              mass;
    FloatColumn*              rmass;
    FloatColumn*              center;
    FloatColumn*              offset;
    FloatColumn*              density;
    FloatColumn*              rotation;
  };


  inline                    BodyData        ()       noexcept;

  explicit inline           BodyData

    ( const Ref<Database>&    bdata )                noexcept;

  explicit inline           BodyData

    ( Ref<Database>&&         bdata )                noexcept;

  inline BodySet            getBodies       () const;
  inline StdColumns         getStdColumns   () const;
  inline FloatColumn*       getMasses       () const;
  inline FloatColumn*       getRMasses      () const;
  inline FloatColumn*       getCenters      () const;
  inline FloatColumn*       getOffsets      () const;
  inline FloatColumn*       getDensities    () const;
  inline FloatColumn*       getRotations    () const;

  static inline BodyData    find

    ( const BodySet&          bodies,
      const Properties&       globdat );

  static BodyData           get

    ( const BodySet&          bodies,
      const Properties&       globdat );


 private:

  FloatColumn*              getFloatColumn_

    ( const char*             name )          const;


 private:

  Ref<Database>             data_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


BodyData                    newBodyData

  ( const BodySet&            bodies );





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline BodyData::BodyData () noexcept
{}


inline BodyData::BodyData

  ( const Ref<Database>&  bdata ) noexcept :

    Super ( bdata )

{}


inline BodyData::BodyData

  ( Ref<Database>&&  bdata ) noexcept :

    Super ( std::move( bdata ) )

{}


//-----------------------------------------------------------------------
//   getBodies
//-----------------------------------------------------------------------


inline BodySet BodyData::getBodies () const
{
  return BodySet

    ( static_cast<BodySet::Data*>( data_->getRowItems() ) );
}


//-----------------------------------------------------------------------
//   getStdColumns
//-----------------------------------------------------------------------


inline BodyData::StdColumns BodyData::getStdColumns () const
{
  StdColumns  cols;

  cols.mass     = getMasses    ();
  cols.rmass    = getRMasses   ();
  cols.center   = getCenters   ();
  cols.offset   = getOffsets   ();
  cols.density  = getDensities ();
  cols.rotation = getRotations ();

  return cols;
}


//-----------------------------------------------------------------------
//   getMasses etc.
//-----------------------------------------------------------------------


inline FloatDBColumn* BodyData::getMasses () const
{
  return getFloatColumn_ ( MASS_COL );
}


inline FloatDBColumn* BodyData::getRMasses () const
{
  return getFloatColumn_ ( RMASS_COL );
}


inline FloatDBColumn* BodyData::getCenters () const
{
  return getFloatColumn_ ( CENTER_COL );
}


inline FloatDBColumn* BodyData::getOffsets () const
{
  return getFloatColumn_ ( OFFSET_COL );
}


inline FloatDBColumn* BodyData::getDensities () const
{
  return getFloatColumn_ ( DENSITY_COL );
}


inline FloatDBColumn* BodyData::getRotations () const
{
  return getFloatColumn_ ( ROTATION_COL );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


inline BodyData BodyData::find

  ( const BodySet&     bodies,
    const Properties&  globdat )

{
  return BodyData ( Database::find( bodies.getData(), globdat ) );
}


JIVE_END_PACKAGE( mbody )

#endif
