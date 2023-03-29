
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
#include <jem/base/IllegalInputException.h>
#include <jive/util/utilities.h>
#include <jive/util/ItemSet.h>
#include <jive/util/StdDatabase.h>
#include <jive/util/Assignable.h>
#include <jive/mbody/Globdat.h>
#include <jive/mbody/BodyData.h>


JIVE_BEGIN_PACKAGE( mbody )


//=======================================================================
//   class BodyData
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BodyData::MASS_COL     = "mass";
const char*  BodyData::RMASS_COL    = "rmass";
const char*  BodyData::CENTER_COL   = "center";
const char*  BodyData::OFFSET_COL   = "offset";
const char*  BodyData::DENSITY_COL  = "density";
const char*  BodyData::ROTATION_COL = "rotation";


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


BodyData BodyData::get

  ( const BodySet&     bodies,
    const Properties&  globdat )

{
  using jive::util::Assignable;

  Assignable<BodyData>  bdata = find ( bodies, globdat );

  if ( bdata )
  {
    bdata = newBodyData ( bodies );

    bdata.store ( globdat );
  }

  return bdata;
}


//-----------------------------------------------------------------------
//   getFloatColumn_
//-----------------------------------------------------------------------


FloatDBColumn* BodyData::getFloatColumn_ ( const char* name ) const
{
  JEM_PRECHECK ( *this );

  FloatColumn*  fcol = data_->findFloatColumn ( name );

  if ( ! fcol )
  {
    fcol = data_->addFloatColumn ( name );
  }

  return fcol;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newBodyData
//-----------------------------------------------------------------------


BodyData newBodyData ( const BodySet&  bodies )
{
  using jive::util::StdDatabase;


  JEM_PRECHECK ( bodies );

  String name = util::joinNames ( bodies.getName(),
                                  Database::MAIN_DB );

  return BodyData (
    jem::newInstance<StdDatabase> ( name, bodies.getData() )
  );
}


JIVE_END_PACKAGE( mbody )
