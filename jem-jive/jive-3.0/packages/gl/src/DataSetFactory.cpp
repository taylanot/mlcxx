
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
#include <jem/util/Properties.h>
#include <jive/util/CtorMap.h>
#include <jive/gl/DataSetFactory.h>


JIVE_BEGIN_PACKAGE( gl )


using jive::util::CtorMap;


//=======================================================================
//   class DataSetFactory::CtorMap_
//=======================================================================


class DataSetFactory::CtorMap_ : public CtorMap<Constructor>
{};


//=======================================================================
//   class DataSetFactory
//=======================================================================

//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void DataSetFactory::declare

  ( const String&  type,
    Constructor    ctor )

{
  JEM_PRECHECK ( ctor );

  CtorMap_::insert ( type, ctor );
}


//-----------------------------------------------------------------------
//   exists
//-----------------------------------------------------------------------


bool DataSetFactory::exists ( const String& type )
{
  return CtorMap_::contains ( type );
}


//-----------------------------------------------------------------------
//   listKnownTypes
//-----------------------------------------------------------------------


StringVector DataSetFactory::listKnownTypes ()
{
  return CtorMap_::listKnownTypes ();
}


//-----------------------------------------------------------------------
//   newInstance
//-----------------------------------------------------------------------


Ref<DataSet> DataSetFactory::newInstance

  ( const String&      type,
    const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat,
    const AppContext&  apx )

{
  Constructor ctor = CtorMap_::find ( type );

  if ( ctor )
  {
    return ctor ( name, conf, props, globdat, apx );
  }
  else
  {
    return nullptr;
  }
}


Ref<DataSet> DataSetFactory::newInstance

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat,
    const AppContext&  apx )

{
  Properties    subConf  = conf.makeProps ( name );
  Properties    subProps = props.getProps ( name );

  Ref<DataSet>  dset;
  String        type;


  subProps.get ( type, TYPE_PROP );
  subConf .set ( TYPE_PROP, type );

  dset = newInstance ( type, name, conf, props, globdat, apx );

  if ( ! dset )
  {
    noSuchTypeError ( subProps.getContext ( TYPE_PROP ),
                      type,
                      "data set",
                      listKnownTypes() );
  }

  return dset;
}


JIVE_END_PACKAGE( gl )
