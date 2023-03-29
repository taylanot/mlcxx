
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */


#include <typeinfo>
#include <jem/base/Class.h>
#include <jem/base/ClassCastException.h>
#include <jem/base/Object.h>


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class RawRef
//=======================================================================


const RawRef  RAW_REF = {};


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   castError
//-----------------------------------------------------------------------


void            castError

  ( const char*   where,
    Collectable*  rhs )

{
  String  whatis;

  if ( rhs )
  {
    Object*  obj = dynamic_cast<Object*> ( rhs );

    if ( obj )
    {
      whatis = obj->getClass()->getWhatis ();
    }
    else
    {
      whatis = String::format ( "an object of type `%s\'",
                                typeid(*rhs).name() );
    }
  }
  else
  {
    whatis = "a nullptr";
  }

  throw ClassCastException (
    where,
    String::format ( "dynamic cast from %s failed", whatis )
  );
}


JEM_END_PACKAGE_BASE
