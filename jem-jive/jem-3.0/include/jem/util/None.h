
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

#ifndef JEM_UTIL_NONE_H
#define JEM_UTIL_NONE_H

#include <jem/base/Object.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class None
//-----------------------------------------------------------------------


class None
{
 public:

  static Class*           getType       ();
  static Ref<Object>      getInstance   ();

  static bool             isInstance

    ( Object&               obj );

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline bool               isNone

  ( const Ref<Object>&      obj );





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   isNone
//-----------------------------------------------------------------------


inline bool isNone ( const Ref<Object>& obj )
{
  if ( obj.getType() == Ref<Object>::OBJECT_TYPE )
  {
    return None::isInstance ( obj.getObject() );
  }
  else
  {
    return false;
  }
}


JEM_END_PACKAGE( util )

#endif
