
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

#ifndef JEM_UTIL_UTILITIES_H
#define JEM_UTIL_UTILITIES_H

#include <jem/defines.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   utility functions
//-----------------------------------------------------------------------


template <class T>

  inline void           clamp

  ( T&                    t,
    T                     hi );

template <class T>

  inline void           clamp

  ( T&                    t,
    T                     lo,
    T                     hi );

template <class T>

  inline T              clamped

  ( T                     t,
    T                     hi );

template <class T>

  inline T              clamped

  ( T                     t,
    T                     lo,
    T                     hi );





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   clamp
//-----------------------------------------------------------------------


template <class T>

  inline void  clamp ( T& t, T hi )

{
  clamp ( t, T(), hi );
}


template <class T>

  inline void  clamp ( T& t, T lo, T hi )

{
  if      ( t < lo )
  {
    t = lo;
  }
  else if ( hi < t )
  {
    t = hi;
  }
}


//-----------------------------------------------------------------------
//   clamped
//-----------------------------------------------------------------------


template <class T>

  inline T  clamped ( T t, T hi )

{
  return clamped ( t, T(), hi );
}


template <class T>

  inline T  clamped ( T t, T lo, T hi )

{
  if      ( t < lo )
  {
    return lo;
  }
  else if ( hi < t )
  {
    return hi;
  }
  else
  {
    return t;
  }
}


JEM_END_PACKAGE( util )

#endif
