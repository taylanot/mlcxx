
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

#ifndef JEM_MP_TYPE_H
#define JEM_MP_TYPE_H

#include <jem/defines.h>


namespace jem
{
  namespace util
  {
    template <class U, class V> class Pair;
  }
}


JEM_BEGIN_PACKAGE( mp )


//-----------------------------------------------------------------------
//   enum Type & related stuff
//-----------------------------------------------------------------------


enum Type
{
  NO_TYPE,
  BYTE,
  CHAR,
  SHORT,
  INT,
  LONG,
  LINT,
  IDX,
  FLOAT,
  DOUBLE,
  INT_PAIR,
  DOUBLE_INT
};


enum
{
  TYPE_COUNT = DOUBLE_INT + 1
};


extern const int TYPE_SIZES[TYPE_COUNT];


typedef util::Pair<int,int>     IntPair;
typedef util::Pair<double,int>  DoubleInt;


inline int    sizeOf   ( Type  t )    noexcept;

const char*   typeName ( int   t )    noexcept;

inline Type   getType  ( byte*      ) noexcept;
inline Type   getType  ( char*      ) noexcept;
inline Type   getType  ( short*     ) noexcept;
inline Type   getType  ( int*       ) noexcept;
inline Type   getType  ( long*      ) noexcept;
inline Type   getType  ( lint*      ) noexcept;
inline Type   getType  ( idx_t*     ) noexcept;
inline Type   getType  ( float*     ) noexcept;
inline Type   getType  ( double*    ) noexcept;
inline Type   getType  ( IntPair*   ) noexcept;
inline Type   getType  ( DoubleInt* ) noexcept;





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   sizeOf
//-----------------------------------------------------------------------


inline int sizeOf ( Type t ) noexcept
{
  return TYPE_SIZES[t];
}


//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


inline Type getType ( byte* ) noexcept
{
  return BYTE;
}


inline Type getType ( char* ) noexcept
{
  return CHAR;
}


inline Type getType ( short* ) noexcept
{
  return SHORT;
}


inline Type getType ( int* ) noexcept
{
  return INT;
}


inline Type getType ( long* ) noexcept
{
  return LONG;
}


inline Type getType ( lint* ) noexcept
{
  return LINT;
}


inline Type getType ( idx_t* ) noexcept
{
  return IDX;
}


inline Type getType ( float* ) noexcept
{
  return FLOAT;
}


inline Type getType ( double* ) noexcept
{
  return DOUBLE;
}


inline Type getType ( IntPair* ) noexcept
{
  return INT_PAIR;
}


inline Type getType ( DoubleInt* ) noexcept
{
  return DOUBLE_INT;
}


JEM_END_PACKAGE( mp )

#endif
