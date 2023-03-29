
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

#ifndef JEM_UTIL_PROPS_UTILITIES_H
#define JEM_UTIL_PROPS_UTILITIES_H

#include <jem/base/array/Array.h>
#include <jem/io/PrintWriter.h>


JEM_BEGIN_PACKAGE( util )
JEM_BEGIN_NAMESPACE( props )


using jem::io::PrintWriter;


//-----------------------------------------------------------------------
//   public functions
//-----------------------------------------------------------------------


String                  joinNames

  ( const String&         first,
    const String&         second );

template <class T>

  inline void           printValue

  ( PrintWriter&          out,
    const T&              val );

template <class T>

  void                  printValue

  ( PrintWriter&          out,
    const Array<T>&       val );

void                    printValue

  ( PrintWriter&          out,
    char                  val );

void                    printValue

  ( PrintWriter&          out,
    const String&         val );



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   printValue (scalar)
//-----------------------------------------------------------------------


template <class T>

  inline void           printValue

  ( PrintWriter&          out,
    const T&              val )

{
  print ( out, val );
}


//-----------------------------------------------------------------------
//   printValue (Array)
//-----------------------------------------------------------------------


template <class T>

  void                  printValue

  ( PrintWriter&          out,
    const Array<T>&       val )

{
  const idx_t  n = val.size ();

  if ( n == 0 )
  {
    print ( out, "[]" );
  }
  else
  {
    print      ( out, "[ " );
    printValue ( out, val[0] );

    for ( idx_t i = 1; i < n; i++ )
    {
      print      ( out, ", " );
      printValue ( out, val[i] );
    }

    print ( out, " ]" );
  }
}


JEM_END_NAMESPACE( props )
JEM_END_PACKAGE( util )

#endif
