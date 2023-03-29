
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


#include <jem/base/StringBuffer.h>
#include <jem/base/ClassCastException.h>
#include <jem/base/IllegalIndexException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jive/util/error.h>


using jem::Tuple;
using jem::ClassCastException;
using jem::IllegalIndexException;
using jem::IllegalArgumentException;


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   indexError
//-----------------------------------------------------------------------


void              indexError

  ( const String&   context,
    idx_t           index,
    idx_t           ubound )

{
  throw IllegalIndexException ( context, index, ubound );
}


void              indexError

  ( const String&   context,
    const String&   name,
    idx_t           index,
    idx_t           ubound )

{
  throw IllegalIndexException (
    context,
    String::format (
      "illegal %s index: %d; valid range is [0,%d)",
      name,
      index, ubound
    )
  );
}


//-----------------------------------------------------------------------
//   sizeError
//-----------------------------------------------------------------------


void              sizeError

  ( const String&   context,
    const String&   name,
    idx_t           actualSize,
    idx_t           formalSize )

{
  throw IllegalArgumentException (
    context,
    String::format (
      "%s has size %d but should have size %d",
      name,
      actualSize,
      formalSize
    )
  );
}


//-----------------------------------------------------------------------
//   shapeError
//-----------------------------------------------------------------------


void                      shapeError

  ( const String&           context,
    const String&           name,
    const Tuple<idx_t,2>&   actualShape,
    const Tuple<idx_t,2>&   formalShape )

{
  throw IllegalArgumentException (
    context,
    String::format (
      "%s has shape (%d,%d) but should have shape (%d,%d)",
      name,
      actualShape[0], actualShape[1],
      formalShape[0], formalShape[1]
    )
  );
}


void                      shapeError

  ( const String&           context,
    const String&           name,
    const Tuple<idx_t,3>&   actualShape,
    const Tuple<idx_t,3>&   formalShape )

{
  throw IllegalArgumentException (
    context,
    String::format (
      "%s has shape (%d,%d,%d) but should have shape (%d,%d,%d)",
      name,
      actualShape[0], actualShape[1],
      actualShape[2],
      formalShape[0], formalShape[1],
      formalShape[2]
    )
  );
}


void                      shapeError

  ( const String&           context,
    const String&           name,
    const Tuple<idx_t,4>&   actualShape,
    const Tuple<idx_t,4>&   formalShape )

{
  using jem::StringBuffer;

  StringBuffer  text;

  print ( text, name, " has shape (" );
  print ( text, actualShape[0], ',', actualShape[1] );
  print ( text, actualShape[2], ',', actualShape[3] );
  print ( text, " but should have shape (" );
  print ( text, actualShape[0], ',', actualShape[1] );
  print ( text, actualShape[2], ',', actualShape[3] );

  throw IllegalArgumentException ( context, text.toString() );
}


//-----------------------------------------------------------------------
//   nonSquareMatrixError
//-----------------------------------------------------------------------


void                      nonSquareMatrixError

  ( const String&           context,
    const Tuple<idx_t,2>&   shape )

{
  throw IllegalArgumentException (
    context,
    String::format (
      "non-square matrix; shape is (%d,%d)", shape[0], shape[1]
    )
  );
}


void                      nonSquareMatrixError

  ( const String&           context,
    const String&           name,
    const Tuple<idx_t,2>&   shape )

{
  throw IllegalArgumentException (
    context,
    String::format (
      "non-square %s matrix; shape is (%d,%d)",
      name,
      shape[0], shape[1]
    )
  );
}


//-----------------------------------------------------------------------
//   classCastError
//-----------------------------------------------------------------------


void              classCastError

  ( const String&   context,
    const String&   actualType,
    const String&   formalType )

{
  throw ClassCastException (
    context,
    String::format (
      "class `%s\' is not a base of class `%s\'",
      formalType,
      actualType
    )
  );
}


JIVE_END_PACKAGE( util )
