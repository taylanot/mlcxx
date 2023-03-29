
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


#include <jem/base/Class.h>
#include <jem/base/Integer.h>
#include <jem/base/ParseException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/PropertyException.h>
#include "Errors.h"


JEM_BEGIN_PACKAGE( util )
JEM_BEGIN_NAMESPACE( props )


//=======================================================================
//   class Errors
//=======================================================================

//-----------------------------------------------------------------------
//   typeMismatch
//-----------------------------------------------------------------------


void Errors::typeMismatch

  ( const String&       where,
    const String&       name,
    const Ref<Object>&  obj,
    Class*              target )

{
  String  targetKind = target->getWhatis ();
  String  sourceKind;

  if ( ! obj )
  {
    sourceKind = "NIL";
  }
  else
  {
    sourceKind = obj.getClass()->getWhatis ();
  }

  throw PropertyException (
    where,
    String::format (
      "property `%s\' has an invalid type; "
      "expected %s but got %s",
      name,
      targetKind,
      sourceKind
    )
  );
}


//-----------------------------------------------------------------------
//   valueOutOfRange
//-----------------------------------------------------------------------


void Errors::valueOutOfRange

  ( const String&  where,
    const String&  name,
    lint           lbound,
    lint           ubound )

{
  throw PropertyException (
    where,
    String::format (
      "property `%s\' out of range; valid range is [%d,%d]",
      name,
      lbound,
      ubound
    )
  );
}


void Errors::valueOutOfRange

  ( const String&  where,
    const String&  name,
    double         lbound,
    double         ubound )

{
  throw PropertyException (
    where,
    String::format (
      "property `%s\' out of range; valid range is [%e,%e]",
      name,
      lbound,
      ubound
    )
  );
}


//-----------------------------------------------------------------------
//   noSuchProperty
//-----------------------------------------------------------------------


void Errors::noSuchProperty

  ( const String&  where,
    const String&  name )

{
  throw PropertyException (
    where,
    String::format (
      "property `%s\' has not been defined", name
    )
  );
}


//-----------------------------------------------------------------------
//   nonAggregateType
//-----------------------------------------------------------------------


void Errors::nonAggregateType

  ( const String&  where,
    const String&  name )

{
  throw PropertyException (
    where,
    String::format (
      "property `%s\' is not an aggregate type", name
    )
  );
}


//-----------------------------------------------------------------------
//   arraySizeOverflow
//-----------------------------------------------------------------------


void Errors::arraySizeOverflow ( lint lineno )
{
  throw PropertyException (
    String ( lineno ),
    "array size overflow"
  );
}


//-----------------------------------------------------------------------
//   invalidOperands
//-----------------------------------------------------------------------


void Errors::invalidOperands ( lint lineno )
{
  throw PropertyException (
    String ( lineno ),
    "invalid operands"
  );
}


//-----------------------------------------------------------------------
//   illegalInput
//-----------------------------------------------------------------------


void Errors::illegalInput

  ( lint           lineno,
    const String&  source,
    const String&  what )

{
  throw IllegalInputException ( getContext( lineno, source ), what );
}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String Errors::getContext

  ( lint           lineno,
    const String&  source )

{
  return String::format ( "%s, line %d", source, lineno );
}


//-----------------------------------------------------------------------
//   setContext
//-----------------------------------------------------------------------


void Errors::setContext

  ( Exception&     ex,
    lint           lineno,
    const String&  source )

{
  lint  n;

  if ( Integer::parse( n, ex.where() ) )
  {
    lineno = n;
  }

  ex.setContext ( getContext( lineno, source ) );
}


JEM_END_NAMESPACE( props )
JEM_END_PACKAGE( util )
