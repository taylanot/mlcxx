
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

#ifndef JEM_BASE_TYPETRAITS_CPP
#define JEM_BASE_TYPETRAITS_CPP

#include <jem/base/TypeTraits.h>


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   TypeTraits specializations
//=======================================================================

//-----------------------------------------------------------------------
//   TypeTraits<void>
//-----------------------------------------------------------------------


String TypeTraits<void>::whatis ()
{
  return String ( "a void" );
}


String TypeTraits<void>::typeName ()
{
  return String ( "void" );
}


//-----------------------------------------------------------------------
//   TypeTraits<bool>
//-----------------------------------------------------------------------


String TypeTraits<bool>::whatis ()
{
  return String ( "a boolean" );
}


String TypeTraits<bool>::typeName ()
{
  return String ( "bool" );
}


//-----------------------------------------------------------------------
//   TypeTraits<byte>
//-----------------------------------------------------------------------


String TypeTraits<byte>::whatis ()
{
  return String ( "a byte" );
}


String TypeTraits<byte>::typeName ()
{
  return String ( "byte" );
}


//-----------------------------------------------------------------------
//   TypeTraits<char>
//-----------------------------------------------------------------------


String TypeTraits<char>::whatis ()
{
  return String ( "a character" );
}


String TypeTraits<char>::typeName ()
{
  return String ( "char" );
}


//-----------------------------------------------------------------------
//   TypeTraits<short>
//-----------------------------------------------------------------------


String TypeTraits<short>::typeName ()
{
  return String ( "short" );
}


//-----------------------------------------------------------------------
//   TypeTraits<int>
//-----------------------------------------------------------------------


String TypeTraits<int>::typeName ()
{
  return String ( "int" );
}


//-----------------------------------------------------------------------
//   TypeTraits<long>
//-----------------------------------------------------------------------


String TypeTraits<long>::typeName ()
{
  return String ( "long" );
}


//-----------------------------------------------------------------------
//   TypeTraits<lint>
//-----------------------------------------------------------------------


String TypeTraits<lint>::typeName ()
{
  return String ( "jem::lint" );
}


//-----------------------------------------------------------------------
//   TypeTraits<idx_t>
//-----------------------------------------------------------------------


String TypeTraits<idx_t>::whatis ()
{
  return String ( "an index" );
}


String TypeTraits<idx_t>::typeName ()
{
  return String ( "jem::idx_t" );
}


//-----------------------------------------------------------------------
//   TypeTraits<float>
//-----------------------------------------------------------------------


String TypeTraits<float>::typeName ()
{
  return String ( "float" );
}


//-----------------------------------------------------------------------
//   TypeTraits<double>
//-----------------------------------------------------------------------


String TypeTraits<double>::typeName ()
{
  return String ( "double" );
}


//-----------------------------------------------------------------------
//   TypeTraits<void*>
//-----------------------------------------------------------------------


String TypeTraits<void*>::whatis ()
{
  return String ( "a pointer" );
}


String TypeTraits<void*>::typeName ()
{
  return String ( "void*" );
}


//-----------------------------------------------------------------------
//   TypeTraits<String>
//-----------------------------------------------------------------------


String TypeTraits<String>::whatis ()
{
  return String ( "a string" );
}


String TypeTraits<String>::typeName ()
{
  return String ( "jem::String" );
}


JEM_END_PACKAGE_BASE

#endif
