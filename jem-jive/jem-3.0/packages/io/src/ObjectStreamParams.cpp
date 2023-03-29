
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


#include <jem/io/ObjectStreamParams.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class ObjectStreamParams
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  ObjectStreamParams::MAGIC          = 1234;
const int  ObjectStreamParams::VERSION        = 1;
const int  ObjectStreamParams::MAX_CHUNK_SIZE = 16 * 1024;


//-----------------------------------------------------------------------
//   getChunkName
//-----------------------------------------------------------------------


const char* ObjectStreamParams::getChunkName ( byte tag ) noexcept
{
  switch ( (int) tag )
  {
  case HEADER_TAG: return "header";
  case DATA_TAG:   return "data";
  case NIL_TAG:    return "null";
  case BOOL_TAG:   return "bool";
  case CHAR_TAG:   return "char";
  case STRING_TAG: return "string";
  case INT_TAG:    return "int";
  case FLOAT_TAG:  return "float";
  case REF_TAG:    return "ref";
  case XREF_TAG:   return "xref";
  case CLASS_TAG:  return "class";
  case OBJECT_TAG: return "object";
  case RESET_TAG:  return "reset";
  default:         return "unknown";
  }
}


JEM_END_PACKAGE( io )
