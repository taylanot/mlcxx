
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

#ifndef JEM_NUMERIC_FUNC_UFUNC_OPCODES_H
#define JEM_NUMERIC_FUNC_UFUNC_OPCODES_H

#include "base.h"


JEM_BEGIN_UFUNC_IMP


//-----------------------------------------------------------------------
//   macro JEM_LIST_OPCODES
//-----------------------------------------------------------------------


#undef  JEM_NULL_MACRO
#undef  JEM_LIST_OPCODES

#define JEM_NULL_MACRO

#define JEM_LIST_OPCODES( prefix )               \
                                                 \
  prefix NOP,                                    \
  prefix RET,                                    \
                                                 \
  prefix LT,                                     \
  prefix LE,                                     \
  prefix GT,                                     \
  prefix GE,                                     \
  prefix OR,                                     \
  prefix AND,                                    \
  prefix ADD,                                    \
  prefix SUB,                                    \
  prefix MUL,                                    \
  prefix DIV,                                    \
  prefix MOD,                                    \
  prefix NEG,                                    \
  prefix NOT,                                    \
                                                 \
  prefix ABS,                                    \
  prefix SQRT,                                   \
  prefix CEIL,                                   \
  prefix FLOOR,                                  \
  prefix POW,                                    \
  prefix POW2,                                   \
  prefix POW3,                                   \
  prefix EXP,                                    \
  prefix LOG,                                    \
  prefix LOG10,                                  \
  prefix SIN,                                    \
  prefix COS,                                    \
  prefix TAN,                                    \
  prefix ASIN,                                   \
  prefix ACOS,                                   \
  prefix ATAN,                                   \
  prefix MAX2,                                   \
  prefix MIN2,                                   \
                                                 \
  prefix LOAD_ARG0,                              \
  prefix LOAD_ARG1,                              \
  prefix LOAD_ARG2,                              \
  prefix LOAD_TMP0,                              \
  prefix LOAD_TMP1,                              \
  prefix LOAD_TMP2,                              \
                                                 \
  prefix LOAD_ARG,                               \
  prefix LOAD_VAR,                               \
  prefix TRAP_VAR,                               \
  prefix LOAD_TMP,                               \
  prefix STORE_TMP,                              \
  prefix STORE_VAR,                              \
  prefix JUMP,                                   \
  prefix JUMP_IF_FALSE,                          \
  prefix MAX,                                    \
  prefix MIN,                                    \
  prefix DEFINED,                                \
                                                 \
  prefix CALL_FUNC,                              \
  prefix TRAP_FUNC


//-----------------------------------------------------------------------
//   class Opcodes
//-----------------------------------------------------------------------


class Opcodes
{
 public:

  enum
  {
    JEM_LIST_OPCODES( JEM_NULL_MACRO )
  };

  enum
  {
    OPCODE_COUNT = TRAP_FUNC + 1
  };


  static inline int   size ( Opcode code );

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline int Opcodes::size ( Opcode code )
{
  if      ( code < LOAD_ARG )
  {
    return 1;
  }
  else if ( code < CALL_FUNC )
  {
    return 2;
  }
  else
  {
    return 3;
  }
}


JEM_END_UFUNC_IMP

#endif
