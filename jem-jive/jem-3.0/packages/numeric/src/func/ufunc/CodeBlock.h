
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

#ifndef JEM_NUMERIC_FUNC_UFUNC_CODEBLOCK_H
#define JEM_NUMERIC_FUNC_UFUNC_CODEBLOCK_H

#include <jem/util/Flex.h>
#include "base.h"


JEM_BEGIN_UFUNC_IMP


using jem::util::Flex;


//-----------------------------------------------------------------------
//   class CodeBlock
//-----------------------------------------------------------------------

class CodeBlock
{
 public:

  enum                    Status
  {
                            UNRESOLVED,
                            RESOLVED,
                            RESOLVING
  };

  enum                    VarKind
  {
                            LITERAL,
                            SAVED,
                            EXTERNAL,
                            UNDEFINED
  };

  struct                  Var
  {
    String                  name;
    double                  value;
    VarKind                 kind;
  };

  struct                  Func
  {
    String                  name;
    Ref<Function>           addr;
  };


  explicit                CodeBlock

    ( const String&         expr = "" );

  void                    readFrom

    ( ObjectInput&          in );

  void                    writeTo

    ( ObjectOutput&         out )          const;

  void                    resolve

    ( const NameSpace&      symbols,
      int                   options );

  void                    unresolve     ();
  void                    trimToSize    ();

  double                  calcValue

    ( double*               vbuf,
      const double*         args );

  double                  calcDeriv

    ( double*               vbuf,
      idx_t                 iarg,
      const double*         args );

  bool                    isUsedArg

    ( idx_t                 iarg )         const;

  idx_t                   markUsedArgs

    ( bool*                 mask )         const;

  inline void             writeCode

    ( Opcode                code );

  inline idx_t            addVar

    ( const String&         name );

  inline idx_t            addLiteral

    ( double                value );

  inline idx_t            addSavedVar

    ( const String&         name,
      double                value );

  inline idx_t            addFunc

    ( const String&         name );


 public:

  String                  expr;

  Status                  status;
  idx_t                   argCount;
  idx_t                   tmpCount;
  idx_t                   extCount;
  idx_t                   maxSSize;
  Flex<Opcode>            opcodes;
  Flex<Var>               vars;
  Flex<Func>              funcs;


 private:

  void                    resolve_      ();

  inline double           evalPow_

    ( double                x,
      double                y )            const;

  void                    evalError_

    ( const String&         what,
      const double*         args = 0 )     const;

  void                    funcError_

    ( const char*           name,
      double                arg )          const;

  void                    funcError_

    ( const char*           name,
      double                arg0,
      double                arg1 )         const;

  void                    trapError_

    ( const Var&            var )          const;

  void                    trapError_

    ( const Func&           func )         const;

  void                    opcodeError_

    ( Opcode                code )         const;

  void                    resolveError_

    ( const String&         what )         const;

  void                    resolveError_

    ( const Var&            var )          const;

  void                    resolveError_

    ( const Func&           func )         const;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   writeCode
//-----------------------------------------------------------------------


inline void CodeBlock::writeCode ( Opcode code )
{
  opcodes.pushBack ( code );
}


//-----------------------------------------------------------------------
//   addVar
//-----------------------------------------------------------------------


idx_t CodeBlock::addVar ( const String&  name )
{
  Var  v = { name, 0.0, UNDEFINED };

  vars.pushBack ( std::move( v ) );

  extCount++;

  status = UNRESOLVED;

  return vars.size() - 1;
}


//-----------------------------------------------------------------------
//   addLiteral
//-----------------------------------------------------------------------


idx_t CodeBlock::addLiteral ( double value )
{
  Var  v = { String(), value, LITERAL };

  vars.pushBack ( std::move( v ) );

  return vars.size() - 1;
}


//-----------------------------------------------------------------------
//   addSavedVar
//-----------------------------------------------------------------------


idx_t CodeBlock::addSavedVar

  ( const String&  name,
    double         value )

{
  Var  v = { name, value, SAVED };

  vars.pushBack ( std::move( v ) );

  return vars.size() - 1;
}


//-----------------------------------------------------------------------
//   addFunc
//-----------------------------------------------------------------------


idx_t CodeBlock::addFunc ( const String& name )
{
  Func  f = { name, nullptr };

  funcs.pushBack ( std::move( f ) );

  extCount++;

  status = UNRESOLVED;

  return funcs.size() - 1;
}


JEM_END_UFUNC_IMP

#endif
