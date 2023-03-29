
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

#ifndef JEM_NUMERIC_FUNC_UFUNC_COMPILER_H
#define JEM_NUMERIC_FUNC_UFUNC_COMPILER_H

#include <jem/util/Flex.h>
#include "Operator.h"
#include "StringMap.h"
#include "Tokenizer.h"


JEM_BEGIN_UFUNC_IMP


using jem::util::Flex;

class CodeBlock;


//-----------------------------------------------------------------------
//   class Compiler
//-----------------------------------------------------------------------


class Compiler
{
 public:

  static const int        COMMA_OP      = -1;

  static const int        ABS_MODE      =  1 << 0;
  static const int        CALL_MODE     =  1 << 1;
  static const int        RVAL_MODE     =  1 << 2;
  static const int        IF_MODE       =  1 << 3;
  static const int        THEN_MODE     =  1 << 4;
  static const int        ELSE_MODE     =  1 << 5;


  enum                    Assoc
  {
                            LEFT_TO_RIGHT,
                            RIGHT_TO_LEFT
  };

  enum                    Rank
  {
                            NULL_RANK   = -1,
                            MIN_RANK,
                            COMMA_RANK  =  MIN_RANK,
                            OR_RANK,
                            AND_RANK,
                            LT_RANK,
                            LE_RANK     =  LT_RANK,
                            GT_RANK     =  LT_RANK,
                            GE_RANK     =  LT_RANK,
                            ADD_RANK,
                            SUB_RANK    =  ADD_RANK,
                            MUL_RANK,
                            DIV_RANK    =  MUL_RANK,
                            MOD_RANK    =  MUL_RANK,
                            POW_RANK,
                            NEG_RANK,
                            NOT_RANK    =  NEG_RANK,
                            CALL_RANK,
                            MAX_RANK    =  CALL_RANK
  };


                          Compiler        ();
                         ~Compiler        ();

  CodeBlock*              parse

    ( ArgList&              args,
      const String&         expr );


 private:

  void                    clear_          ();
  void                    parseSave_      ();

  void                    parseLet_

    ( ArgList&              args );

  void                    parseExpr_

    ( ArgList&              args );

  void                    parseSymbol_

    ( ArgList&              args );

  void                    parseDefined_

    ( ArgList&              args );

  void                    parseBuiltin_

    ( int                   token );

  void                    parsePowExpr_   ();

  void                    beginSubExpr_

    ( int                   token = '(' );

  idx_t                   endSubExpr_

    ( int                   token = '(' );

  void                    endAbsExpr_     ();

  void                    beginFuncCall_

    ( idx_t                 index );

  void                    endFuncCall_    ();
  void                    beginIfExpr_    ();
  void                    beginThenExpr_  ();
  void                    endThenExpr_    ();
  void                    beginElseExpr_  ();
  void                    endElseExpr_    ();

  void                    pushOperator_

    ( int                   rank,
      Opcode                opcode,
      Assoc                 assoc,
      idx_t                 argc );

  inline void             pushOperator_

    ( Operator*             op );

  void                    popOperator_    ();

  void                    pushLiteral_

    ( double                value );

  idx_t                   declareTmp_

    ( const String&         name );

  idx_t                   declareVar_

    ( const String&         name );

  idx_t                   declareFunc_

    ( const String&         name );

  String                  getFuncName_

    ( idx_t                 index,
      int                   token )          const;

  inline void             pushOperand_    ();
  inline void             popOperand_     ();

  inline void             enterMode_

    ( int                   mode );

  void                    parseError_     () const;

  void                    parseError_

    ( const String&         what )           const;

  void                    overflowError_

    ( const String&         what )           const;


 private:

  FastAllocator           allocator_;

  CodeBlock*              code_;
  Tokenizer               tokzer_;

  int                     mode_;
  idx_t                   depth_;
  idx_t                   stackBase_;
  idx_t                   stackSize_;

  bool                    isClean_;

  StringMap               tmpMap_;
  StringMap               varMap_;
  StringMap               funcMap_;

  Flex<Operator*>         opStack_;
  Flex<idx_t>             jmpStack_;

};


JEM_END_UFUNC_IMP

#endif
