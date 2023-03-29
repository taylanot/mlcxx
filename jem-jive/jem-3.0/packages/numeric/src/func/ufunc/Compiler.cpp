
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


#include <cmath>
#include <jem/base/limits.h>
#include <jem/base/Error.h>
#include <jem/base/ParseException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/StringUtils.h>
#include "Opcodes.h"
#include "CodeBlock.h"
#include "Constants.h"
#include "Compiler.h"


JEM_BEGIN_UFUNC_IMP


using jem::util::StringUtils;


//=======================================================================
//   class Compiler
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


Compiler::Compiler () : allocator_ ( sizeof(Operator) )
{
  code_      = 0;
  mode_      = 0;
  depth_     = 0;
  stackBase_ = 0;
  stackSize_ = 0;
  isClean_   = true;
}


Compiler::~Compiler ()
{
  if ( ! isClean_ )
  {
    clear_ ();
  }
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


CodeBlock* Compiler::parse

  ( ArgList&       args,
    const String&  expr )

{
  CodeBlock*  ret;
  int         token;
  idx_t       n;


  if ( ! isClean_ )
  {
    clear_ ();
  }

  tokzer_.setInput ( expr );

  code_ = new CodeBlock ( expr );
  token = tokzer_.nextToken ();

  if ( token == Tokens::SAVE )
  {
    parseSave_ ();

    token = tokzer_.nextToken ();
  }

  if ( token == Tokens::LET )
  {
    parseLet_ ( args );

    token = tokzer_.nextToken ();
  }

  if ( token != Tokens::RETURN )
  {
    tokzer_.pushBackToken ();
  }

  parseExpr_ ( args );

  if ( tokzer_.nextToken() != Tokens::END )
  {
    parseError_ ();
  }

  popOperand_ ();

  n = args.argCount ();

  if ( n > code_->argCount )
  {
    code_->argCount = n;
  }

  code_->tmpCount = tmpMap_.size ();

  code_->writeCode  ( Opcodes::RET );
  code_->trimToSize ();

  ret   = code_;
  code_ = 0;

  return ret;
}


//-----------------------------------------------------------------------
//   clear_
//-----------------------------------------------------------------------


void Compiler::clear_ ()
{
  if ( code_ )
  {
    delete code_;
    code_ = 0;
  }

  for ( idx_t i = 0; i < opStack_.size(); i++ )
  {
    if ( opStack_[i] )
    {
      allocator_.dealloc ( opStack_[i] );
    }
  }

  tmpMap_  .clear ();
  varMap_  .clear ();
  funcMap_ .clear ();
  opStack_ .clear ();
  jmpStack_.clear ();

  mode_      = 0;
  depth_     = 0;
  stackBase_ = 0;
  stackSize_ = 0;
  isClean_   = true;
}


//-----------------------------------------------------------------------
//   parseSave_
//-----------------------------------------------------------------------


void Compiler::parseSave_ ()
{
  String  name;
  double  value;
  double  sign;
  int     token;
  idx_t   index;


  do
  {
    token = tokzer_.nextToken ();

    if ( token != Tokens::SYMBOL )
    {
      break;
    }

    name = tokzer_.getSymbol ();

    if ( tokzer_.nextToken() != '=' )
    {
      parseError_ ( "expected an `=\'" );
    }

    if ( varMap_.contains( name ) )
    {
      parseError_ (
        String::format (
          "saved variable `%s\' declared twice", name
        )
      );
    }

    token = tokzer_.nextToken ();
    sign  = 1.0;
    value = 0.0;

    if      ( token == '-' )
    {
      sign  = -sign;
      token = tokzer_.nextToken ();
    }
    else if ( token == '+' )
    {
      token = tokzer_.nextToken ();
    }

    switch ( token )
    {
    case Tokens::NUMBER:

      value = tokzer_.getNumber ();

      break;

    case Tokens::PI:

      value = Constants::PI;

      break;

    case Tokens::EPS:

      value = Constants::EPS;

      break;

    case Tokens::BIGNUM:

      value = Constants::BIGNUM;

      break;

    case Tokens::TINY:

      value = Constants::TINY;

      break;

    case Tokens::TRUE:

      value = Constants::TRUE;

      break;

    case Tokens::FALSE:

      value = Constants::FALSE;

      break;

    case Tokens::SYMBOL:
    {
      String  symbol = tokzer_.getSymbol ();

      index = findSymbol ( symbol, varMap_ );

      if ( index < 0 ||
           code_->vars[index].kind == CodeBlock::UNDEFINED )
      {
        parseError_ ( "undefined variable: " + symbol );
      }

      value = code_->vars[index].value;

      break;
    }

    default:

      parseError_ ( "expected a number, constant or "
                    "variable name" );
    }

    index = code_->addSavedVar ( name, sign * value );

    if ( index > (idx_t) maxOf<Opcode>() )
    {
      overflowError_ ( "too many saved variables" );
    }

    varMap_.insert ( makePair( name, index ) );

    token = tokzer_.nextToken ();
  }
  while ( token == ',' || token == ';' );

  tokzer_.pushBackToken ();
}


//-----------------------------------------------------------------------
//   parseLet_
//-----------------------------------------------------------------------


void Compiler::parseLet_ ( ArgList& args )
{
  String  name;
  Opcode  opcode;
  int     token;
  idx_t   index;


  do
  {
    token = tokzer_.nextToken ();

    if ( token != Tokens::SYMBOL )
    {
      break;
    }

    name = tokzer_.getSymbol ();

    if ( tokzer_.nextToken() != '=' )
    {
      parseError_ ( "expected an `=\'" );
    }

    parseExpr_ ( args );

    index = findSymbol ( name, varMap_ );

    if ( index >= 0 &&
         code_->vars[index].kind == CodeBlock::SAVED )
    {
      opcode = Opcodes::STORE_VAR;
    }
    else
    {
      index  = declareTmp_ ( name );
      opcode = Opcodes::STORE_TMP;
    }

    code_->writeCode ( opcode );
    code_->writeCode ( (Opcode) index  );

    popOperand_ ();

    token = tokzer_.nextToken ();
  }
  while ( token == ',' || token == ';' );

  tokzer_.pushBackToken ();
}


//-----------------------------------------------------------------------
//   parseExpr_
//-----------------------------------------------------------------------


void Compiler::parseExpr_ ( ArgList& args )
{
  const idx_t  depth0 = depth_;

  bool         done;
  int          token;
  idx_t        xopc;


  depth_ = 0;

  beginSubExpr_ ( Tokens::END );

  mode_  = 0;
  done   = false;

  while ( ! done )
  {
    token = tokzer_.nextToken ();

    switch ( token )
    {
    case Tokens::END:

      done = true;

      break;

    case Tokens::NUMBER:

      pushLiteral_ ( tokzer_.getNumber() );

      break;

    case Tokens::SYMBOL:

      parseSymbol_ ( args );

      break;

    case Tokens::IF:

      beginIfExpr_ ();

      break;

    case Tokens::ELSE:

      beginElseExpr_ ();

      break;

    case Tokens::ELIF:

      endThenExpr_ ();

      if ( tokzer_.nextToken() != '(' )
      {
        parseError_ ( "expected a `(\'" );
      }

      beginSubExpr_ ();
      enterMode_    ( IF_MODE );

      break;

    case Tokens::ENDIF:

      endElseExpr_  ();

      break;

    case '(':

      if ( mode_ & RVAL_MODE )
      {
        parseError_ ();
      }

      beginSubExpr_ ();

      break;

    case ')':

      if ( depth_ == 0 )
      {
        done = true;
        break;
      }

      if ( mode_ & CALL_MODE )
      {
        endFuncCall_ ();
        break;
      }

      if ( mode_ & IF_MODE )
      {
        beginThenExpr_ ();
        break;
      }

      if ( mode_ & ELSE_MODE )
      {
        endElseExpr_ ();
        break;
      }

      xopc = endSubExpr_ ();

      if ( xopc )
      {
        enterMode_( RVAL_MODE );
      }

      break;

    case ',':

      if ( mode_ & CALL_MODE )
      {
        pushOperator_ ( COMMA_RANK, COMMA_OP, LEFT_TO_RIGHT, 2 );
        break;
      }

      if ( mode_ & IF_MODE )
      {
        beginThenExpr_ ();
        break;
      }

      if ( mode_ & THEN_MODE )
      {
        beginElseExpr_ ();
        break;
      }

      done = true;

      break;

    case '|':

      if ( mode_ & ABS_MODE )
      {
        endAbsExpr_ ();
      }
      else
      {
        enterMode_    ( RVAL_MODE );
        beginSubExpr_ ( token );
        enterMode_    ( ABS_MODE );
      }

      break;

    case Tokens::OR:

      pushOperator_ ( OR_RANK,  Opcodes::OR,  LEFT_TO_RIGHT, 2 );

      break;

    case Tokens::AND:

      pushOperator_ ( AND_RANK, Opcodes::AND, LEFT_TO_RIGHT, 2 );

      break;

    case '<':

      pushOperator_ ( LT_RANK,  Opcodes::LT,  LEFT_TO_RIGHT, 2 );

      break;

    case Tokens::LE:

      pushOperator_ ( LE_RANK,  Opcodes::LE,  LEFT_TO_RIGHT, 2 );

      break;

    case '>':

      pushOperator_ ( GT_RANK,  Opcodes::GT,  LEFT_TO_RIGHT, 2 );

      break;

    case Tokens::GE:

      pushOperator_ ( GE_RANK,  Opcodes::GE,  LEFT_TO_RIGHT, 2 );

      break;

    case '+':

      if ( mode_ & RVAL_MODE )
      {
        pushOperator_ ( ADD_RANK, Opcodes::ADD, LEFT_TO_RIGHT, 2 );
      }

      break;

    case '-':

      if ( mode_ & RVAL_MODE )
      {
        pushOperator_ ( SUB_RANK, Opcodes::SUB, LEFT_TO_RIGHT, 2 );
      }
      else
      {
        pushOperator_ ( NEG_RANK, Opcodes::NEG, RIGHT_TO_LEFT, 1 );
      }

      break;

    case '!':

      pushOperator_ ( NOT_RANK, Opcodes::NOT, RIGHT_TO_LEFT, 1 );

      break;

    case '*':

      pushOperator_ ( MUL_RANK, Opcodes::MUL, LEFT_TO_RIGHT, 2 );

      break;

    case '/':

      pushOperator_ ( DIV_RANK, Opcodes::DIV, LEFT_TO_RIGHT, 2 );

      break;

    case '%':

      pushOperator_ ( MOD_RANK, Opcodes::MOD, LEFT_TO_RIGHT, 2 );

      break;

    case '^':

      parsePowExpr_ ();

      break;

    case Tokens::PI:

      pushLiteral_ ( Constants::PI );

      break;

    case Tokens::EPS:

      pushLiteral_ ( Constants::EPS );

      break;

    case Tokens::BIGNUM:

      pushLiteral_ ( Constants::BIGNUM );

      break;

    case Tokens::TINY:

      pushLiteral_ ( Constants::TINY );

      break;

    case Tokens::TRUE:

      pushLiteral_ ( Constants::TRUE );

      break;

    case Tokens::FALSE:

      pushLiteral_ ( Constants::FALSE );

      break;

    case Tokens::DEFINED:

      parseDefined_ ( args );

      break;

    default:

      if ( Tokens::isBuiltin( token ) )
      {
        parseBuiltin_ ( token );
      }
      else
      {
        done = true;
      }
    }
  }

  endSubExpr_ ( Tokens::END );

  if ( depth_ != 0 )
  {
    throw Error ( JEM_FUNC, "non-zero expression depth" );
  }

  tokzer_.pushBackToken ();

  xopc = stackSize_ - stackBase_;

  if ( xopc == 0 )
  {
    parseError_ ( "empty expression" );
  }

  if ( xopc > 1 )
  {
    throw Error ( JEM_FUNC,
                  "oops, operand stack size larger than one" );
  }

  depth_ = depth0;
}


//-----------------------------------------------------------------------
//   parseSymbol_
//-----------------------------------------------------------------------


void Compiler::parseSymbol_ ( ArgList& args )
{
  String   name = tokzer_.getSymbol ();

  idx_t    index;


  enterMode_ ( RVAL_MODE );

  if ( tokzer_.nextToken() == '(' )
  {
    beginFuncCall_ ( declareFunc_( name ) );

    return;
  }

  tokzer_.pushBackToken ();

  index = findSymbol ( name, tmpMap_ );

  if ( index >= 0 )
  {
    if ( index <= 2 )
    {
      code_->writeCode ( (Opcode) (Opcodes::LOAD_TMP0 + index) );
    }
    else
    {
      code_->writeCode ( Opcodes::LOAD_TMP );
      code_->writeCode ( (Opcode) index );
    }

    pushOperand_ ();

    return;
  }

  index = findSymbol ( name, varMap_ );

  if ( index >= 0 &&
       code_->vars[index].kind != CodeBlock::UNDEFINED )
  {
    code_->writeCode ( Opcodes::LOAD_VAR );
    code_->writeCode ( (Opcode) index );

    pushOperand_ ();

    return;
  }

  index = args.getArgIndex ( name );

  if ( index >= 0 )
  {
    if ( index > (idx_t) maxOf<Opcode>() )
    {
      overflowError_ ( "function argument index out of range" );
    }

    if ( index >= code_->argCount )
    {
      code_->argCount = (idx_t) index + 1;
    }

    if ( index <= 2 )
    {
      code_->writeCode ( (Opcode) (Opcodes::LOAD_ARG0 + index) );
    }
    else
    {
      code_->writeCode ( Opcodes::LOAD_ARG );
      code_->writeCode ( (Opcode) index );
    }

    pushOperand_ ();

    return;
  }

  index = declareVar_ ( name );

  code_->writeCode ( Opcodes::TRAP_VAR );
  code_->writeCode ( (Opcode) index );

  pushOperand_ ();
}


//-----------------------------------------------------------------------
//   parseDefined_
//-----------------------------------------------------------------------


void Compiler::parseDefined_ ( ArgList& args )
{
  String  name;
  int     token;
  idx_t   index;


  token = tokzer_.nextToken ();

  if      ( token == Tokens::SYMBOL )
  {
    name = tokzer_.getSymbol ();
  }
  else if ( token == '(' )
  {
    if ( tokzer_.nextToken() != Tokens::SYMBOL )
    {
      parseError_ ( "expected a symbol" );
    }

    name = tokzer_.getSymbol ();

    if ( tokzer_.nextToken() != ')' )
    {
      parseError_ ( "expected a `)\'" );
    }
  }
  else
  {
    parseError_ ( "expected a `(\' or a symbol" );
  }

  index = findSymbol ( name, varMap_ );

  if ( index >= 0 &&
       code_->vars[index].kind != CodeBlock::UNDEFINED )
  {
    if ( tmpMap_.contains( name ) ||
         args.getArgIndex( name ) >= 0 )
    {
      pushLiteral_ ( 1.0 );
      return;
    }
  }

  enterMode_ ( RVAL_MODE );

  index = declareVar_ ( name );

  code_->writeCode ( Opcodes::DEFINED );
  code_->writeCode ( (Opcode) index );

  pushOperand_ ();
}


//-----------------------------------------------------------------------
//   parseBuiltin_
//-----------------------------------------------------------------------


void Compiler::parseBuiltin_ ( int token )
{
  Operator*  op;
  Opcode     opcode;
  idx_t      argc;


  enterMode_ ( RVAL_MODE );

  if ( tokzer_.nextToken() != '(' )
  {
    String  name = Tokens::toString ( token );

    parseError_ (
      String::format (
        "expected a `(\' after builtin function `%s\'", name
      )
    );
  }

  argc = 1;

  switch ( token )
  {
  case Tokens::ABS:   opcode = Opcodes::ABS;   break;
  case Tokens::SQRT:  opcode = Opcodes::SQRT;  break;
  case Tokens::CEIL:  opcode = Opcodes::CEIL;  break;
  case Tokens::FLOOR: opcode = Opcodes::FLOOR; break;
  case Tokens::EXP:   opcode = Opcodes::EXP;   break;
  case Tokens::LOG:   opcode = Opcodes::LOG;   break;
  case Tokens::LOG10: opcode = Opcodes::LOG10; break;
  case Tokens::SIN:   opcode = Opcodes::SIN;   break;
  case Tokens::COS:   opcode = Opcodes::COS;   break;
  case Tokens::TAN:   opcode = Opcodes::TAN;   break;
  case Tokens::ASIN:  opcode = Opcodes::ASIN;  break;
  case Tokens::ACOS:  opcode = Opcodes::ACOS;  break;
  case Tokens::ATAN:  opcode = Opcodes::ATAN;  break;

  case Tokens::POW:

    opcode = Opcodes::POW;
    argc   = 2;

    break;

  case Tokens::MAX:

    opcode = Opcodes::MAX;
    argc   = VARARGS + 1;

    break;

  case Tokens::MIN:

    opcode = Opcodes::MIN;
    argc   = VARARGS + 1;

    break;

  default:

    opcode = Opcodes::NOP;

    throw Error (
      JEM_FUNC,
      "unexpected builtin: " + Tokens::toString ( token )
    );
  }

  op = (Operator*) allocator_.alloc ();

  op->rank     =  CALL_RANK;
  op->token    =  token;
  op->opcode   =  opcode;
  op->argCount =  argc;
  op->index    = -1;

  pushOperator_ ( op );
  beginSubExpr_ ();
  enterMode_    ( CALL_MODE );
}


//-----------------------------------------------------------------------
//   parsePowExpr_
//-----------------------------------------------------------------------


void Compiler::parsePowExpr_ ()
{
  if ( tokzer_.nextToken() == Tokens::NUMBER )
  {
    String  token = tokzer_.getTokenString ();

    if ( token == "2" || token == "2.0" )
    {
      pushOperator_ ( POW_RANK, Opcodes::POW2, LEFT_TO_RIGHT, 1 );
      enterMode_    ( RVAL_MODE );
      return;
    }

    if ( token == "3" || token == "3.0" )
    {
      pushOperator_ ( POW_RANK, Opcodes::POW3, LEFT_TO_RIGHT, 1 );
      enterMode_    ( RVAL_MODE );
      return;
    }
  }

  tokzer_.pushBackToken ();

  pushOperator_ ( POW_RANK, Opcodes::POW, LEFT_TO_RIGHT, 2 );
}


//-----------------------------------------------------------------------
//   beginSubExpr_
//-----------------------------------------------------------------------


void Compiler::beginSubExpr_ ( int token )
{
  Operator*  op = (Operator*) allocator_.alloc ();

  op->rank      = NULL_RANK;
  op->token     = token;
  op->opcode    = Opcodes::NOP;
  op->mode      = mode_;
  op->stackBase = stackBase_;

  pushOperator_ ( op );

  mode_         = 0;
  stackBase_    = stackSize_;

  depth_++;
}


//-----------------------------------------------------------------------
//   endSubExpr_
//-----------------------------------------------------------------------


idx_t Compiler::endSubExpr_ ( int token )
{
  Operator*  op;
  idx_t      xopc;


  while ( opStack_.size() && opStack_.back()->rank >= MIN_RANK )
  {
    popOperator_ ();
  }

  if ( ! opStack_.size() )
  {
    parseError_ ();
  }

  op = opStack_.back ();

  if ( op->token != token )
  {
    if      ( token != Tokens::END )
    {
      parseError_ ();
    }
    else if ( op->token > 0 )
    {
      parseError_ (
        String::format ( "unterminated `%c\'", op->token )
      );
    }
    else
    {
      parseError_ ( "unexpected end of expression" );
    }
  }

  xopc       = stackSize_ - stackBase_;
  mode_      = op->mode;
  stackBase_ = op->stackBase;

  opStack_  .popBack ();
  allocator_.dealloc ( op );

  depth_--;

  JEM_ASSERT ( depth_ >= 0 );

  return xopc;
}


//-----------------------------------------------------------------------
//   endAbsExpr_
//-----------------------------------------------------------------------


void Compiler::endAbsExpr_ ()
{
  const idx_t  xopc = endSubExpr_ ( '|' );

  if ( xopc != 1 )
  {
    parseError_ (
      String::format (
        "wrong number of parameters (%d) in abs operator", xopc
      )
    );
  }

  code_->writeCode ( Opcodes::ABS );

  popOperand_  ();
  pushOperand_ ();
}


//-----------------------------------------------------------------------
//   beginFuncCall_
//-----------------------------------------------------------------------


void Compiler::beginFuncCall_ ( idx_t index )
{
  Operator*  op = (Operator*) allocator_.alloc ();

  op->rank      = CALL_RANK;
  op->token     = Tokens::NONE;
  op->opcode    = Opcodes::TRAP_FUNC;
  op->argCount  = VARARGS;
  op->index     = index;

  pushOperator_ ( op );
  beginSubExpr_ ();
  enterMode_    ( CALL_MODE );
}


//-----------------------------------------------------------------------
//   endFuncCall_
//-----------------------------------------------------------------------


void Compiler::endFuncCall_ ()
{
  JEM_ASSERT ( mode_ & CALL_MODE );

  Operator*  op;
  Opcode     opcode;
  int        token;
  idx_t      argc;
  idx_t      index;
  idx_t      xopc;


  xopc   = endSubExpr_   ();
  op     = opStack_.back ();

  token  = op->token;
  opcode = op->opcode;
  argc   = op->argCount;
  index  = op->index;

  allocator_.dealloc ( op );
  opStack_  .popBack ();

  if ( (argc >= 0 && xopc != argc) ||
       (argc <  0 && xopc < (argc - VARARGS)) )
  {
    String  name = getFuncName_ ( index, token );
    String  hint;

    if ( argc < 0 )
    {
      hint = String::format ( "(expected at least %d)",
                              argc - VARARGS );
    }
    else
    {
      hint = String::format ( "(expected %d)", argc );
    }

    parseError_ (
      String::format (
        "wrong number of parameters in call of function `%s\' ",
        name
      ) + hint
    );
  }

  if ( opcode == Opcodes::MAX && xopc == 2 )
  {
    opcode = Opcodes::MAX2;
    argc   = 2;
  }

  if ( opcode == Opcodes::MIN && xopc == 2 )
  {
    opcode = Opcodes::MIN2;
    argc   = 2;
  }

  code_->writeCode ( opcode );

  if ( index >= 0 )
  {
    code_->writeCode ( (Opcode) index );
  }

  if ( argc < 0 )
  {
    if ( xopc > (idx_t) maxOf<Opcode>() )
    {
      String  name = getFuncName_ ( index, token );

      overflowError_ (
        String::format (
          "too many parameters in call of function `%s\'",
          name
        )
      );
    }

    code_->writeCode ( (Opcode) xopc );
  }

  stackSize_ -= xopc;

  pushOperand_ ();
}


//-----------------------------------------------------------------------
//   beginIfExpr_
//-----------------------------------------------------------------------


void Compiler::beginIfExpr_ ()
{
  Operator*  op;


  if ( tokzer_.nextToken() != '(' )
  {
    parseError_ ( "expected a `(\'" );
  }

  enterMode_ ( RVAL_MODE );

  op = (Operator*) allocator_.alloc ();

  op->rank      = NULL_RANK;
  op->token     = Tokens::IF;
  op->opcode    = Opcodes::NOP;
  op->mode      = mode_;
  op->firstJump = jmpStack_.size ();

  pushOperator_ ( op );
  beginSubExpr_ ();
  enterMode_    ( IF_MODE );
}


//-----------------------------------------------------------------------
//   beginThenExpr_
//-----------------------------------------------------------------------


void Compiler::beginThenExpr_ ()
{
  if ( ! (mode_ & IF_MODE) )
  {
    parseError_ ();
  }

  if ( endSubExpr_() == 0 )
  {
    parseError_ ( "empty test expression" );
  }

  JEM_ASSERT ( opStack_.back()->token == Tokens::IF );

  popOperand_ ();

  code_->opcodes.pushBack ( Opcodes::JUMP_IF_FALSE );
  code_->opcodes.pushBack ( -1 );

  jmpStack_.pushBack ( code_->opcodes.size() - 1 );

  beginSubExpr_ ( Tokens::IF );
  enterMode_    ( THEN_MODE );
}


//-----------------------------------------------------------------------
//   endThenExpr_
//-----------------------------------------------------------------------


void Compiler::endThenExpr_ ()
{
  idx_t  dist;
  idx_t  addr;
  idx_t  jmp;


  if ( ! (mode_ & THEN_MODE) )
  {
    parseError_ ();
  }

  if ( endSubExpr_( Tokens::IF ) == 0 )
  {
    parseError_ ( "empty if-expression" );
  }

  JEM_ASSERT ( opStack_.back()->token == Tokens::IF );

  popOperand_ ();

  code_->writeCode ( Opcodes::JUMP );
  code_->writeCode ( -1 );

  addr = code_->opcodes.size ();
  jmp  = jmpStack_     .back ();
  dist = addr - jmp;

  if ( dist > (idx_t) maxOf<Opcode>() )
  {
    overflowError_ ( "jump distance out of range" );
  }

  code_->opcodes[jmp] = (Opcode) dist;

  jmpStack_.back() = addr - 1;
}


//-----------------------------------------------------------------------
//   beginElseExpr_
//-----------------------------------------------------------------------


void Compiler::beginElseExpr_ ()
{
  endThenExpr_  ();
  beginSubExpr_ ( Tokens::ELSE );
  enterMode_    ( ELSE_MODE );
}


//-----------------------------------------------------------------------
//   endElseExpr_
//-----------------------------------------------------------------------


void Compiler::endElseExpr_ ()
{
  Operator*  op;
  idx_t      dist;
  idx_t      addr;
  idx_t      jmpc;
  idx_t      jmp;


  if ( ! (mode_ & ELSE_MODE) )
  {
    parseError_ ();
  }

  if ( endSubExpr_( Tokens::ELSE ) == 0 )
  {
    parseError_ ( "empty else-expression" );
  }

  JEM_ASSERT ( opStack_.back()->token == Tokens::IF );

  op   = opStack_.back       ();
  addr = code_->opcodes.size ();
  jmpc = jmpStack_.size() - op->firstJump;

  for ( ; jmpc > 0; jmpc-- )
  {
    jmp  = jmpStack_.back ();
    dist = addr - jmp;

    if ( dist > (idx_t) maxOf<Opcode>() )
    {
      overflowError_ ( "jump distance out of range" );
    }

    code_->opcodes[jmp] = (Opcode) dist;

    jmpStack_.popBack ();
  }

  mode_ = op->mode;

  opStack_  .popBack ();
  allocator_.dealloc ( op );
}


//-----------------------------------------------------------------------
//   pushOperator_
//-----------------------------------------------------------------------


void Compiler::pushOperator_

  ( int     rank,
    Opcode  opcode,
    Assoc   assoc,
    idx_t   argc )

{
  JEM_ASSERT ( rank >= MIN_RANK );

  Operator*  op;


  if ( argc > 1 && ! (mode_ & RVAL_MODE) )
  {
    parseError_ ();
  }

  while ( opStack_.size() && opStack_.back()->rank > rank )
  {
    popOperator_ ();
  }

  if ( assoc == LEFT_TO_RIGHT )
  {
    while ( opStack_.size() && opStack_.back()->rank == rank )
    {
      popOperator_ ();
    }
  }

  op = (Operator*) allocator_.alloc ();

  op->rank      =  rank;
  op->token     =  tokzer_.getToken ();
  op->opcode    =  opcode;
  op->index     = -1;
  op->argCount  =  argc;

  pushOperator_ ( op );

  mode_ &= ~RVAL_MODE;
}


inline void Compiler::pushOperator_ ( Operator* op )
{
  if ( opStack_.size() >= maxOf<Opcode>() )
  {
    overflowError_ ( "operator stack overflow" );
  }

  try
  {
    opStack_.pushBack ( op );
  }
  catch ( ... )
  {
    allocator_.dealloc ( op );
    throw;
  }
}


//-----------------------------------------------------------------------
//   popOperator_
//-----------------------------------------------------------------------


void Compiler::popOperator_ ()
{
  Operator*  op     = opStack_.back ();

  Opcode     opcode = op->opcode;
  int        token  = op->token;
  idx_t      argc   = op->argCount;


  allocator_.dealloc ( op );
  opStack_  .popBack ();

  if ( stackSize_ - stackBase_ < argc )
  {
    String  name = Tokens::toString ( token );

    parseError_ (
      String::format ( "operand missing for operator `%s\'", name )
    );
  }

  if ( opcode >= 0 )
  {
    stackSize_ -= argc;

    pushOperand_ ();

    code_->writeCode ( opcode );
  }
}


//-----------------------------------------------------------------------
//   pushLiteral_
//-----------------------------------------------------------------------


void Compiler::pushLiteral_ ( double value )
{
  const idx_t  n = code_->vars.size ();

  idx_t        index;


  enterMode_ ( RVAL_MODE );

  index = -1;

  for ( int i = 0; i < n; i++ )
  {
    CodeBlock::Var&  v= code_->vars[i];

    if ( v.name.size() == 0 && isTiny( v.value - value ) )
    {
      index = i;
      break;
    }
  }

  if ( index < 0 )
  {
    index = code_->addLiteral ( value );

    if ( index > (idx_t) maxOf<Opcode>() )
    {
      overflowError_ ( "too many literals" );
    }
  }

  code_->writeCode ( Opcodes::LOAD_VAR );
  code_->writeCode ( (Opcode) index );

  pushOperand_ ();
}


//-----------------------------------------------------------------------
//   declareTmp_
//-----------------------------------------------------------------------


idx_t Compiler::declareTmp_ ( const String& name )
{
  idx_t  index = findSymbol ( name, tmpMap_ );

  if ( index < 0 )
  {
    index = tmpMap_.size ();

    if ( index > (idx_t) maxOf<Opcode>() )
    {
      overflowError_ ( "too many local variables" );
    }

    tmpMap_.insert ( makePair( name, index ) );
  }

  return index;
}


//-----------------------------------------------------------------------
//   declareVar_
//-----------------------------------------------------------------------


idx_t Compiler::declareVar_ ( const String& name )
{
  idx_t  index = findSymbol ( name, varMap_ );

  if ( index < 0 )
  {
    index = code_->addVar ( name );

    if ( index > (idx_t) maxOf<Opcode>() )
    {
      overflowError_ ( "too many external variable references" );
    }

    varMap_.insert ( makePair( name, index ) );
  }

  return index;
}


//-----------------------------------------------------------------------
//   declareFunc_
//-----------------------------------------------------------------------


idx_t Compiler::declareFunc_ ( const String& name )
{
  idx_t  index = findSymbol ( name, funcMap_ );

  if ( index < 0 )
  {
    index = code_->addFunc ( name );

    if ( index > (idx_t) maxOf<Opcode>() )
    {
      overflowError_ ( "too many external function references" );
    }

    funcMap_.insert ( makePair( name, index ) );
  }

  return index;
}


//-----------------------------------------------------------------------
//   getFuncName_
//-----------------------------------------------------------------------


String Compiler::getFuncName_

  ( idx_t  index,
    int    token ) const

{
  if ( index >= 0 )
  {
    return code_->funcs[index].name;
  }
  else
  {
    return Tokens::toString ( token );
  }
}


//-----------------------------------------------------------------------
//   pushOperand_
//-----------------------------------------------------------------------


inline void Compiler::pushOperand_ ()
{
  if ( stackSize_ >= maxOf( stackSize_ ) )
  {
    overflowError_ ( "operand stack overflow" );
  }

  stackSize_++;

  if ( stackSize_ > code_->maxSSize )
  {
    code_->maxSSize = stackSize_;
  }
}


//-----------------------------------------------------------------------
//   popOperand_
//-----------------------------------------------------------------------


inline void Compiler::popOperand_ ()
{
  stackSize_--;

  JEM_ASSERT ( stackSize_ >= stackBase_ );
}


//-----------------------------------------------------------------------
//   enterMode_
//-----------------------------------------------------------------------


inline void Compiler::enterMode_ ( int mode )
{
  if ( mode_ & mode )
  {
    parseError_ ();
  }

  mode_ |= mode;
}


//-----------------------------------------------------------------------
//   parseError_
//-----------------------------------------------------------------------


void Compiler::parseError_ () const
{
  String  tok = tokzer_.getTokenString ();
  String  ex  = StringUtils::shorten   ( code_->expr, ERR_WIDTH );

  throw ParseException (
    ERR_CONTEXT,
    String::format (
      "invalid expression `%s\': parse error before `%s\'",
      ex,
      tok
    )
  );
}


void Compiler::parseError_ ( const String& what ) const
{
  String  ex = StringUtils::shorten ( code_->expr, ERR_WIDTH );

  throw ParseException (
    ERR_CONTEXT,
    String::format (
      "invalid expression `%s\': %s", ex, what
    )
  );
}


//-----------------------------------------------------------------------
//   overflowError_
//-----------------------------------------------------------------------


void Compiler::overflowError_ ( const String& what ) const
{
  String  ex = StringUtils::shorten ( code_->expr, ERR_WIDTH );

  throw IllegalInputException (
    ERR_CONTEXT,
    String::format (
      "invalid expression `%s\': %s", ex, what
    )
  );
}


JEM_END_UFUNC_IMP
