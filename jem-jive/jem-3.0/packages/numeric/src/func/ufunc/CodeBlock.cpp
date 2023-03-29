
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
#include <jem/ccident.h>
#include <jem/base/limits.h>
#include <jem/base/Class.h>
#include <jem/base/Error.h>
#include <jem/base/Float.h>
#include <jem/base/StringBuffer.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/StringUtils.h>
#include <jem/numeric/func/FuncException.h>
#include "Opcodes.h"
#include "CodeBlock.h"


#undef JEM_USE_COMPUTED_GOTO


#if ( defined(JEM_CXX_GCC) )
#  define JEM_USE_COMPUTED_GOTO
#endif


JEM_BEGIN_UFUNC_IMP


using jem::util::StringUtils;


//=======================================================================
//   private helper functions
//=======================================================================

//-----------------------------------------------------------------------
//   maxloc
//-----------------------------------------------------------------------


static idx_t      maxloc

  ( const double*   v,
    idx_t           n )

{
  double  x = v[0];
  idx_t   j = 0;

  for ( idx_t i = 1; i < n; i++ )
  {
    if ( v[i] > x )
    {
      x = v[i];
      j = i;
    }
  }

  return j;
}


//-----------------------------------------------------------------------
//   minloc
//-----------------------------------------------------------------------


static idx_t      minloc

  ( const double*   v,
    idx_t           n )

{
  double  x = v[0];
  idx_t   j = 0;

  for ( idx_t i = 1; i < n; i++ )
  {
    if ( v[i] < x )
    {
      x = v[i];
      j = i;
    }
  }

  return j;
}


//-----------------------------------------------------------------------
//   toBool
//-----------------------------------------------------------------------


inline double     toBool

  ( double          x )

{
  return ( (x * x < 0.25) ? 0.0 : 1.0 );
}


//-----------------------------------------------------------------------
//   isTrue
//-----------------------------------------------------------------------


inline bool       isTrue

  ( double          x )

{
  return ( (x * x < 0.25) ? false : true );
}


//-----------------------------------------------------------------------
//   isFalse
//-----------------------------------------------------------------------


inline bool       isFalse

  ( double          x )

{
  return ( (x * x < 0.25) ? true : false );
}


//=======================================================================
//   class CodeBlock
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


CodeBlock::CodeBlock ( const String& e ) : expr ( e )
{
  status   = RESOLVED;
  argCount = 0;
  tmpCount = 0;
  extCount = 0;
  maxSSize = 0;
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void CodeBlock::readFrom ( ObjectInput& in )
{
  uidx_t  sz;
  int     stat;
  int     kind;
  idx_t   n;


  decode ( in, expr );
  decode ( in, stat );
  decode ( in, argCount );
  decode ( in, tmpCount );
  decode ( in, extCount );
  decode ( in, maxSSize );
  decode ( in, opcodes  );

  status = (Status) stat;

  decode ( in, sz );

  n = (idx_t) sz;

  vars.resize ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    Var&  v = vars[i];

    decode ( in, v.name, v.value, kind );

    v.kind = (VarKind) kind;
  }

  decode ( in, sz );

  n = (idx_t) sz;

  funcs.resize ( n );

  for ( idx_t i = 0; i < n; i++ )
  {
    Func&  f = funcs[i];

    decode ( in, f.name, f.addr );
  }
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void CodeBlock::writeTo ( ObjectOutput& out ) const
{
  encode ( out, expr );
  encode ( out, (int) status );
  encode ( out, argCount );
  encode ( out, tmpCount );
  encode ( out, extCount );
  encode ( out, maxSSize );
  encode ( out, opcodes  );

  idx_t  n = vars.size ();

  encode ( out, n );

  for ( idx_t i = 0; i < n; i++ )
  {
    const Var&  v = vars[i];

    encode ( out, v.name, v.value, (int) v.kind );
  }

  n = funcs.size ();

  encode ( out, n );

  for ( idx_t i = 0; i < n; i++ )
  {
    const Func&  f = funcs[i];

    encode ( out, f.name, f.addr );
  }
}


//-----------------------------------------------------------------------
//   resolve
//-----------------------------------------------------------------------


void CodeBlock::resolve

  ( const NameSpace&  symbols,
    int               options )

{
  if ( extCount == 0 )
  {
    status = RESOLVED;
    return;
  }

  bool   found;
  idx_t  i, n;


  status = RESOLVING;
  n      = vars.size ();

  if ( n > 0 )
  {
    for ( i = 0; i < n; i++ )
    {
      Var&  v = vars[i];

      if ( ! (v.kind == UNDEFINED || v.kind == EXTERNAL) )
      {
        continue;
      }

      v.kind = UNDEFINED;
      found  = false;

      try
      {
        found  = symbols.getVariable ( v.value, v.name );
      }
      catch ( const Exception& ex )
      {
        resolveError_ ( ex.what() );
      }

      if      ( found )
      {
        v.kind = EXTERNAL;
      }
      else if ( options & UserFunc::RESOLVE_STRICT )
      {
        resolveError_ ( v );
      }
    }
  }

  n = funcs.size ();

  if ( n > 0 )
  {
    for ( i = 0; i < n; i++ )
    {
      Func&  f = funcs[i];

      try
      {
        f.addr = symbols.getFunction ( f.name );
      }
      catch ( const Exception& ex )
      {
        resolveError_ ( ex.what() );
      }

      if ( f.addr )
      {
        UserFunc*  ufunc = dynamicCast<UserFunc*> ( f.addr );

        if ( ufunc )
        {
          ufunc->resolve ( symbols );
        }
      }
      else if ( options & UserFunc::RESOLVE_STRICT )
      {
        resolveError_ ( f );
      }
    }
  }

  resolve_ ();

  status = RESOLVED;
}


//-----------------------------------------------------------------------
//   unresolve
//-----------------------------------------------------------------------


void CodeBlock::unresolve ()
{
  idx_t  i, n;


  status = RESOLVING;
  n      = vars.size ();

  for ( i = 0; i < n; i++ )
  {
    Var&  v = vars[i];

    if ( v.kind == EXTERNAL )
    {
      v.kind = UNDEFINED;
    }
  }

  n = funcs.size ();

  if ( n > 0 )
  {
    for ( i = 0; i < n; i++ )
    {
      UserFunc*  u = dynamicCast<UserFunc*> ( funcs[i].addr );

      if ( u )
      {
        u->unresolve ();
      }

      funcs[i].addr = nullptr;
    }
  }

  status = UNRESOLVED;
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void CodeBlock::trimToSize ()
{
  opcodes.trimToSize ();
  vars   .trimToSize ();
  funcs  .trimToSize ();
}


//-----------------------------------------------------------------------
//   calcValue
//-----------------------------------------------------------------------


#undef JEM_CASE_OP
#undef JEM_NEXT_OP

#ifdef JEM_USE_COMPUTED_GOTO
#  define JEM_CASE_OP( OP )  OP
#  define JEM_NEXT_OP        goto *jtb[*ops++]
#else
#  define JEM_CASE_OP( OP )  case Opcodes::OP
#  define JEM_NEXT_OP        break
#endif


double CodeBlock::calcValue

  ( double*        vbuf,
    const double*  args )

{
#ifdef JEM_USE_COMPUTED_GOTO

  static const void*  JUMP_TABLE[Opcodes::OPCODE_COUNT] =
  {
    JEM_LIST_OPCODES( && )
  };

  const void**  JEM_RESTRICT  jtb = JUMP_TABLE;

#endif

  const Opcode* JEM_RESTRICT  ops = opcodes.addr ();

  double*       JEM_RESTRICT  sp  = vbuf;
  double*       JEM_RESTRICT  tmp = vbuf + maxSSize;

  Function*     func;
  double        x;
  idx_t         k;


  sp--;

#ifdef JEM_USE_COMPUTED_GOTO

  goto *jtb[*ops++];

#else

  while ( true )
  {
    switch ( *ops++ )
    {

#endif

  JEM_CASE_OP( NOP ):

    JEM_NEXT_OP;

  JEM_CASE_OP( RET ):

    goto done;

  JEM_CASE_OP( OR ):

    sp--;

    sp[0] = toBool ( std::fabs(sp[0]) + std::fabs(sp[1]) );

    JEM_NEXT_OP;

  JEM_CASE_OP( AND ):

    sp--;

    sp[0] = toBool ( sp[0] * sp[1] );

    JEM_NEXT_OP;

  JEM_CASE_OP( LT ):

    sp--;

    sp[0] = (sp[0] < sp[1]) ? 1.0 : 0.0;

    JEM_NEXT_OP;

  JEM_CASE_OP( LE ):

    sp--;

    sp[0] = (sp[0] <= sp[1]) ? 1.0 : 0.0;

    JEM_NEXT_OP;

  JEM_CASE_OP( GT ):

    sp--;

    sp[0] = (sp[0] > sp[1]) ? 1.0 : 0.0;

    JEM_NEXT_OP;

  JEM_CASE_OP( GE ):

    sp--;

    sp[0] = (sp[0] >= sp[1]) ? 1.0 : 0.0;

    JEM_NEXT_OP;

  JEM_CASE_OP( ADD ):

    sp--;

    sp[0] += sp[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( SUB ):

    sp--;

    sp[0] -= sp[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( MUL ):

    sp--;

    sp[0] *= sp[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( DIV ):

    sp--;

    if ( isTiny( sp[1] ) )
    {
      evalError_ ( "division by zero", args );
    }

    sp[0] /= sp[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( MOD ):

    sp--;

    if ( isTiny( sp[1] ) )
    {
      evalError_ ( "modulo division by zero", args );
    }

    sp[0] = std::fmod ( sp[0], sp[1] );

    JEM_NEXT_OP;

  JEM_CASE_OP( NEG ):

    sp[0] = -sp[0];

    JEM_NEXT_OP;

  JEM_CASE_OP( NOT ):

    sp[0] = isTrue( sp[0] ) ? 0.0 : 1.0;

    JEM_NEXT_OP;

  JEM_CASE_OP( ABS ):

    sp[0] = std::fabs ( sp[0] );

    JEM_NEXT_OP;

  JEM_CASE_OP( SQRT ):

    if ( sp[0] < 0.0 )
    {
      funcError_ ( "sqrt", sp[0] );
    }

    sp[0] = std::sqrt ( sp[0] );

    JEM_NEXT_OP;

  JEM_CASE_OP( CEIL ):

    sp[0] = std::ceil ( sp[0] );

    JEM_NEXT_OP;

  JEM_CASE_OP( FLOOR ):

    sp[0] = std::floor ( sp[0] );

    JEM_NEXT_OP;

  JEM_CASE_OP( POW ):

    sp--;

    sp[0] = evalPow_ ( sp[0], sp[1] );

    JEM_NEXT_OP;

  JEM_CASE_OP( POW2 ):

    sp[0] *= sp[0];

    JEM_NEXT_OP;

  JEM_CASE_OP( POW3 ):

    sp[0] *= sp[0] * sp[0];

   JEM_NEXT_OP;

  JEM_CASE_OP( EXP ):

    x = std::exp ( sp[0] );

    // Check for overflow.

    if ( ! (x <= maxOf<double>()) )
    {
      funcError_ ( "exp", sp[0] );
    }

    sp[0] = x;

    JEM_NEXT_OP;

  JEM_CASE_OP( LOG ):

    if ( sp[0] <= 0.0 )
    {
      funcError_ ( "log", sp[0] );
    }

    sp[0] = std::log ( sp[0] );

    JEM_NEXT_OP;

  JEM_CASE_OP( LOG10 ):

    if ( sp[0] <= 0.0 )
    {
      funcError_ ( "log10", sp[0] );
    }

    sp[0] = std::log10 ( sp[0] );

    JEM_NEXT_OP;

  JEM_CASE_OP( SIN ):

    sp[0] = std::sin ( sp[0] );

    JEM_NEXT_OP;

  JEM_CASE_OP( COS ):

    sp[0] = std::cos ( sp[0] );

    JEM_NEXT_OP;

  JEM_CASE_OP( TAN ):

    sp[0] = std::tan ( sp[0] );

    JEM_NEXT_OP;

  JEM_CASE_OP( ASIN ):

    if ( sp[0] < -1.0 || sp[0] > 1.0 )
    {
      funcError_ ( "asin", sp[0] );
    }

    sp[0] = std::asin ( sp[0] );

    JEM_NEXT_OP;

  JEM_CASE_OP( ACOS ):

    if ( sp[0] < -1.0 || sp[0] > 1.0 )
    {
      funcError_ ( "acos", sp[0] );
    }

    sp[0] = std::acos ( sp[0] );

    JEM_NEXT_OP;

  JEM_CASE_OP( ATAN ):

    sp[0] = std::atan ( sp[0] );

    JEM_NEXT_OP;

  JEM_CASE_OP( MAX2 ):

    sp--;

    sp[0] = (sp[0] > sp[1]) ? sp[0] : sp[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( MIN2 ):

    sp--;

    sp[0] = (sp[0] < sp[1]) ? sp[0] : sp[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_ARG0 ):

    sp++;

    sp[0] = args[0];

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_ARG1 ):

    sp++;

    sp[0] = args[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_ARG2 ):

    sp++;

    sp[0] = args[2];

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_TMP0 ):

    sp++;

    sp[0] = tmp[0];

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_TMP1 ):

    sp++;

    sp[0] = tmp[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_TMP2 ):

    sp++;

    sp[0] = tmp[2];

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_ARG ):

    sp++;

    sp[0] = args[*ops++];

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_VAR ):

    sp++;

    sp[0] = vars[*ops++].value;

    JEM_NEXT_OP;

  JEM_CASE_OP( TRAP_VAR ):

    trapError_ ( vars[*ops++] );

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_TMP ):

    sp++;

    sp[0] = tmp[*ops++];

    JEM_NEXT_OP;

  JEM_CASE_OP( STORE_TMP ):

    sp--;

    tmp[*ops++] = sp[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( STORE_VAR ):

    sp--;

    vars[*ops++].value = sp[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( JUMP ):

    ops += ops[0];

    JEM_NEXT_OP;

  JEM_CASE_OP( JUMP_IF_FALSE ):

    sp--;

    if ( isFalse( sp[1] ) )
    {
      ops += ops[0];
    }
    else
    {
      // Skip the jump distance.

      ops++;
    }

    JEM_NEXT_OP;

  JEM_CASE_OP( MAX ):

    k   = *ops++;
    sp -= (k - 1);
    k   = maxloc ( sp, k );

    sp[0] = sp[k];

    JEM_NEXT_OP;

  JEM_CASE_OP( MIN ):

    k   = *ops++;
    sp -= (k - 1);
    k   = minloc ( sp, k );

    sp[0] = sp[k];

    JEM_NEXT_OP;

  JEM_CASE_OP( DEFINED ):

    sp++;

    sp[0] = (vars[*ops++].kind == UNDEFINED) ? 0.0 : 1.0;

    JEM_NEXT_OP;

  JEM_CASE_OP( CALL_FUNC ):

    func = funcs[*ops++].addr.get ();
    k    = *ops++;
    sp  -= (k - 1);

    sp[0] = func->getValue ( sp );

    JEM_NEXT_OP;

  JEM_CASE_OP( TRAP_FUNC ):

    trapError_ ( funcs[*ops++] );

#ifndef JEM_USE_COMPUTED_GOTO

    default:

      opcodeError_ ( ops[-1] );
    }
  }

#endif

 done:

  x = sp[0];

  // Make sure that the result is a valid number.

  if ( Float::isNaN( x ) )
  {
    evalError_ ( "value out of range", args );
  }

  return x;
}


//-----------------------------------------------------------------------
//   calcDeriv
//-----------------------------------------------------------------------


double CodeBlock::calcDeriv

  ( double*        vbuf,
    idx_t          iarg,
    const double*  args )

{
#ifdef JEM_USE_COMPUTED_GOTO

  static const void*  JUMP_TABLE[Opcodes::OPCODE_COUNT] =
  {
    JEM_LIST_OPCODES( && )
  };

  const void**  JEM_RESTRICT  jtb = JUMP_TABLE;

#endif

  const Opcode* JEM_RESTRICT  ops = opcodes.addr ();

  double*       JEM_RESTRICT  sp  = vbuf;
  double*       JEM_RESTRICT  dp  = sp  + maxSSize;
  double*       JEM_RESTRICT  tmp = dp  + maxSSize;
  double*       JEM_RESTRICT  dmp = tmp + tmpCount;

  Function*     func;
  double        x, y;
  idx_t         k;


  dp--; sp--;

#ifdef JEM_USE_COMPUTED_GOTO

  goto *jtb[*ops++];

#else

  while ( true )
  {
    switch ( *ops++ )
    {

#endif

  JEM_CASE_OP( NOP ):

    JEM_NEXT_OP;

  JEM_CASE_OP( RET ):

    goto done;

  JEM_CASE_OP( OR ):

    dp--; sp--;

    dp[0] = 0.0;
    sp[0] = toBool ( std::fabs(sp[0]) + std::fabs(sp[1]) );

    JEM_NEXT_OP;

  JEM_CASE_OP( AND ):

    dp--; sp--;

    dp[0] = 0.0;
    sp[0] = toBool ( sp[0] * sp[1] );

    JEM_NEXT_OP;

  JEM_CASE_OP( LT ):

    dp--; sp--;

    dp[0] = 0.0;
    sp[0] = (sp[0] < sp[1]) ? 1.0 : 0.0;

    JEM_NEXT_OP;

  JEM_CASE_OP( LE ):

    dp--; sp--;

    dp[0] = 0.0;
    sp[0] = (sp[0] <= sp[1]) ? 1.0 : 0.0;

    JEM_NEXT_OP;

  JEM_CASE_OP( GT ):

    dp--; sp--;

    dp[0] = 0.0;
    sp[0] = (sp[0] > sp[1]) ? 1.0 : 0.0;

    JEM_NEXT_OP;

  JEM_CASE_OP( GE ):

    dp--; sp--;

    dp[0] = 0.0;
    sp[0] = (sp[0] >= sp[1]) ? 1.0 : 0.0;

    JEM_NEXT_OP;

  JEM_CASE_OP( ADD ):

    dp--; sp--;

    dp[0] += dp[1];
    sp[0] += sp[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( SUB ):

    dp--; sp--;

    dp[0] -= dp[1];
    sp[0] -= sp[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( MUL ):

    dp--; sp--;

    dp[0]  = dp[0] * sp[1] + dp[1] * sp[0];
    sp[0] *= sp[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( DIV ):

    dp--; sp--;

    x = sp[1];

    if ( isTiny( x ) )
    {
      evalError_ ( "division by zero", args );
    }

    x = 1.0 / x;

    dp[0]  = x * (dp[0] - x * sp[0] * dp[1]);
    sp[0] *= x;

    JEM_NEXT_OP;

  JEM_CASE_OP( MOD ):

    dp--; sp--;

    if ( isTiny( sp[1] ) )
    {
      evalError_ ( "modulo division by zero", args );
    }

    x = std::fmod ( sp[0], sp[1] );

    dp[0] += (x - sp[0]) * dp[1] / sp[1];
    sp[0]  =  x;

    JEM_NEXT_OP;

  JEM_CASE_OP( NEG ):

    dp[0] = -dp[0];
    sp[0] = -sp[0];

    JEM_NEXT_OP;

  JEM_CASE_OP( NOT ):

    dp[0] = 0.0;
    sp[0] = isTrue( sp[0] ) ? 0.0 : 1.0;

    JEM_NEXT_OP;

  JEM_CASE_OP( ABS ):

    if ( sp[0] < 0 )
    {
      dp[0] = -dp[0];
      sp[0] = -sp[0];
    }

    JEM_NEXT_OP;

  JEM_CASE_OP( SQRT ):

    if ( sp[0] < 0.0 )
    {
      funcError_ ( "sqrt", sp[0] );
    }

    x = std::sqrt ( sp[0] );

    if ( ! isTiny( dp[0] ) )
    {
      if ( isTiny( x ) )
      {
        evalError_ ( "division by zero", args );
      }

      dp[0] /= 2.0 * x;
    }

    sp[0] = x;

    JEM_NEXT_OP;

  JEM_CASE_OP( CEIL ):

    dp[0] = 0.0;
    sp[0] = std::ceil ( sp[0] );

    JEM_NEXT_OP;

  JEM_CASE_OP( FLOOR ):

    dp[0] = 0.0;
    sp[0] = std::floor ( sp[0] );

    JEM_NEXT_OP;

  JEM_CASE_OP( POW ):

    dp--; sp--;

    x = evalPow_ ( sp[0], sp[1] );
    y = 0.0;

    if ( ! isTiny( dp[0] ) )
    {
      y += dp[0] * sp[1]  * evalPow_ ( sp[0], sp[1] - 1.0 );
    }

    if ( ! isTiny( dp[1] ) )
    {
      if ( sp[0] <= 0.0 )
      {
        funcError_ ( "log", sp[0] );
      }

      y += dp[1] * x * ::log ( sp[0] );
    }

    dp[0] = y;
    sp[0] = x;

    JEM_NEXT_OP;

  JEM_CASE_OP( POW2 ):

    x = sp[0];

    dp[0] *= x * 2.0;
    sp[0] *= x;

    JEM_NEXT_OP;

  JEM_CASE_OP( POW3 ):

    x = sp[0] * sp[0];

    dp[0] *= x * 3.0;
    sp[0] *= x;

    JEM_NEXT_OP;

  JEM_CASE_OP( EXP ):

    x = std::exp ( sp[0] );

    // Check for overflow.

    if ( ! (x <= maxOf<double>()) )
    {
      funcError_ ( "exp", sp[0] );
    }

    dp[0] *= x;
    sp[0]  = x;

    JEM_NEXT_OP;

  JEM_CASE_OP( LOG ):

    x = sp[0];

    if ( x <= 0.0 )
    {
      funcError_ ( "log", x );
    }

    dp[0] /= x;
    sp[0]  = std::log ( x );

    JEM_NEXT_OP;

  JEM_CASE_OP( LOG10 ):

    x = sp[0];

    if ( x <= 0.0 )
    {
      funcError_ ( "log10", x );
    }

    dp[0] /= x * std::log   ( 10.0 );
    sp[0]  =     std::log10 ( x );

    JEM_NEXT_OP;

  JEM_CASE_OP( SIN ):

    x = sp[0];

    dp[0] *= std::cos ( x );
    sp[0]  = std::sin ( x );

    JEM_NEXT_OP;

  JEM_CASE_OP( COS ):

    x = sp[0];

    dp[0] *= -std::sin ( x );
    sp[0]  =  std::cos ( x );

    JEM_NEXT_OP;

  JEM_CASE_OP( TAN ):

    x = sp[0];

    if ( ! isTiny( dp[0] ) )
    {
      y = std::cos ( x );

      if ( isTiny( y ) )
      {
        evalError_ ( "division by zero", args );
      }

      dp[0] /= y * y;
    }

    sp[0] = std::tan ( x );

    JEM_NEXT_OP;

  JEM_CASE_OP( ASIN ):

    x = sp[0];

    if ( x < -1.0 || x > 1.0 )
    {
      funcError_ ( "asin", x );
    }

    dp[0] /= std::sqrt ( 1.0 - x * x );
    sp[0]  = std::asin ( x );

    JEM_NEXT_OP;

  JEM_CASE_OP( ACOS ):

    x = sp[0];

    if ( x < -1.0 || x > 1.0 )
    {
      funcError_ ( "acos", x );
    }

    dp[0] /= -std::sqrt ( 1.0 - x * x );
    sp[0]  =  std::acos ( x );

    JEM_NEXT_OP;

  JEM_CASE_OP( ATAN ):

    x = sp[0];

    dp[0] /= 1.0 + x * x;
    sp[0]  = std::atan ( x );

    JEM_NEXT_OP;

  JEM_CASE_OP( MAX2 ):

    dp--; sp--;

    if ( sp[1] > sp[0] )
    {
      dp[0] = dp[1];
      sp[0] = sp[1];
    }

    JEM_NEXT_OP;

  JEM_CASE_OP( MIN2 ):

    dp--; sp--;

    if ( sp[0] > sp[1] )
    {
      dp[0] = dp[1];
      sp[0] = sp[1];
    }

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_ARG0 ):

    dp++; sp++;

    dp[0] = (iarg == 0) ? 1.0 : 0.0;
    sp[0] = args[0];

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_ARG1 ):

    dp++; sp++;

    dp[0] = (iarg == 1) ? 1.0 : 0.0;
    sp[0] = args[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_ARG2 ):

    dp++; sp++;

    dp[0] = (iarg == 2) ? 1.0 : 0.0;
    sp[0] = args[2];

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_TMP0 ):

    dp++; sp++;

    dp[0] = dmp[0];
    sp[0] = tmp[0];

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_TMP1 ):

    dp++; sp++;

    dp[0] = dmp[1];
    sp[0] = tmp[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_TMP2 ):

    dp++; sp++;

    dp[0] = dmp[2];
    sp[0] = tmp[2];

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_ARG ):

    dp++; sp++;

    k = *ops++;

    dp[0] = (iarg == k) ? 1.0 : 0.0;
    sp[0] = args[k];

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_VAR ):

    dp++; sp++;

    dp[0] = 0.0;
    sp[0] = vars[*ops++].value;

    JEM_NEXT_OP;

  JEM_CASE_OP( TRAP_VAR ):

    trapError_ ( vars[*ops++] );

    JEM_NEXT_OP;

  JEM_CASE_OP( LOAD_TMP ):

    dp++; sp++;

    k = *ops++;

    dp[0] = dmp[k];
    sp[0] = tmp[k];

    JEM_NEXT_OP;

  JEM_CASE_OP( STORE_TMP ):

    dp--; sp--;

    k = *ops++;

    dmp[k] = dp[1];
    tmp[k] = sp[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( STORE_VAR ):

    dp--; sp--;

    vars[*ops++].value = sp[1];

    JEM_NEXT_OP;

  JEM_CASE_OP( JUMP ):

    ops += ops[0];

    JEM_NEXT_OP;

  JEM_CASE_OP( JUMP_IF_FALSE ):

    dp--; sp--;

    if ( isFalse( sp[1] ) )
    {
      ops += ops[0];
    }
    else
    {
      // Skip the jump distance.

      ops++;
    }

    JEM_NEXT_OP;

  JEM_CASE_OP( MAX ):

    k   = *ops++;
    dp -= (k - 1);
    sp -= (k - 1);
    k   = maxloc ( sp, k );

    dp[0] = dp[k];
    sp[0] = sp[k];

    JEM_NEXT_OP;

  JEM_CASE_OP( MIN ):

    k   = *ops++;
    dp -= (k - 1);
    sp -= (k - 1);
    k   = minloc ( sp, k );

    dp[0] = dp[k];
    sp[0] = sp[k];

    JEM_NEXT_OP;

  JEM_CASE_OP( DEFINED ):

    dp++; sp++;

    dp[0] = 0.0;
    sp[0] = (vars[*ops++].kind == UNDEFINED) ? 0.0 : 1.0;

    JEM_NEXT_OP;

  JEM_CASE_OP( CALL_FUNC ):

    func = funcs[*ops++].addr.get ();
    k    = *ops++;
    dp  -= (k - 1);
    sp  -= (k - 1);
    x    = 0.0;

    for ( idx_t i = 0; i < k; i++ )
    {
      if ( ! isTiny( dp[i] ) )
      {
        x += dp[i] * func->getDeriv ( i, sp );
      }
    }

    dp[0] = x;
    sp[0] = func->getValue ( sp );

    JEM_NEXT_OP;

  JEM_CASE_OP( TRAP_FUNC ):

    trapError_ ( funcs[*ops++] );

#ifndef JEM_USE_COMPUTED_GOTO

    default:

      opcodeError_ ( ops[-1] );
    }
  }

#endif

 done:

  x = dp[0];

  // Make sure that the result is a valid number.

  if ( Float::isNaN( x ) )
  {
    evalError_ ( "value out of range", args );
  }

  return x;
}


//-----------------------------------------------------------------------
//   isUsedArg
//-----------------------------------------------------------------------


bool CodeBlock::isUsedArg ( idx_t iarg ) const
{
  const idx_t    n   = opcodes.size ();
  const Opcode*  ops = opcodes.addr ();

  if ( iarg < 3 )
  {
    Opcode  target = (Opcode) (Opcodes::LOAD_ARG0 + iarg);

    for ( idx_t i = 0; i < n; i += Opcodes::size( ops[i] ) )
    {
      if ( ops[i] == target )
      {
        return true;
      }
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i += Opcodes::size( ops[i] ) )
    {
      if ( ops[i] == Opcodes::LOAD_ARG &&
           iarg   == (idx_t) ops[i + 1] )
      {
        return true;
      }
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   markUsedArgs
//-----------------------------------------------------------------------


idx_t CodeBlock::markUsedArgs ( bool* mask ) const
{
  const idx_t    n   = opcodes.size ();
  const Opcode*  ops = opcodes.addr ();

  Opcode  code;
  idx_t   i, k;


  i = 0;
  k = 0;

  while ( i < n )
  {
    code = ops[i];

    switch ( code )
    {
    case Opcodes::LOAD_ARG0:
    case Opcodes::LOAD_ARG1:
    case Opcodes::LOAD_ARG2:

      mask[code - Opcodes::LOAD_ARG0] = true;

      k++;

      break;

    case Opcodes::LOAD_ARG:

      mask[ops[i + 1]] = true;

      k++;

      break;

    default:

      ;
    }

    i += Opcodes::size ( code );
  }

  return k;
}


//-----------------------------------------------------------------------
//   resolve_
//-----------------------------------------------------------------------


void CodeBlock::resolve_ ()
{
  const idx_t  n   = opcodes.size ();
  Opcode*      ops = opcodes.addr ();

  Function*   func;
  Opcode      code;
  idx_t       argc;
  idx_t       idx;
  idx_t       addr;
  idx_t       i;


  i = 0;

  while ( i < n )
  {
    code = ops[i];

    switch ( code )
    {
    case Opcodes::JUMP:
    case Opcodes::JUMP_IF_FALSE:

      addr = i + ops[i + 1];

      if ( addr < 0 || addr > n )
      {
        throw Error (
          ERR_CONTEXT,
          String::format (
            "illegal jump address: %d; valid range is [0,%d)",
            addr, n
          )
        );
      }

      break;

    case Opcodes::LOAD_VAR:
    case Opcodes::TRAP_VAR:

      idx = ops[i + 1];

      if ( idx < 0 || idx >= vars.size() )
      {
        throw Error (
          ERR_CONTEXT,
          String::format (
            "illegal variable index: %d; valid range is [0,%d)",
            idx, vars.size()
          )
        );
      }

      if ( vars[idx].kind == UNDEFINED )
      {
        ops[i] = Opcodes::TRAP_VAR;
      }
      else
      {
        ops[i] = Opcodes::LOAD_VAR;
      }

      break;

    case Opcodes::STORE_VAR:

      idx = ops[i + 1];

      if ( idx < 0 || idx >= vars.size() )
      {
        throw Error (
          ERR_CONTEXT,
          String::format (
            "illegal variable index: %d; valid range is [0,%d)",
            idx, vars.size()
          )
        );
      }

      if ( vars[idx].kind != SAVED )
      {
        throw Error (
          ERR_CONTEXT,
          String::format (
            "variable `%s\' is not a saved variable",
            vars[idx].name
          )
        );
      }

      break;

    case Opcodes::CALL_FUNC:
    case Opcodes::TRAP_FUNC:

      idx  = ops[i + 1];

      if ( idx < 0 || idx >= funcs.size() )
      {
        throw Error (
          ERR_CONTEXT,
          String::format (
            "illegal function index: %d; valid range is [0,%d)",
            idx, funcs.size()
          )
        );
      }

      func = funcs[idx].addr.get ();

      if (  ! func )
      {
        ops[i] = Opcodes::TRAP_FUNC;
      }
      else
      {
        argc = (idx_t) ops[i + 2];

        if ( argc != func->argCount() )
        {
          ops[i] = Opcodes::TRAP_FUNC;

          resolveError_ (
            String::format (
              "wrong number of parameters in call of function `%s\' "
              "(should be %d)",
              funcs[idx].name,
              func->argCount ()
            )
          );
        }

        ops[i] = Opcodes::CALL_FUNC;
      }

      break;

    default:

      ;
    }

    i += Opcodes::size ( code );
  }
}


//-----------------------------------------------------------------------
//   evalPow_
//-----------------------------------------------------------------------


inline double CodeBlock::evalPow_

  ( double  x,
    double  y ) const

{
  if ( (y < 0.0 &&   isTiny( x )) ||
       (x < 0.0 && ! isTiny( y - ::floor( y + 0.5 ) )) )
  {
    funcError_ ( "pow", x, y );
  }

  return std::pow ( x, y );
}


//-----------------------------------------------------------------------
//   evalError_
//-----------------------------------------------------------------------


void CodeBlock::evalError_

  ( const String&  what,
    const double*  args ) const

{
  StringBuffer  buf;

  buf += "error evaluating expression `";
  buf += StringUtils::shorten ( expr, ERR_WIDTH );
  buf += '\'';

  if ( argCount && args )
  {
    buf += " with arguments (";
    buf += args[0];

    for ( idx_t i = 1; i < argCount; i++ )
    {
      buf += ", ";
      buf += args[i];
    }

    buf += ')';
  }

  if ( what.size() > 0 )
  {
    buf += ": ";
    buf += what;
  }

  throw FuncException ( ERR_CONTEXT, buf.toString() );
}


//-----------------------------------------------------------------------
//   funcError_
//-----------------------------------------------------------------------


void CodeBlock::funcError_

  ( const char*  name,
    double       arg ) const

{
  evalError_ (
    String::format ( "failed to compute `%s(%e)\'", name, arg )
  );
}


void CodeBlock::funcError_

  ( const char*  name,
    double       arg0,
    double       arg1 ) const

{
  evalError_ (
    String::format (
      "failed to compute `%s(%e,%e)\'", name, arg0, arg1
    )
  );
}


//-----------------------------------------------------------------------
//   trapError_
//-----------------------------------------------------------------------


void CodeBlock::trapError_ ( const Var& var ) const
{
  evalError_ ( "undefined variable: " + var.name );
}


void CodeBlock::trapError_ ( const Func& func ) const
{
  evalError_ ( "undefined function: " + func.name );
}


//-----------------------------------------------------------------------
//   opcodeError_
//-----------------------------------------------------------------------


void CodeBlock::opcodeError_ ( Opcode code ) const
{
  String  ex = StringUtils::shorten ( expr, ERR_WIDTH );

  throw Error (
    ERR_CONTEXT,
    String::format (
      "oops, illegal opcode (%d) while evaluating the "
      "expression `%s\'", (idx_t) code, & ex
    )
  );
}


//-----------------------------------------------------------------------
//   resolveError_
//-----------------------------------------------------------------------


void CodeBlock::resolveError_ ( const String& what ) const
{
  String  ex = StringUtils::shorten ( expr, ERR_WIDTH );

  throw FuncException (
    ERR_CONTEXT,
    String::format (
      "error resolving expression `%s\': %s", ex, what
    )
  );
}


void CodeBlock::resolveError_ ( const Var& var ) const
{
  resolveError_ ( "undefined variable: " + var.name );
}


void CodeBlock::resolveError_ ( const Func& func ) const
{
  resolveError_ ( "undefined function: " + func.name );
}


JEM_END_UFUNC_IMP
