
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


#include <cctype>
#include <jem/base/rtti.h>
#include <jem/base/Float.h>
#include <jem/base/AutoPointer.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/ParseException.h>
#include "ufunc/Opcodes.h"
#include "ufunc/Compiler.h"
#include "ufunc/CodeBlock.h"
#include "ufunc/StdArgList.h"
#include "ufunc/StdNameSpace.h"
#include "ufunc/EmptyNameSpace.h"


JEM_DEFINE_CUSTOM_CLASS( jem::numeric::UserFunc );


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class UserFunc::ArgList
//=======================================================================


UserFunc::ArgList::~ArgList ()
{}


//=======================================================================
//   class UserFunc::NameSpace
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


UserFunc::NameSpace::~NameSpace ()
{}


//-----------------------------------------------------------------------
//   toFloat
//-----------------------------------------------------------------------


double UserFunc::NameSpace::toFloat

  ( const Ref<Object>&  obj,
    const String&       name )

{
  double  x;

  if ( toValue( x, obj ) )
  {
    return x;
  }

  bool  b;

  if ( toValue( b, obj ) )
  {
    return b ? 1.0 : 0.0;
  }

  String  s = obj.toString ();

  if ( Float::parse( x, s ) )
  {
    return x;
  }

  throw IllegalInputException (
    JEM_FUNC,
    String::format ( "symbol `%s\' is not a number", name )
  );

  return 0.0;
}


//=======================================================================
//   class UserFunc
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


UserFunc::UserFunc ()
{
  using ufunc::Opcodes;
  using ufunc::CodeBlock;

  AutoPointer<CodeBlock>  code = new CodeBlock ( "0.0" );

  code->argCount = 0;
  code->maxSSize = 1;

  idx_t  index = code->addLiteral ( 0.0 );

  code->opcodes.pushBack ( Opcodes::LOAD_VAR );
  code->opcodes.pushBack ( (ufunc::Opcode) index );

  code->trimToSize ();

  code_ = code.release ();
}


UserFunc::UserFunc ( ufunc::CodeBlock* code )
{
  JEM_PRECHECK2 ( code, "NULL CodeBlock pointer" );

  code_ = code;
}


UserFunc::UserFunc

  ( const String&  args,
    const String&  expr )

{
  ufunc::StdArgList  argList ( args );
  ufunc::Compiler    compiler;

  code_ = compiler.parse ( argList, expr );
}


UserFunc::~UserFunc ()
{
  delete code_;
  code_ = 0;
}


//-----------------------------------------------------------------------
//   makeClass
//-----------------------------------------------------------------------


Class* UserFunc::makeClass ()
{
  ClassTemplate<Self>*  myClass = new ClassTemplate<Self> ();

  myClass->setWhatis      ( "a user-defined function" );
  myClass->addDefaultCtor ();

  return myClass;
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String UserFunc::toString () const
{
  return code_->expr;
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void UserFunc::readFrom ( ObjectInput& in )
{
  code_->readFrom ( in );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void UserFunc::writeTo ( ObjectOutput& out ) const
{
  code_->writeTo ( out );
}


//-----------------------------------------------------------------------
//   argCount
//-----------------------------------------------------------------------


idx_t UserFunc::argCount () const
{
  return code_->argCount;
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


double UserFunc::getValue ( const double* args ) const
{
  const idx_t  bufSize = code_->tmpCount + code_->maxSSize;

  if ( bufSize <= 32 )
  {
    double  vbuf[32];

    return code_->calcValue ( vbuf, args );
  }
  else
  {
    util::Flex<double>  vbuf ( bufSize );

    return code_->calcValue ( vbuf.addr(), args );
  }
}


//-----------------------------------------------------------------------
//   getDeriv
//-----------------------------------------------------------------------


double UserFunc::getDeriv

  ( idx_t          iarg,
    const double*  args ) const

{
  if ( ! code_->isUsedArg( iarg ) )
  {
    return 0.0;
  }

  const idx_t  bufSize = 2 * (code_->tmpCount + code_->maxSSize);

  if ( bufSize <= 32 )
  {
    double  vbuf[32];

    return code_->calcDeriv ( vbuf, iarg, args );
  }
  else
  {
    util::Flex<double>  vbuf ( bufSize );

    return code_->calcDeriv ( vbuf.addr(), iarg, args );
  }
}


//-----------------------------------------------------------------------
//   markUsedArgs
//-----------------------------------------------------------------------


idx_t UserFunc::markUsedArgs ( bool* mask ) const
{
  const idx_t  argc = code_->argCount;

  for ( idx_t i = 0; i < argc; i++ )
  {
    mask[i] = false;
  }

  return code_->markUsedArgs ( mask );
}


//-----------------------------------------------------------------------
//   resolve
//-----------------------------------------------------------------------


void UserFunc::resolve ( ResolveOptions options )
{
  ufunc::EmptyNameSpace  ns;

  this->resolve ( ns, options );
}


void UserFunc::resolve

  ( const Dictionary&  dict,
    ResolveOptions     options )

{
  ufunc::StdNameSpace  ns ( dict );

  this->resolve ( ns, options );
}


void UserFunc::resolve

  ( const NameSpace&  ns,
    ResolveOptions    options )

{
  using ufunc::CodeBlock;

  if ( code_->status == CodeBlock::RESOLVED )
  {
    return;
  }

  if ( code_->extCount > 0 )
  {
    code_->resolve ( ns, options );
  }
  else
  {
    code_->status = CodeBlock::RESOLVED;
  }
}


//-----------------------------------------------------------------------
//   unresolve
//-----------------------------------------------------------------------


void UserFunc::unresolve ()
{
  using ufunc::CodeBlock;

  if ( code_->status == CodeBlock::RESOLVED && code_->extCount > 0 )
  {
    code_->unresolve ();
  }
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


Ref<UserFunc> UserFunc::parse

  ( ArgList&       args,
    const String&  expr )

{
  ufunc::Compiler  compiler;

  return newInstance<Self> ( compiler.parse ( args, expr ) );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   parseUserFunc
//-----------------------------------------------------------------------


void              parseUserFunc

  ( String&         name,
    String&         args,
    String&         expr,
    const String&   input )

{
  const char*  s = input.addr ();
  const idx_t  n = input.size ();

  idx_t        i, j, k;

  for ( i = 0; i < n && std::isspace(s[i]); i++ ) ;

  j = i;

  while ( j < n && ! (std::isspace(s[j]) || s[j] == '(') )
  {
    j++;
  }

  if ( i == j )
  {
    throw ParseException ( input, "no function name specified" );
  }

  name = input[slice(i,j)];

  for ( i = j; i < n && std::isspace(s[i]); i++ ) ;

  if ( i == n || s[i] != '(' )
  {
    throw ParseException ( input,
                           "missing `(\' after function name" );
  }

  for ( i++;   i < n && std::isspace(s[i]); i++ ) ;

  for ( j = i; j < n && s[j] != ')';   j++ ) ;

  if ( j == n )
  {
    throw ParseException ( input,
                           "missing `)\' after argument list" );
  }

  for ( k = j; k > i && std::isspace(s[k - 1]); k-- ) ;

  args = input[slice(i,k)];

  for ( i = j + 1; i < n && std::isspace(s[i]); i++ ) ;

  if ( i == n || s[i] != '=' )
  {
    throw ParseException ( input,
                           "missing `=\' after argument list" );
  }

  for ( i++;   i < n && std::isspace(s[i]);     i++ ) ;

  for ( j = n; j > i && std::isspace(s[j - 1]); j-- ) ;

  expr = input[slice(i,j)];
}


JEM_END_PACKAGE( numeric )
