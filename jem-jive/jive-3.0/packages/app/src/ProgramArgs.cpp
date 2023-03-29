
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


#include <cstring>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/RangeException.h>
#include <jive/app/Globdat.h>
#include <jive/app/ProgramArgs.h>


JEM_DEFINE_CLASS( jive::app::ProgramArgs );


JIVE_BEGIN_PACKAGE( app )


using jem::newInstance;


//=======================================================================
//   class ProgramArgs
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


ProgramArgs::ProgramArgs ( const String& arg0 )
{
  try
  {
    alloc_  ( 1 );
    append_ ( arg0 );
  }
  catch ( ... )
  {
    free_ ();
    throw;
  }
}


ProgramArgs::ProgramArgs

  ( const String&  arg0,
    const String&  arg1 )

{
  try
  {
    alloc_  ( 2 );
    append_ ( arg0 );
    append_ ( arg1 );
  }
  catch ( ... )
  {
    free_ ();
    throw;
  }
}


ProgramArgs::ProgramArgs

  ( const String&  arg0,
    const String&  arg1,
    const String&  arg2 )

{
  try
  {
    alloc_  ( 3 );
    append_ ( arg0 );
    append_ ( arg1 );
    append_ ( arg2 );
  }
  catch ( ... )
  {
    free_ ();
    throw;
  }
}


ProgramArgs::ProgramArgs ( const StringVector& args )
{

  if ( args.size() > jem::maxOf<int>() )
  {
    throw jem::RangeException (
      JEM_FUNC,
      String::format (
        "too many program arguments: %d (maximum is %d)",
        args.size(), jem::maxOf<int>()
      )
    );
  }

  const int  n = (int) args.size ();

  try
  {
    if ( n == 0 )
    {
      alloc_  ( 1 );
      append_ ( "" );
    }
    else
    {
      alloc_ ( n );

      for ( idx_t i = 0; i < n; i++ )
      {
        append_ ( args[i] );
      }
    }
  }
  catch ( ... )
  {
    free_ ();
    throw;
  }
}


ProgramArgs::ProgramArgs

  ( int     argc,
    char**  argv )

{
  try
  {
    if ( argc == 0 )
    {
      alloc_  ( 1 );
      append_ ( "" );
    }
    else
    {
      alloc_ ( argc );

      for ( int i = 0; i < argc; i++ )
      {
        append_ ( argv[i] );
      }
    }
  }
  catch ( ... )
  {
    free_ ();
    throw;
  }
}


ProgramArgs::ProgramArgs ( const Self& rhs )
{
  try
  {
    alloc_ ( rhs.oldArgc_ );

    for ( int i = 0; i < rhs.oldArgc_; i++ )
    {
      append_ ( rhs.oldArgv_[i] );
    }
  }
  catch ( ... )
  {
    free_ ();
    throw;
  }
}


ProgramArgs::~ProgramArgs ()
{
  free_ ();
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> ProgramArgs::clone () const
{
  return  newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   dup
//-----------------------------------------------------------------------


Ref<ProgramArgs> ProgramArgs::dup () const
{
  return  newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   argc
//-----------------------------------------------------------------------


int& ProgramArgs::argc ()
{
  newArgc_ = oldArgc_;

  return newArgc_;
}


//-----------------------------------------------------------------------
//   argv
//-----------------------------------------------------------------------


char** ProgramArgs::argv ()
{
  for ( int i = 0; i < oldArgc_; i++ )
  {
    newArgv_[i] = oldArgv_[i];
  }

  return newArgv_;
}


//-----------------------------------------------------------------------
//   commit
//-----------------------------------------------------------------------


void ProgramArgs::commit ()
{
  if ( newArgc_ < 1 )
  {
    newArgc_ = 1;
  }

  if ( newArgc_ > oldArgc_ )
  {
    newArgc_ = oldArgc_;
  }

  for ( int i = 1; i < newArgc_; i++ )
  {
    oldArgv_[i] = newArgv_[i];
  }

  for ( int i = newArgc_; i < oldArgc_; i++ )
  {
    oldArgv_[i] = 0;
  }

  oldArgc_ = newArgc_;
}


//-----------------------------------------------------------------------
//   toArray
//-----------------------------------------------------------------------


StringVector ProgramArgs::toArray () const
{
  StringVector  args ( oldArgc_ );

  for ( int i = 0; i < oldArgc_; i++ )
  {
    args[i] = jem::clone ( oldArgv_[i] );
  }

  return args;
}


//-----------------------------------------------------------------------
//   getProgramName
//-----------------------------------------------------------------------


String ProgramArgs::getProgramName () const
{
  return jem::clone ( orgArgv_[0] );
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void ProgramArgs::store ( const Properties& globdat ) const
{
  Ref<Object> self = const_cast<Self*> ( this );

  globdat.set ( Globdat::PROGRAM_ARGS, self );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<ProgramArgs> ProgramArgs::get ( const Properties& globdat )
{
  Ref<Self>  args;

  if ( ! globdat.find( args, Globdat::PROGRAM_ARGS ) )
  {
    args = newInstance<Self> ();
  }

  return args;
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<ProgramArgs> ProgramArgs::find ( const Properties& globdat )
{
  Ref<Self>  args;

  globdat.find ( args, Globdat::PROGRAM_ARGS );

  return args;
}


//-----------------------------------------------------------------------
//   alloc_
//-----------------------------------------------------------------------


void ProgramArgs::alloc_ ( int argc )
{
  // Make sure that the method free_ can be safely called when an
  // exception is thrown in this method.

  newArgc_ = oldArgc_ = orgArgc_ = 0;
  newArgv_ = oldArgv_ = orgArgv_ = 0;

  const idx_t  n = 3 * argc + 2;
  char**       p = new char*[n];

  for ( idx_t i = 0; i < n; i++ )
  {
    p[i] = 0;
  }

  newArgv_ = p + argc;
  oldArgv_ = p + (argc + argc + 1);
  orgArgv_ = p;
}


//-----------------------------------------------------------------------
//   free_
//-----------------------------------------------------------------------


void ProgramArgs::free_ ()
{
  if ( orgArgv_ )
  {
    for ( int i = orgArgc_ - 1; i >= 0; i-- )
    {
      if ( orgArgv_[i] )
      {
        delete [] orgArgv_[i];
      }
    }

    delete [] orgArgv_;

    newArgv_ = oldArgv_ = orgArgv_ = 0;
  }
}


//-----------------------------------------------------------------------
//   append_
//-----------------------------------------------------------------------


void ProgramArgs::append_ ( const String& arg )
{
  const idx_t  n = arg.size ();
  char*        p = new char[n + 1];

  std::memcpy ( p, arg.addr(), (size_t) n );

  p[n] = '\0';

  newArgv_[newArgc_++] = p;
  oldArgv_[oldArgc_++] = p;
  orgArgv_[orgArgc_++] = p;
}


JIVE_END_PACKAGE( app )
