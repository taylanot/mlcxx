
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


#include <jem/base/assert.h>
#include <jem/base/utilities.h>
#include <jem/base/Clonable.h>
#include <jem/base/thread/ThreadLocal.h>


JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class ThreadLocal
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ThreadLocal::ThreadLocal ()
{
  for ( int i = 0; i < ROW_COUNT; i++ )
  {
    data_[i] = 0;
  }
}


ThreadLocal::~ThreadLocal ()
{
  int  i, j, k, n;

  k = 0;

  do
  {
    n = 0;

    for ( i = 0; i < ROW_COUNT; i++ )
    {
      if ( data_[i] )
      {
        for ( j = 0; j < COL_COUNT; j++ )
        {
          if ( data_[i][j] != nullptr )
          {
            n++;

            try
            {
              data_[i][j] = nullptr;
            }
            catch ( ... ) {}
          }
        }
      }
    }

    if ( locale_ )
    {
      n++;

      try
      {
        locale_ = nullptr;
      }
      catch ( ... ) {}
    }

    if ( sysdata_ )
    {
      n++;

      try
      {
        sysdata_ = nullptr;
      }
      catch ( ... ) {}
    }

    k++;
  }
  while ( (k < MAX_ITER) && (n > 0) );

  if ( n > 0 )
  {
    jem::abort ( JEM_FUNC,
                 "failed to destroy all thread local data.\n"
                 "Possibly, there is a cyclic data dependency\n"
                 "in the program." );
  }

  for ( i = 0; i < ROW_COUNT; i++ )
  {
    delete [] data_[i];
    data_[i] = nullptr;
  }
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<ThreadLocal> ThreadLocal::clone () const
{
  Ref<ThreadLocal>  tl = newInstance<ThreadLocal> ();
  Clonable*         c  = nullptr;

  for ( int k = 0; k < MAX_KEYS; k++ )
  {
    c = dynamicCast<Clonable*> ( get( k ) );

    if ( c )
    {
      tl->set ( k, c->clone() );
    }
  }

  c = dynamicCast<Clonable*> ( locale_ );

  if ( c )
  {
    tl->locale_ = c->clone ();
  }

  c = dynamicCast<Clonable*> ( sysdata_ );

  if ( c )
  {
    tl->sysdata_ = c->clone ();
  }

  return tl;
}


//-----------------------------------------------------------------------
//   set
//-----------------------------------------------------------------------


void ThreadLocal::set ( int key, const Ref<Object>& obj )
{
  JEM_ASSERT_NOTHROW ( key >= 0 && key < MAX_KEYS );

  int  i = key >> SHIFT_BITS;
  int  j = key  & BIT_MASK;


  if ( data_[i] == nullptr && obj == nullptr )
  {
    return;
  }

  if ( data_[i] == nullptr )
  {
    data_[i] = new Ref<Object>[COL_COUNT];
  }

  data_[i][j] = obj;
}


//-----------------------------------------------------------------------
//   setLocale
//-----------------------------------------------------------------------


void ThreadLocal::setLocale ( const Ref<Object>& obj )
{
  locale_ = obj;
}


//-----------------------------------------------------------------------
//   setSysdata
//-----------------------------------------------------------------------


void ThreadLocal::setSysdata ( const Ref<Object>& obj )
{
  sysdata_ = obj;
}


JEM_END_PACKAGE_BASE
