
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


#include <cstring>
#include <jem/base/limits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalIndexException.h>
#include <jem/base/array/Array.h>
#include <jem/numeric/func/Function.h>


JEM_DEFINE_CUSTOM_CLASS( jem::numeric::Function );


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class Function
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


Function::~Function ()
{}


//-----------------------------------------------------------------------
//   makeClass
//-----------------------------------------------------------------------


Class* Function::makeClass ()
{
  ClassTemplate<Self>*  myClass = new ClassTemplate<Self> ();

  myClass->setWhatis ( "a function" );

  return myClass;
}


//-----------------------------------------------------------------------
//   getDeriv
//-----------------------------------------------------------------------


double Function::getDeriv

  ( idx_t          iarg,
    const double*  args ) const

{
  const idx_t  argc = argCount ();

  double       f1, f2;
  double       x1, x2, dx;


  if ( iarg < 0 || iarg >= argc )
  {
    throw IllegalIndexException ( JEM_FUNC, iarg, argc );
  }

  dx = 1.0e-6 * args[iarg];

  if ( isTiny( dx ) )
  {
    return 0.0;
  }

  x1 = args[iarg] - dx;
  x2 = args[iarg] + dx;
  dx = x2 - x1;

  if ( argc <= 16 )
  {
    double v[16];

    std::memcpy ( v, args, (size_t) argc * sizeof(double) );

    v[iarg] = x1;
    f1      = getValue ( v );
    v[iarg] = x2;
    f2      = getValue ( v );
  }
  else
  {
    Array<double>  v ( argc );

    std::memcpy ( v.addr(), args, (size_t) argc * sizeof(double) );

    v[iarg] = x1;
    f1      = getValue ( v.addr() );
    v[iarg] = x2;
    f2      = getValue ( v.addr() );
  }

  return ((f2 - f1) / dx);
}


//-----------------------------------------------------------------------
//   markUsedArgs
//-----------------------------------------------------------------------


idx_t Function::markUsedArgs ( bool* mask ) const
{
  const idx_t  argc = argCount ();

  for ( idx_t i = 0; i < argc; i++ )
  {
    mask[i] = true;
  }

  return argc;
}


JEM_END_PACKAGE( numeric )
