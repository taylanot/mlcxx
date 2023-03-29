
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
#include <jem/gl/byte.h>
#include <jem/gl/Array.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   convertArrays
//-----------------------------------------------------------------------


template <class T, class U>

  inline void         convertArrays

  ( const Array<T>&     lhs,
    const Array<U>&     rhs )

{
  JEM_PRECHECK ( lhs.size() == rhs.size() );

  const idx_t  n = lhs.size ();

  if ( lhs.isContiguous() && rhs.isContiguous() )
  {
    const U*  rp = rhs.addr ();
    T*        lp = lhs.addr ();

    for ( idx_t i = 0; i < n; i++ )
    {
      convert ( lp[i], rp[i] );
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      convert ( lhs[i], rhs[i] );
    }
  }
}


template <class T, class U>

  inline void         convertArrays

  ( const Array<T,2>&   lhs,
    const Array<U,2>&   rhs )

{
  JEM_PRECHECK ( lhs.size(0) == rhs.size(0) &&
                 lhs.size(1) == rhs.size(1) );

  const idx_t  m = lhs.size (0);
  const idx_t  n = rhs.size (1);

  if ( lhs.isContiguous() && rhs.isContiguous() )
  {
    const idx_t  sz = m * n;
    const U*     rp = rhs.addr ();
    T*           lp = lhs.addr ();

    for ( idx_t i = 0; i < sz; i++ )
    {
      convert ( lp[i], rp[i] );
    }
  }
  else
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      for ( idx_t i = 0; i < m; i++ )
      {
        convert ( lhs(i,j), rhs(i,j) );
      }
    }
  }
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   convert
//-----------------------------------------------------------------------


void                      convert

  ( const GLbyteVector&     lhs,
    const GLfloatVector&    rhs )

{
  convertArrays ( lhs, rhs );
}


void                      convert

  ( const GLbyteMatrix&     lhs,
    const GLfloatMatrix&    rhs )

{
  convertArrays ( lhs, rhs );
}


void                      convert

  ( const GLbyteVector&     lhs,
    const GLdoubleVector&   rhs )

{
  convertArrays ( lhs, rhs );
}


void                      convert

  ( const GLbyteMatrix&     lhs,
    const GLdoubleMatrix&   rhs )

{
  convertArrays ( lhs, rhs );
}


void                      convert

  ( const GLubyteVector&    lhs,
    const GLfloatVector&    rhs )

{
  convertArrays ( lhs, rhs );
}


void                      convert

  ( const GLubyteMatrix&    lhs,
    const GLfloatMatrix&    rhs )

{
  convertArrays ( lhs, rhs );
}


void                      convert

  ( const GLubyteVector&    lhs,
    const GLdoubleVector&   rhs )

{
  convertArrays ( lhs, rhs );
}


void                      convert

  ( const GLubyteMatrix&    lhs,
    const GLdoubleMatrix&   rhs )

{
  convertArrays ( lhs, rhs );
}


JEM_END_PACKAGE( gl )
