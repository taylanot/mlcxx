
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

#ifndef JEM_NUMERIC_ALGEBRA_MATMULCHAIN_H
#define JEM_NUMERIC_ALGEBRA_MATMULCHAIN_H

#include <jem/base/array/Array.h>


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   class MatmulChain (primary template)
//-----------------------------------------------------------------------


template <class T, int N> class MatmulChain {};


//-----------------------------------------------------------------------
//   class MatmulChain<T,0>
//-----------------------------------------------------------------------


template <class T>

  class MatmulChain<T,0>

{
 public:

  static const int          LENGTH = 0;


  inline                    MatmulChain () noexcept;

  inline const Array<T,2>&  matmul

    ( const Array<T>&         v,
      const Array<T>&         w );


 private:

  MatmulChain&              operator = ( const MatmulChain& );


 private:

  Array<T,2>                mout_;

};


//-----------------------------------------------------------------------
//   class MatmulChain<T,1>
//-----------------------------------------------------------------------


template <class T>

  class MatmulChain<T,1>

{
 public:

  static const int          LENGTH = 1;


  inline                    MatmulChain () noexcept;

  const Array<T>&           matmul

    ( const Array<T,2>&       a,
      const Array<T>&         v );

  inline const Array<T>&    matmul

    ( const Array<T>&         v,
      const Array<T,2>&       a );

  inline const Array<T,2>&  matmul

    ( const Array<T>&         v,
      const Array<T,2>&       a,
      const Array<T>&         w );


 private:

  MatmulChain&              operator = ( const MatmulChain& );

  inline void               vresize_

    ( idx_t                   m,
      idx_t                   n );


 private:

  Array<T,2>                mout_;
  Array<T>                  vout_;
  Array<T>                  args_;

};


//-----------------------------------------------------------------------
//   class MatmulChain<T,2>
//-----------------------------------------------------------------------


template <class T>

  class MatmulChain<T,2>

{
 public:

  static const int          LENGTH = 2;


  inline                    MatmulChain () noexcept;

  const Array<T,2>&         matmul

    ( const Array<T,2>&       a,
      const Array<T,2>&       b );

  const Array<T>&           matmul

    ( const Array<T,2>&       a,
      const Array<T,2>&       b,
      const Array<T>&         v );

  inline const Array<T>&    matmul

    ( const Array<T>&         v,
      const Array<T,2>&       a,
      const Array<T,2>&       b );

  inline const Array<T,2>&  matmul

    ( const Array<T>&         v,
      const Array<T,2>&       a,
      const Array<T,2>&       b,
      const Array<T>&         w );


 private:

  MatmulChain&              operator = ( const MatmulChain& );

  inline void               mresize_

    ( idx_t                   m,
      idx_t                   n,
      idx_t                   p );

  inline void               vresize_

    ( idx_t                   m,
      idx_t                   p );


 private:

  Array<T,2>                mout_;
  Array<T>                  vout_;
  Array<T>                  args_;

};



//-----------------------------------------------------------------------
//   class MatmulChain<T,3>
//-----------------------------------------------------------------------


template <class T>

  class MatmulChain<T,3>

{
 public:

  static const int          LENGTH = 3;


  inline                    MatmulChain () noexcept;

  const Array<T,2>&         matmul

    ( const Array<T,2>&       a,
      const Array<T,2>&       b,
      const Array<T,2>&       c );

  const Array<T>&           matmul

    ( const Array<T,2>&       a,
      const Array<T,2>&       b,
      const Array<T,2>&       c,
      const Array<T>&         v );

  inline const Array<T>&    matmul

    ( const Array<T>&         v,
      const Array<T,2>&       a,
      const Array<T,2>&       b,
      const Array<T,2>&       c );

  inline const Array<T,2>&  matmul

    ( const Array<T>&         v,
      const Array<T,2>&       a,
      const Array<T,2>&       b,
      const Array<T,2>&       c,
      const Array<T>&         w );


 private:

  MatmulChain&              operator = ( const MatmulChain& );

  inline void               mresize_

    ( idx_t                   m,
      idx_t                   n,
      idx_t                   p,
      idx_t                   q );

  inline void               vresize_

    ( idx_t                   m,
      idx_t                   n,
      idx_t                   q );


 private:

  Array<T,2>                mout_;
  Array<T>                  vout_;
  Array<T>                  args_;
  Array<T>                  tmp_;

};


//-----------------------------------------------------------------------
//   class MatmulChain<T,4>
//-----------------------------------------------------------------------


template <class T>

  class MatmulChain<T,4>

{
 public:

  static const int          LENGTH = 4;


  inline                    MatmulChain () noexcept;

  const Array<T,2>&         matmul

    ( const Array<T,2>&       a,
      const Array<T,2>&       b,
      const Array<T,2>&       c,
      const Array<T,2>&       d );

  const Array<T>&           matmul

    ( const Array<T,2>&       a,
      const Array<T,2>&       b,
      const Array<T,2>&       c,
      const Array<T,2>&       d,
      const Array<T>&         v );

  inline const Array<T>&    matmul

    ( const Array<T>&         v,
      const Array<T,2>&       a,
      const Array<T,2>&       b,
      const Array<T,2>&       c,
      const Array<T,2>&       d );

  inline const Array<T,2>&  matmul

    ( const Array<T>&         v,
      const Array<T,2>&       a,
      const Array<T,2>&       b,
      const Array<T,2>&       c,
      const Array<T,2>&       d,
      const Array<T>&         w );


 private:

  MatmulChain&              operator = ( const MatmulChain& );

  inline void               mresize_

    ( idx_t                   m,
      idx_t                   n,
      idx_t                   p,
      idx_t                   q,
      idx_t                   r );

  inline void               vresize_

    ( idx_t                   m,
      idx_t                   p,
      idx_t                   r );


 private:

  Array<T,2>                mout_;
  Array<T>                  vout_;
  Array<T>                  args_;
  Array<T>                  tmp_;

};


//-----------------------------------------------------------------------
//   disabled MatmulChain classes for type bool
//-----------------------------------------------------------------------


template <> class MatmulChain<bool,0> {};
template <> class MatmulChain<bool,1> {};
template <> class MatmulChain<bool,2> {};
template <> class MatmulChain<bool,3> {};



//#######################################################################
//   Implementation
//#######################################################################


JEM_END_PACKAGE( numeric )

#include <jem/numeric/algebra/MatmulChain.tcc>

#endif
