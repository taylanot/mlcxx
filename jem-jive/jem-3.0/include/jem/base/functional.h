
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

#ifndef JEM_BASE_FUNCTIONAL_H
#define JEM_BASE_FUNCTIONAL_H

#include <jem/base/promote.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class UnaryFunc
//-----------------------------------------------------------------------


template <class T, class R = typename PromoteType<T>::ResultType>

  class UnaryFunc

{
 public:

  typedef T   argument_type;
  typedef R   result_type;

  typedef T   ArgType;
  typedef R   ResultType;

};


//-----------------------------------------------------------------------
//   class LogicalNot
//-----------------------------------------------------------------------


template <class T>

  class LogicalNot : public UnaryFunc<T,bool>

{
 public:

  bool operator () ( const T& x ) const
  {
    return (! x);
  }

};


//-----------------------------------------------------------------------
//   class BitwiseNot
//-----------------------------------------------------------------------


template <class T>

  class BitwiseNot : public UnaryFunc<T>

{
 public:

  typename UnaryFunc<T>::ResultType

    operator () ( const T& x ) const

  {
    return ( ~x );
  }

};


//-----------------------------------------------------------------------
//   class Negate
//-----------------------------------------------------------------------


template <class T>

  class Negate : public UnaryFunc<T>

{
 public:

  typename UnaryFunc<T>::ResultType

    operator () ( const T& arg ) const

  {
    return -arg;
  }

};


//-----------------------------------------------------------------------
//   class StaticCast
//-----------------------------------------------------------------------


template <class U, class T>

  class StaticCast : public UnaryFunc<T,U>

{
 public:

  U operator () ( const T& arg ) const
  {
    return U ( arg );
  }

};


//-----------------------------------------------------------------------
//   class BinaryFunc
//-----------------------------------------------------------------------


template <class T, class U,
          class R = typename PromoteTypes<T,U>::ResultType>

  class BinaryFunc

{
 public:

  typedef T   first_argument_type;
  typedef U   second_argument_type;
  typedef R   result_type;

  typedef T   FirstArgType;
  typedef U   SecondArgType;
  typedef R   ResultType;

};


//-----------------------------------------------------------------------
//   class Multiplies
//-----------------------------------------------------------------------


template <class T, class U = T>

  class Multiplies : public BinaryFunc<T,U>

{
 public:

  typename BinaryFunc<T,U>::ResultType

    operator () ( const T& x, const U& y ) const

  {
    return (x * y);
  }

};


//-----------------------------------------------------------------------
//   class Divides
//-----------------------------------------------------------------------


template <class T, class U = T>

  class Divides : public BinaryFunc<T,U>

{
 public:

  typename BinaryFunc<T,U>::ResultType

    operator () ( const T& x, const U& y ) const

  {
    return (x / y);
  }

};


//-----------------------------------------------------------------------
//   class Modulus
//-----------------------------------------------------------------------


template <class T, class U = T>

  class Modulus : public BinaryFunc<T,U>

{
 public:

  typename BinaryFunc<T,U>::ResultType

    operator () ( const T& x, const U& y ) const

  {
    return (x % y);
  }

};


//-----------------------------------------------------------------------
//   class Plus
//-----------------------------------------------------------------------


template <class T, class U = T>

  class Plus : public BinaryFunc<T,U>

{
 public:

  typename BinaryFunc<T,U>::ResultType

    operator () ( const T& x, const U& y ) const

  {
    return (x + y);
  }

};


//-----------------------------------------------------------------------
//   class Minus
//-----------------------------------------------------------------------


template <class T, class U = T>

  class Minus : public BinaryFunc<T,U>

{
 public:

  typename BinaryFunc<T,U>::ResultType

    operator () ( const T& x, const U& y ) const

  {
    return (x - y);
  }

};


//-----------------------------------------------------------------------
//   class Maximum
//-----------------------------------------------------------------------


template <class T>

  class Maximum : public BinaryFunc<T,T,T>

{
 public:

  const T& operator () ( const T& x, const T& y ) const
  {
    if ( x > y )
    {
      return x;
    }
    else
    {
      return y;
    }
  }

};


//-----------------------------------------------------------------------
//   class Minimum
//-----------------------------------------------------------------------


template <class T>

  class Minimum : public BinaryFunc<T,T,T>

{
 public:

  const T& operator () ( const T& x, const T& y ) const
  {
    if ( x < y )
    {
      return x;
    }
    else
    {
      return y;
    }
  }

};


//-----------------------------------------------------------------------
//   class LessThan
//-----------------------------------------------------------------------


template <class T, class U = T>

  class LessThan : public BinaryFunc<T,U,bool>

{
 public:

  bool operator () ( const T& x, const U& y ) const
  {
    return (x < y);
  }

};


//-----------------------------------------------------------------------
//   class LessEqual
//-----------------------------------------------------------------------


template <class T, class U = T>

  class LessEqual : public BinaryFunc<T,U,bool>

{
 public:

  bool operator () ( const T& x, const U& y ) const
  {
    return (x <= y);
  }

};


//-----------------------------------------------------------------------
//   class GreaterThan
//-----------------------------------------------------------------------


template <class T, class U = T>

  class GreaterThan : public BinaryFunc<T,U,bool>

{
 public:

  bool operator () ( const T& x, const U& y ) const
  {
    return (x > y);
  }

};


//-----------------------------------------------------------------------
//   class GreaterEqual
//-----------------------------------------------------------------------


template <class T, class U = T>

  class GreaterEqual : public BinaryFunc<T,U,bool>

{
 public:

  bool  operator () ( const T& x, const U& y ) const
  {
    return (x >= y);
  }

};


//-----------------------------------------------------------------------
//   class EqualTo
//-----------------------------------------------------------------------


template <class T, class U = T>

  class EqualTo : public BinaryFunc<T,U,bool>

{
 public:

  bool operator () ( const T& x, const U& y ) const
  {
    return (x == y);
  }

};


//-----------------------------------------------------------------------
//   class NotEqual
//-----------------------------------------------------------------------


template <class T, class U = T>

  class NotEqual : public BinaryFunc<T,U,bool>

{
 public:

  bool operator () ( const T& x, const U& y ) const
  {
    return (x != y);
  }

};


//-----------------------------------------------------------------------
//   class LogicalAnd
//-----------------------------------------------------------------------


template <class T, class U = T>

  class LogicalAnd : public BinaryFunc<T,U,bool>

{
 public:

  bool operator () ( const T& x, const U& y ) const

  {
    return (x && y);
  }

};


//-----------------------------------------------------------------------
//   class LogicalOr
//-----------------------------------------------------------------------


template <class T, class U = T>

  class LogicalOr : public BinaryFunc<T,U,bool>

{
 public:

  bool operator () ( const T& x, const U& y ) const
  {
    return (x || y);
  }

};



//-----------------------------------------------------------------------
//   class BitwiseAnd
//-----------------------------------------------------------------------


template <class T, class U = T>

  class BitwiseAnd : public BinaryFunc<T,U>

{
 public:

  typename BinaryFunc<T,U>::ResultType

    operator () ( const T& x, const U& y ) const

  {
    return (x & y);
  }

};


//-----------------------------------------------------------------------
//   class BitwiseOr
//-----------------------------------------------------------------------


template <class T, class U = T>

  class BitwiseOr : public BinaryFunc<T,U>

{
 public:

  typename BinaryFunc<T,U>::ResultType

    operator () ( const T& x, const U& y ) const

  {
    return (x | y);
  }

};


//-----------------------------------------------------------------------
//   class BitwiseXor
//-----------------------------------------------------------------------


template <class T, class U = T>

  class BitwiseXor : public BinaryFunc<T,U>

{
 public:

  typename BinaryFunc<T,U>::ResultType

    operator () ( const T& x, const U& y ) const

  {
    return (x ^ y);
  }

};


//-----------------------------------------------------------------------
//   class BitwiseLeftShift
//-----------------------------------------------------------------------


template <class T, class U = T>

  class BitwiseLeftShift : public BinaryFunc<T,U,T>

{
 public:

  T operator () ( const T& x, const U& y ) const
  {
    return (x << y);
  }

};


//-----------------------------------------------------------------------
//   class BitwiseRightShift
//-----------------------------------------------------------------------


template <class T, class U = T>

  class BitwiseRightShift : public BinaryFunc<T,U,T>

{
 public:

  T operator () ( const T& x, const U& y ) const
  {
    return (x >> y);
  }

};


//-----------------------------------------------------------------------
//   class Reduction
//-----------------------------------------------------------------------


template <class T, class R>

  class Reduction

{
 public:

  typedef T   ArgType;
  typedef R   ResultType;

};


//-----------------------------------------------------------------------
//   class BinaryReduction
//-----------------------------------------------------------------------


template <class Op>

  class BinaryReduction

{
 public:

  typedef typename Op::FirstArgType  ArgType;
  typedef typename Op::ResultType    ResultType;


  explicit BinaryReduction ( const Op& op = Op() ) :

    op_     ( op ),
    result_ ( ResultType() )

  {}


  BinaryReduction ( const BinaryReduction& rhs ) :

    op_     ( rhs.op_     ),
    result_ ( rhs.result_ )

  {}


  void init ( const ArgType& a )
  {
    result_ = a;
  }


  void feed ( const ArgType& a )
  {
    result_ = op_ ( result_, a );
  }


  ResultType result () const
  {
    return result_;
  }


 private:

  Op          op_;
  ResultType  result_;

};


//-----------------------------------------------------------------------
//   class KahanSum
//-----------------------------------------------------------------------


template <class T>

  class KahanSum : public Reduction<T,T>

{
 public:

  KahanSum () :

    err_    ( T() ),
    result_ ( T() )

  {}


  KahanSum ( const KahanSum& rhs ) :

    err_    ( rhs.err_    ),
    result_ ( rhs.result_ )

  {}


  void init ( const T& r )
  {
    err_    = T ();
    result_ = r;
  }


  void feed ( const T& a )
  {
    T  t = a - err_;
    T  u = result_ + t;

    err_    = (u - result_) - t;
    result_ = u;
  }


  T result () const
  {
    return result_;
  }


 private:

  T  err_;
  T  result_;

};


//-----------------------------------------------------------------------
//   class TrueCount
//-----------------------------------------------------------------------


template <class T>

  class TrueCount : public Reduction<T,idx_t>

{
 public:

  TrueCount () :

    count_ ( 0_idx )

  {}


  TrueCount ( const TrueCount& rhs ) :

    count_ ( rhs.count_ )

  {}


  void init ( const T& a )
  {
    count_ = (a) ? 1_idx : 0_idx;
  }


  void feed ( const T& a )
  {
    if ( a )
    {
      count_++;
    }
  }


  idx_t result () const
  {
    return count_;
  }


 private:

  idx_t  count_;

};


//-----------------------------------------------------------------------
//   class EqualCount
//-----------------------------------------------------------------------


template <class T, class U = T>

  class EqualCount : public Reduction<T,idx_t>

{
 public:

  explicit EqualCount ( const T& value ) :

    value_ ( value ),
    count_ ( 0_idx )

  {}


  EqualCount ( const EqualCount& rhs ) :

    value_ ( rhs.value_ ),
    count_ ( rhs.count_ )

  {}


  void init ( const T& a )
  {
    count_ = (a == value_) ? 1_idx : 0_idx;
  }


  void feed ( const T& a )
  {
    if ( a == value_ )
    {
      count_++;
    }
  }


  idx_t result () const
  {
    return count_;
  }


 private:

  U      value_;
  idx_t  count_;

};


JEM_END_PACKAGE_BASE

#endif
