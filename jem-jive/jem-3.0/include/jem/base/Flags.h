
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

#ifndef JEM_BASE_FLAGS_H
#define JEM_BASE_FLAGS_H

#include <initializer_list>
#include <jem/defines.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class Flags
//-----------------------------------------------------------------------


template <class T>

  class Flags

{
 private:

  class                   Nop_;
  typedef int            (Nop_::*Null);

 public:

  typedef T               Enum;
  typedef std::
    initializer_list<T>   List;


  constexpr inline        Flags

    ( Null                  zero = 0 )           noexcept;

  constexpr inline        Flags

    ( Enum                  rhs )                noexcept;

  constexpr inline        Flags

    ( List                  rhs )                noexcept;

  constexpr
    explicit inline       Flags

    ( int                   rhs )                noexcept;

  constexpr inline        operator int  () const noexcept;
  constexpr inline bool   operator !    () const noexcept;
  constexpr inline Flags  operator ~    () const noexcept;

  inline Flags&           operator &=

    ( int                   rhs )                noexcept;

  inline Flags&           operator &=

    ( Flags                 rhs )                noexcept;

  inline Flags&           operator &=

    ( Enum                  rhs )                noexcept;

  inline Flags&           operator |=

    ( Flags                 rhs )                noexcept;

  inline Flags&           operator |=

    ( Enum                  rhs )                noexcept;

  inline Flags&           operator ^=

    ( Flags                 rhs )                noexcept;

  inline Flags&           operator ^=

    ( Enum                  rhs )                noexcept;

  constexpr inline Flags  operator &

    ( int                   rhs )          const noexcept;

  constexpr inline Flags  operator &

    ( Flags                 rhs )          const noexcept;

  constexpr inline Flags  operator &

    ( Enum                  rhs )          const noexcept;

  constexpr inline Flags  operator |

    ( Flags                 rhs )          const noexcept;

  constexpr inline Flags  operator |

    ( Enum                  rhs )          const noexcept;

  constexpr inline Flags  operator ^

    ( Flags                 rhs )          const noexcept;

  constexpr inline Flags  operator ^

    ( Enum                  rhs )          const noexcept;

  inline Flags            set

    ( Enum                  flag,
      bool                  yesno = true )       noexcept;

  inline Flags            toggle

    ( Enum                  flag )               noexcept;

  constexpr inline bool   test

    ( Enum                  flag )         const noexcept;

  constexpr inline bool   testall

    ( Flags                 flags )        const noexcept;

  constexpr inline bool   testany

    ( Flags                 flags )        const noexcept;


 private:

  constexpr
    inline static int     eval_

    ( typename List::
        const_iterator      it,
      typename List::
        const_iterator      end )                noexcept;


 private:

  int                     flags_;

};


//-----------------------------------------------------------------------
//   related functions and macros
//-----------------------------------------------------------------------


template <class Input,
          class T>

  inline void             decode

  ( Input&                  in,
    Flags<T>&               flags );

template <class Output,
          class T>

  inline void             encode

  ( Output&                 out,
    Flags<T>                flags );

template <class Output,
          class T>

  inline void             print

  ( Output&                 out,
    Flags<T>                flags );


#undef  JEM_DEFINE_FLAG_OPS
#define JEM_DEFINE_FLAG_OPS( F )                          \
                                                          \
constexpr inline F        operator |                      \
                                                          \
  ( F::Enum                 lhs,                          \
    F::Enum                 rhs )                noexcept \
                                                          \
{                                                         \
  return (F( lhs ) | F( rhs ));                           \
}



//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T>

  constexpr inline Flags<T>::Flags ( Null zero ) noexcept :

  flags_ ( 0 )

{}


template <class T>

  constexpr inline Flags<T>::Flags ( Enum rhs ) noexcept :

  flags_ ( (int) rhs )

{}


template <class T>

  constexpr inline Flags<T>::Flags ( List rhs ) noexcept :

  flags_ ( eval_( rhs.begin(), rhs.end() ) )

{}


template <class T>

  constexpr inline Flags<T>::Flags ( int rhs ) noexcept :

  flags_ ( rhs )

{}


//-----------------------------------------------------------------------
//   unary operators
//-----------------------------------------------------------------------


template <class T>

  constexpr inline Flags<T>::operator int () const noexcept

{
  return flags_;
}


template <class T>

  constexpr inline bool Flags<T>::operator ! () const noexcept

{
  return (flags_ ? false : true);
}


template <class T>

  constexpr inline Flags<T> Flags<T>::operator ~ () const noexcept

{
  return Flags ( ~flags_ );
}


//-----------------------------------------------------------------------
//   computed assignment operators
//-----------------------------------------------------------------------


template <class T>

  inline Flags<T>& Flags<T>::operator &= ( int rhs ) noexcept

{
  flags_ &= rhs;

  return *this;
}


template <class T>

  inline Flags<T>& Flags<T>::operator &= ( Flags rhs ) noexcept

{
  flags_ &= rhs.flags_;

  return *this;
}


template <class T>

  inline Flags<T>& Flags<T>::operator &= ( Enum rhs ) noexcept

{
  flags_ &= (int) rhs;

  return *this;
}


template <class T>

  inline Flags<T>& Flags<T>::operator |= ( Flags rhs ) noexcept

{
  flags_ |= rhs.flags_;

  return *this;
}


template <class T>

  inline Flags<T>& Flags<T>::operator |= ( Enum rhs ) noexcept

{
  flags_ |= (int) rhs;

  return *this;
}


template <class T>

  inline Flags<T>& Flags<T>::operator ^= ( Flags rhs ) noexcept

{
  flags_ ^= rhs.flags_;

  return *this;
}


template <class T>

  inline Flags<T>& Flags<T>::operator ^= ( Enum rhs ) noexcept

{
  flags_ ^= (int) rhs;

  return *this;
}


//-----------------------------------------------------------------------
//   binary operators
//-----------------------------------------------------------------------


template <class T>

  constexpr inline Flags<T>
  Flags<T>::operator & ( int rhs ) const noexcept

{
  return Flags ( flags_ & rhs );
}


template <class T>

  constexpr inline Flags<T>
  Flags<T>::operator & ( Flags rhs ) const noexcept

{
  return Flags ( flags_ & rhs.flags_ );
}


template <class T>

  constexpr inline Flags<T>
  Flags<T>::operator & ( Enum rhs ) const noexcept

{
  return Flags ( flags_ & (int) rhs );
}


template <class T>

  constexpr inline Flags<T>
  Flags<T>::operator | ( Flags rhs ) const noexcept

{
  return Flags ( flags_ | rhs.flags_ );
}


template <class T>

  constexpr inline Flags<T>
  Flags<T>::operator | ( Enum rhs ) const noexcept

{
  return Flags ( flags_ | (int) rhs );
}


template <class T>

  constexpr inline Flags<T>
  Flags<T>::operator ^ ( Flags rhs ) const noexcept

{
  return Flags ( flags_ ^ rhs.flags_ );
}


template <class T>

  constexpr inline Flags<T>
  Flags<T>::operator ^ ( Enum rhs ) const noexcept

{
  return Flags ( flags_ ^ (int) rhs );
}


//-----------------------------------------------------------------------
//   binary operators (no members)
//-----------------------------------------------------------------------


template <class T>

  constexpr inline Flags<T>
  operator & ( int lhs, Flags<T> rhs ) noexcept

{
  return (rhs & lhs);
}


template <class T>

  constexpr inline Flags<T>
  operator & ( T lhs, Flags<T> rhs ) noexcept

{
  return (rhs & lhs);
}


template <class T>

  constexpr inline Flags<T>
  operator | ( T lhs, Flags<T> rhs ) noexcept

{
  return (rhs | lhs);
}


template <class T>

  constexpr inline Flags<T>
  operator ^ ( T lhs, Flags<T> rhs ) noexcept

{
  return (rhs ^ lhs);
}


template <class T>

  constexpr inline bool
  operator == ( Flags<T> lhs, T rhs ) noexcept

{
  return ((int) lhs == (int) rhs);
}


template <class T>

  constexpr inline bool
  operator == ( T lhs, Flags<T> rhs ) noexcept

{
  return ((int) lhs == (int) rhs);
}


template <class T>

  constexpr inline bool
  operator != ( Flags<T> lhs, T rhs ) noexcept

{
  return ((int) lhs != (int) rhs);
}


template <class T>

  constexpr inline bool
  operator != ( T lhs, Flags<T> rhs ) noexcept

{
  return ((int) lhs != (int) rhs);
}


//-----------------------------------------------------------------------
//   set
//-----------------------------------------------------------------------


template <class T>

  inline Flags<T> Flags<T>::set

    ( Enum  flag,
      bool  yesno ) noexcept

{
  if ( yesno )
  {
    flags_ |=   (int) flag;
  }
  else
  {
    flags_ &= ~((int) flag);
  }

  return *this;
}


//-----------------------------------------------------------------------
//   toggle
//-----------------------------------------------------------------------


template <class T>

  inline Flags<T> Flags<T>::toggle ( Enum flag ) noexcept

{
  flags_ ^= (int) flag;

  return *this;
}


//-----------------------------------------------------------------------
//   test
//-----------------------------------------------------------------------


template <class T>

  constexpr inline bool Flags<T>::test ( Enum flag ) const noexcept

{
  return ((((int) flag & flags_) == (int) flag) &&
          (((int) flag != 0) || (flags_ == 0)));
}


//-----------------------------------------------------------------------
//   testall
//-----------------------------------------------------------------------


template <class T>

  constexpr inline bool
  Flags<T>::testall ( Flags flags ) const noexcept

{
  return (((flags.flags_ & flags_) == flags.flags_) &&
          ((flags.flags_ != 0) || (flags_ == 0)));
}


//-----------------------------------------------------------------------
//   testany
//-----------------------------------------------------------------------


template <class T>

  constexpr inline bool
  Flags<T>::testany ( Flags flags ) const noexcept

{
  return (((flags.flags_ & flags_) != 0) ||
          (flags.flags_ == flags_));
}


//-----------------------------------------------------------------------
//   eval_
//-----------------------------------------------------------------------


template <class T>

  constexpr inline int Flags<T>::eval_

  ( typename List::const_iterator  it,
    typename List::const_iterator  end ) noexcept

{
  return (it == end ? 0 : (int) *it | eval_( it + 1, end ));
}


//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input,
          class T>

  inline void  decode

  ( Input&       in,
    Flags<T>&    flags )

{
  int  itmp;

  decode ( in, itmp );

  flags = Flags<T> ( itmp );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output,
          class T>

  inline void  encode

  ( Output&      out,
    Flags<T>     flags )

{
  encode ( out, (int) flags );
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output,
          class T>

  inline void  print

  ( Output&      out,
    Flags<T>     flags )

{
  print ( out, (int) flags );
}


JEM_END_PACKAGE_BASE

#endif
