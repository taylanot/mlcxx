
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

#ifndef JEM_UTIL_PATTERN_H
#define JEM_UTIL_PATTERN_H

#include <jem/base/String.h>
#include <jem/util/Range.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class Pattern
//-----------------------------------------------------------------------


class Pattern
{
 public:

  typedef Range<idx_t>    Span;


  static void             check

    ( const String&         pattern );

  static bool             matches

    ( const String&         pattern,
      const String&         word );

  static bool             contains

    ( const String&         pattern,
      const String&         word );


  static inline Span      findShortestMatch

    ( const String&         pattern,
      const String&         word );

  static inline Span      findShortestMatch

    ( const String&         pattern,
      idx_t                 istart,
      const String&         word );

  static Span             findShortestMatch

    ( const String&         pattern,
      idx_t                 istart,
      const char*           word,
      idx_t                 wsize );

  static inline Span      rfindShortestMatch

    ( const String&         pattern,
      const String&         word );

  static inline Span      rfindShortestMatch

    ( const String&         pattern,
      idx_t                 istart,
      const String&         word );

  static Span             rfindShortestMatch

    ( const String&         pattern,
      idx_t                 istart,
      const char*           word,
      idx_t                 wsize );

  static inline Span      findLongestMatch

    ( const String&         pattern,
      const String&         word );

  static inline Span      findLongestMatch

    ( const String&         pattern,
      idx_t                 istart,
      const String&         word );

  static Span             findLongestMatch

    ( const String&         pattern,
      idx_t                 istart,
      const char*           word,
      idx_t                 wsize );

  static inline Span      rfindLongestMatch

    ( const String&         pattern,
      const String&         word );

  static inline Span      rfindLongestMatch

    ( const String&         pattern,
      idx_t                 istart,
      const String&         word );

  static Span             rfindLongestMatch

    ( const String&         pattern,
      idx_t                 istart,
      const char*           word,
      idx_t                 wsize );


 private:

  class                   Utils_;
  friend class            Utils_;

  static const char*      CLASS_NAME_;

};




//#######################################################################
//   Implementation
//#######################################################################


inline Range<idx_t> Pattern::findShortestMatch

  ( const String&  pattern,
    const String&  word )

{
  return findShortestMatch ( pattern, 0,
                             word.addr(), word.size() );
}


inline Range<idx_t> Pattern::findShortestMatch

  ( const String&  pattern,
    idx_t          istart,
    const String&  word )

{
  return findShortestMatch ( pattern, istart,
                             word.addr(), word.size() );
}


inline Range<idx_t> Pattern::rfindShortestMatch

  ( const String&  pattern,
    const String&  word )

{
  return rfindShortestMatch ( pattern, word.size() - 1,
                              word.addr(), word.size() );
}


inline Range<idx_t> Pattern::rfindShortestMatch

  ( const String&  pattern,
    idx_t          istart,
    const String&  word )

{
  return rfindShortestMatch ( pattern, istart,
                              word.addr(), word.size() );
}


inline Range<idx_t> Pattern::findLongestMatch

  ( const String&  pattern,
    const String&  word )

{
  return findLongestMatch ( pattern, 0,
                            word.addr(), word.size() );
}


inline Range<idx_t> Pattern::findLongestMatch

  ( const String&  pattern,
    idx_t          istart,
    const String&  word )

{
  return findLongestMatch ( pattern, istart,
                            word.addr(), word.size() );
}


inline Range<idx_t> Pattern::rfindLongestMatch

  ( const String&  pattern,
    const String&  word )

{
  return rfindLongestMatch ( pattern, word.size() - 1,
                             word.addr(), word.size() );
}


inline Range<idx_t> Pattern::rfindLongestMatch

  ( const String&  pattern,
    idx_t          istart,
    const String&  word )

{
  return rfindLongestMatch ( pattern, istart,
                             word.addr(), word.size() );
}


JEM_END_PACKAGE( util )

#endif
