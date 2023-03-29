
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

#ifndef JIVE_GRAPH_PQUEUE_H
#define JIVE_GRAPH_PQUEUE_H

#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( graph )


//-----------------------------------------------------------------------
//   class PQueue
//-----------------------------------------------------------------------


class PQueue
{
 public:

  typedef PQueue      Self;


                      PQueue      ();

                      PQueue

    ( idx_t             nodeCount,
      idx_t             maxGain );

                     ~PQueue      ();

  void                init

    ( idx_t             nodeCount,
      idx_t             maxGain );

  void                swap

    ( PQueue&           rhs );

  void                clear       ();
  idx_t               popFront    ();

  void                remove

    ( idx_t             inode );

  void                insert

    ( idx_t             inode,
      idx_t             gain );

  void                update

    ( idx_t             inode,
      idx_t             gain );

  inline idx_t        size        () const;
  inline idx_t        maxGain     () const;


 private:

  class               Node_;


                      PQueue      ( const Self& );
  Self&               operator =  ( const Self& );

  void                init_

    ( idx_t             nodeCount,
      idx_t             maxGain );

  void                free_       ();

  inline Node_*       getChain_

    ( idx_t             gain )       const;


 private:

  static const int    MAX_CHAINS_;

  Node_*              nodes_;
  Node_*              chains_;
  Node_*              head_;

  idx_t               nodeCount_;
  idx_t               chainCount_;
  double              scale_;
  idx_t               maxGain_;
  idx_t               size_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t PQueue::size () const
{
  return size_;
}


//-----------------------------------------------------------------------
//   maxGain
//-----------------------------------------------------------------------


inline idx_t PQueue::maxGain () const
{
  return maxGain_;
}


JIVE_END_PACKAGE( graph )

#endif
