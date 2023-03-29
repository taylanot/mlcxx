
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
#include <jem/base/Error.h>
#include "ReduceTree.h"


JEM_BEGIN_PACKAGE   ( mp )
JEM_BEGIN_NAMESPACE ( mt )


//=======================================================================
//   class ReduceTree
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ReduceTree::ReduceTree ( int threadCount )
{
  JEM_PRECHECK ( threadCount > 0 );

  if ( threadCount <= 1 )
  {
    nodes_     = 0;
    leaves_    = new Node* [1];
    leaves_[0] = 0;
    depth_     = 1;
  }
  else
  {
    idx_t  i, j, k, n;


    n = k  = 1;
    depth_ = 1;

    while ( 4 * k <= threadCount )
    {
      k *= 2;
      n += k;

      depth_++;
    }

    if ( threadCount > (2 * k) )
    {
      depth_++;
    }

    n      += (threadCount - 2 * k);
    nodes_  = 0;
    leaves_ = 0;

    try
    {
      nodes_  = new Node  [n];
      leaves_ = new Node* [threadCount];
    }
    catch ( ... )
    {
      if ( nodes_ )
      {
        delete [] nodes_;
      }
      if ( leaves_ )
      {
        delete [] leaves_;
      }

      throw;
    }

    n = n - 1;

    for ( i = 0, k = 1; k < n; i++, k += 2 )
    {
      nodes_[k + 0].parent = nodes_ + i;
      nodes_[k + 1].parent = nodes_ + i;
    }

    j = 0;

    if ( k == n )
    {
      nodes_ [k].parent = nodes_ + i;
      leaves_[j]        = nodes_ + i;

      i++;
      j++;
    }

    for ( ; i <= n; i++, j += 2 )
    {
      leaves_[j + 0] = nodes_ + i;
      leaves_[j + 1] = nodes_ + i;
    }

    JEM_ASSERT ( j == threadCount );
  }
}


ReduceTree::~ReduceTree ()
{
  if ( nodes_ )
  {
    delete [] nodes_;
  }

  delete [] leaves_;

  nodes_  = 0;
  leaves_ = 0;
}


JEM_END_NAMESPACE ( mt )
JEM_END_PACKAGE   ( mp )
