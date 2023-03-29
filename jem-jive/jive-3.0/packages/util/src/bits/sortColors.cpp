
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


#include <jem/base/assert.h>
#include <jem/base/Array.h>
#include <jive/util/utilities.h>


JIVE_BEGIN_PACKAGE( util )


//=======================================================================
//   private functions
//=======================================================================

//-----------------------------------------------------------------------
//   sortSmallColorRange
//-----------------------------------------------------------------------


static IdxVector    sortSmallColorRange

  ( const IdxVector&  iperm,
    const IdxVector&  colors,
    idx_t             minColor,
    idx_t             maxColor )

{
  const idx_t  n            = colors.size ();
  const idx_t  colorRange   = maxColor - minColor + 1;

  IdxVector    colorOffsets ( colorRange );
  IdxVector    listOffsets;

  idx_t        listCount;
  idx_t        i, j, k, m;


  colorOffsets = 0;

  for ( i = 0; i < n; i++ )
  {
    colorOffsets[colors[i] - minColor]++;
  }

  listCount = jem::count ( colorOffsets );

  listOffsets.resize ( listCount + 1 );

  j = k = 0;

  for ( i = 0; i < colorRange; i++ )
  {
    m = colorOffsets[i];
    colorOffsets[i] = k;

    if ( m > 0 )
    {
      listOffsets[j] = k;
      k += m;
      j++;
    }
  }

  listOffsets[listCount] = k;

  for ( i = 0; i < n; i++ )
  {
    j        = colors[i] - minColor;
    k        = colorOffsets[j];
    iperm[k] = i;

    colorOffsets[j]++;
  }

  return listOffsets;
}


//-----------------------------------------------------------------------
//   sortLargeColorRange
//-----------------------------------------------------------------------


static IdxVector    sortLargeColorRange

  ( const IdxVector&  iperm,
    const IdxVector&  colors )

{
  const idx_t  n = colors.size ();

  IdxVector    listOffsets;

  idx_t        listCount;
  idx_t        icolor;
  idx_t        jcolor;
  idx_t        i, j;


  for ( i = 0; i < n; i++ )
  {
    iperm[i] = i;
  }

  jem::sort ( iperm, colors );

  icolor    = colors[iperm[0]];
  listCount = 1;

  for ( i = 1; i < n; i++ )
  {
    jcolor = colors[iperm[i]];

    if ( jcolor != icolor )
    {
      icolor = jcolor;
      listCount++;
    }
  }

  listOffsets.resize ( listCount + 1 );

  listOffsets[0] = 0;
  icolor         = colors[iperm[0]];
  j              = 1;

  for ( i = 1; i < n; i++ )
  {
    jcolor = colors[iperm[i]];

    if ( jcolor != icolor )
    {
      listOffsets[j] = i;
      icolor         = jcolor;
      j++;
    }
  }

  listOffsets[listCount] = n;

  return listOffsets;
}


//=======================================================================
//   public functions
//=======================================================================

//-----------------------------------------------------------------------
//   sortColors
//-----------------------------------------------------------------------


IdxVector           sortColors

  ( const IdxVector&  iperm,
    const IdxVector&  colors )

{
  JEM_PRECHECK2 ( iperm.size() == colors.size(),
                  "Array size mismatch" );

  const idx_t  n = colors.size ();

  if ( n == 0 )
  {
    IdxVector offsets ( 1 );

    offsets[0] = 0;

    return offsets;
  }
  else
  {
    idx_t  minColor = colors[0];
    idx_t  maxColor = colors[0];

    for ( idx_t i = 1; i < n; i++ )
    {
      idx_t  color = colors[i];

      if ( color < minColor )
      {
        minColor = color;
      }
      else if ( color > maxColor )
      {
        maxColor = color;
      }
    }

    return sortColors ( iperm, colors, minColor, maxColor );
  }
}


IdxVector           sortColors

  ( const IdxVector&  iperm,
    const IdxVector&  colors,
    idx_t             minColor,
    idx_t             maxColor )

{
  JEM_PRECHECK2 ( iperm.size() == colors.size(),
                  "Array size mismatch" );

  const idx_t  n = colors.size ();

  if ( n == 0 || maxColor < minColor )
  {
    IdxVector offsets ( 1 );

    offsets[0] = 0;

    return offsets;
  }
  else if ( maxColor - minColor > 2 * n )
  {
    return sortLargeColorRange ( iperm, colors );
  }
  else
  {
    return sortSmallColorRange ( iperm, colors,
                                 minColor, maxColor );
  }
}


JIVE_END_PACKAGE( util )
