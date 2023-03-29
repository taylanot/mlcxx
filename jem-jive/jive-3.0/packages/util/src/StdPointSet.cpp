
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


#include <cstring>
#include <jem/pragmas.h>
#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Flex.h>
#include <jive/util/error.h>
#include <jive/util/EventMask.h>
#include <jive/util/Reordering.h>
#include <jive/util/HashItemMap.h>
#include <jive/util/StdPointSet.h>


JEM_DEFINE_SERIABLE_CLASS( jive::util::StdPointSet );


JIVE_BEGIN_PACKAGE( util )


using jem::checkedCast;
using jem::newInstance;
using jem::util::Flex;


//=======================================================================
//   class StdPointSet::Data_
//=======================================================================


class StdPointSet::Data_ : public jem::Collectable
{
 public:

  typedef Data_         Self;

  static const int      NEW_SIZE = 1 << 0;
  static const int      NEW_GEOM = 1 << 1;


  explicit              Data_

    ( const String&       ptName );

  void                  readFrom

    ( ObjectInput&        in );

  void                  writeTo

    ( ObjectOutput&       out )        const;

  inline void           checkDim

    ( idx_t               idim )       const;

  inline void           checkPoint

    ( idx_t               ipoint )     const;

  inline void           checkSize

    ( const Vector&       coords,
      idx_t               size )       const;

  inline void           checkShape

    ( const Matrix&       coords,
      idx_t               colCount )   const;

  inline void           addPoint

    ( const Vector&       coords,
      StdPointSet*        pntSet );


 public:

  String                pointName;
  Flex<double>          points;
  EventMask             events;

  idx_t                 rank;
  idx_t                 pointCount;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


StdPointSet::Data_::Data_ ( const String&  ptName ) :

  pointName ( ptName )

{
  rank       = 0;
  pointCount = 0;

  if ( ! pointName.size() )
  {
    pointName = "point";
  }
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void StdPointSet::Data_::readFrom ( ObjectInput& in )
{
  uidx_t  r;

  decode ( in, pointName, points, r );

  rank = (idx_t) r;

  if ( rank > 0 )
  {
    pointCount = points.size() / rank;
  }
  else
  {
    pointCount = 0;
  }

  if ( rank * pointCount != points.size() )
  {
    jem::io::decodeError ( JEM_FUNC );
  }

  events.reset ();
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void StdPointSet::Data_::writeTo ( ObjectOutput& out ) const
{
  encode ( out, pointName, points, rank );
}


//-----------------------------------------------------------------------
//   checkDim
//-----------------------------------------------------------------------


inline void StdPointSet::Data_::checkDim ( idx_t idim ) const
{
  if ( idim < 0 || idim >= rank )
  {
    if ( pointCount > 0 )
    {
      indexError ( CLASS_NAME, "dimension", idim, rank );
    }
  }
}


//-----------------------------------------------------------------------
//   checkPoint
//-----------------------------------------------------------------------


inline void StdPointSet::Data_::checkPoint ( idx_t ipoint ) const
{
  if ( ipoint < 0 || ipoint >= pointCount )
  {
    indexError ( CLASS_NAME, pointName, ipoint, pointCount );
  }
}


//-----------------------------------------------------------------------
//   checkSize
//-----------------------------------------------------------------------


inline void StdPointSet::Data_::checkSize

  ( const Vector&  coords,
    idx_t          size ) const

{
  if ( coords.size() < size )
  {
    sizeError ( CLASS_NAME, "coordinate vector",
                coords.size(), size );
  }
}


//-----------------------------------------------------------------------
//   checkShape
//-----------------------------------------------------------------------


inline void StdPointSet::Data_::checkShape

  ( const Matrix&  coords,
    idx_t          colCount ) const

{
  if ( coords.size(0) != rank || coords.size(1) < colCount )
  {
    if ( pointCount > 0 )
    {
      shapeError ( CLASS_NAME, "coordinate matrix",
                   coords.shape (),
                   jem::shape ( rank, colCount ) );
    }
  }
}


//-----------------------------------------------------------------------
//   addPoint
//-----------------------------------------------------------------------


inline void StdPointSet::Data_::addPoint

  ( const Vector&  coords,
    StdPointSet*   pntSet )

{
  const idx_t  n = coords.size ();

  if ( pointCount )
  {
    checkSize ( coords, rank );
  }

  for ( idx_t i = 0; i < n; i++ )
  {
    points.pushBack ( coords[i] );
  }

  pointCount++;

  if ( pointCount == 1 )
  {
    rank = n;

    pntSet->newRankEvent.emit ( n, *pntSet );
    pntSet->changeEvent .emit (    *pntSet );
  }

  events.emit ( NEW_SIZE, pntSet );
}


//=======================================================================
//   class StdPointSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


StdPointSet::StdPointSet () : Super ( "points" )
{
  data_ = newInstance<Data_>       ( "point" );
  map_  = newInstance<HashItemMap> ();
}


StdPointSet::StdPointSet

  ( const String&         name,
    const String&         pointName,
    const Ref<XItemMap>&  map ) :

    Super ( name ),
    map_  (  map )

{
  data_ = newInstance<Data_> ( pointName );

  if ( map_ )
  {
    map_->clear ();
  }
  else
  {
    map_ = newInstance<HashItemMap> ();
  }
}


StdPointSet::StdPointSet ( const Self& rhs ) :

  Super ( rhs.myName_ )

{
  data_ = newInstance<Data_>    ( * rhs.data_ );
  map_  = checkedCast<XItemMap> (   rhs.map_->clone() );
}


StdPointSet::~StdPointSet ()
{}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


void StdPointSet::resetEvents ()
{
  data_->events.reset ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void StdPointSet::readFrom ( ObjectInput& in )
{
  decode ( in, myName_ );
  data_->readFrom ( in );
  decode ( in, map_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void StdPointSet::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_ );
  data_->writeTo ( out );
  encode ( out, map_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> StdPointSet::clone () const
{
  return newInstance<Self> ( *this );
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


idx_t StdPointSet::size () const
{
  return data_->pointCount;
}


//-----------------------------------------------------------------------
//   rank
//-----------------------------------------------------------------------


idx_t StdPointSet::rank () const
{
  return data_->rank;
}


//-----------------------------------------------------------------------
//   getItemName
//-----------------------------------------------------------------------


String StdPointSet::getItemName () const
{
  return data_->pointName;
}


//-----------------------------------------------------------------------
//   getItemMap
//-----------------------------------------------------------------------


ItemMap* StdPointSet::getItemMap () const
{
  return map_.get ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void StdPointSet::clear ()
{
  Data_&  d = * data_;

  if ( d.pointCount > 0 )
  {
    map_   ->clear ();
    d.points.clear ();
    d.events.reset ();

    d.rank       = 0;
    d.pointCount = 0;

    newSizeEvent.emit ( 0, *this );
    changeEvent .emit (    *this );
  }
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void StdPointSet::reserve ( idx_t n )
{
  Data_&  d = * data_;

  if ( d.rank > 0 )
  {
    d.points.reserve ( d.rank * n );
  }
  else
  {
    d.points.reserve ( n );
  }

  map_->reserve ( n );
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void StdPointSet::trimToSize ()
{
  data_->points.trimToSize ();
  map_        ->trimToSize ();
}


//-----------------------------------------------------------------------
//   addPoint
//-----------------------------------------------------------------------


idx_t StdPointSet::addPoint ( const Vector& coords )
{
  Data_&       d  = * data_;

  const idx_t  ip = map_->addItem ();

  d.addPoint ( coords, this );

  return ip;
}


idx_t StdPointSet::addPoint

  ( idx_t          id,
    const Vector&  coords )

{
  Data_&       d  = * data_;

  const idx_t  ip = map_->addItem ( id );

  d.addPoint ( coords, this );

  return ip;
}


//-----------------------------------------------------------------------
//   addPoints
//-----------------------------------------------------------------------


idx_t StdPointSet::addPoints ( const Matrix& coords )
{
  Data_&       d  = * data_;

  const idx_t  m  = coords.size (0);
  const idx_t  n  = coords.size (1);

  const idx_t  ip = map_->addItems ( n );


  if ( n == 0 )
  {
    return ip;
  }

  d.checkShape ( coords, n );

  if ( coords.isContiguous() )
  {
    d.points.pushBack ( coords.addr(), coords.addr() + (m * n) );
  }
  else
  {
    for ( idx_t j = 0; j < n; j++ )
    {
      for ( idx_t i = 0; i < m; i++ )
      {
        d.points.pushBack ( coords(i,j) );
      }
    }
  }

  d.pointCount += n;

  if ( d.pointCount == n )
  {
    d.rank = m;

    newRankEvent.emit ( m, *this );
    changeEvent .emit (    *this );
  }

  d.events.emit ( Data_::NEW_SIZE, this );

  return ip;
}


//-----------------------------------------------------------------------
//   reorderPoints
//-----------------------------------------------------------------------


void StdPointSet::reorderPoints ( const Reordering& reord )
{
  JEM_PRECHECK2 ( reord.oldSize() == size(),
                  "invalid Reordering" );

  Data_&        d = * data_;

  const idx_t*  JEM_RESTRICT  rmap = reord.rmapPtr ();
  double*       JEM_RESTRICT  xp   = d.points.addr ();

  const idx_t   start   = reord.start   ();
  const idx_t   newSize = reord.newSize ();
  const idx_t   r       = d.rank;


  map_->reorderItems ( reord );

  if ( reord.isSimple() )
  {
    if      ( r == 1_idx )
    {
      for ( idx_t i = start; i < newSize; i++ )
      {
        xp[i] = xp[rmap[i]];
      }
    }
    else if ( r == 2_idx )
    {
      for ( idx_t ipoint = start; ipoint < newSize; ipoint++ )
      {
        idx_t  i = 2 * ipoint;
        idx_t  j = 2 * rmap[ipoint];

        xp[i + 0] = xp[j + 0];
        xp[i + 1] = xp[j + 1];
      }
    }
    else if ( r == 3_idx )
    {
      for ( idx_t ipoint = start; ipoint < newSize; ipoint++ )
      {
        idx_t  i = 3 * ipoint;
        idx_t  j = 3 * rmap[ipoint];

        xp[i + 0] = xp[j + 0];
        xp[i + 1] = xp[j + 1];
        xp[i + 2] = xp[j + 2];
      }
    }
    else if ( r >= 4_idx )
    {
      for ( idx_t ipoint = start; ipoint < newSize; ipoint++ )
      {
        idx_t  i = r * ipoint;
        idx_t  j = r * rmap[ipoint];

        for ( idx_t k = 0; k < r; k++ )
        {
          xp[i + k] = xp[j + k];
        }
      }
    }

    d.points.resize ( r * newSize );
  }
  else
  {
    Flex<double>  tmpPoints ( r * newSize );

    double* JEM_RESTRICT  y = tmpPoints.addr ();


    if ( start > 0 )
    {
      std::memcpy ( y, xp, (size_t) (r * start) * sizeof(double) );

      y += (r * start);
    }

    for ( idx_t ipoint = start; ipoint < newSize; ipoint++, y += r )
    {
      idx_t  j = r * rmap[ipoint];

      for ( idx_t k = 0; k < r; k++ )
      {
        y[k] = xp[j + k];
      }
    }

    d.points.swap ( tmpPoints );
  }

  d.pointCount = newSize;

  d.events.reset     ();
  newOrderEvent.emit ( reord, *this );

  if ( newSize != reord.oldSize() )
  {
    newSizeEvent.emit ( newSize, *this );
  }

  changeEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   getPointCoords
//-----------------------------------------------------------------------


void StdPointSet::getPointCoords

  ( const Vector&  coords,
    idx_t          ipoint ) const

{
  const Data_&   d = * data_;
  const idx_t    r = d.rank;

  const double*  JEM_RESTRICT  xp = d.points.addr ();


  d.checkPoint ( ipoint );
  d.checkSize  ( coords, r );

  xp += r * ipoint;

  for ( idx_t i = 0; i < r; i++ )
  {
    coords[i] = xp[i];
  }
}


//-----------------------------------------------------------------------
//   getPointXCoord
//-----------------------------------------------------------------------


double StdPointSet::getPointXCoord

  ( idx_t  ipoint,
    idx_t  idim ) const

{
  const Data_&  d = * data_;

  d.checkPoint ( ipoint );
  d.checkDim   ( idim   );

  return d.points[ipoint * d.rank + idim];
}


//-----------------------------------------------------------------------
//   setPointCoords
//-----------------------------------------------------------------------


void StdPointSet::setPointCoords

  ( idx_t          ipoint,
    const Vector&  coords )

{
  Data_&       d = * data_;
  const idx_t  r = d.rank;

  double*      JEM_RESTRICT  xp = d.points.addr ();


  d.checkPoint ( ipoint );
  d.checkSize  ( coords, r );

  xp += r * ipoint;

  for ( idx_t i = 0; i < r; i++ )
  {
    xp[i] = coords[i];
  }

  d.events.emit ( Data_::NEW_GEOM, this );
}


//-----------------------------------------------------------------------
//   setPointXCoord
//-----------------------------------------------------------------------


void StdPointSet::setPointXCoord

  ( idx_t   ipoint,
    double  coord,
    idx_t   idim )

{
  Data_&  d = * data_;

  d.checkPoint ( ipoint );
  d.checkDim   ( idim   );

  d.points[ipoint * d.rank + idim] = coord;

  d.events.emit ( Data_::NEW_GEOM, this );
}


//-----------------------------------------------------------------------
//   getCoords
//-----------------------------------------------------------------------


void StdPointSet::getCoords ( const Matrix& coords ) const
{
  const Data_&   d = * data_;

  const double*  JEM_RESTRICT  xp = d.points.addr ();

  const idx_t    r = d.rank;
  const idx_t    n = d.pointCount;


  d.checkShape ( coords, n );

  if ( coords.stride(0) == 1_idx &&
       coords.stride(1) == (idx_t) r )
  {
    std::memcpy ( coords.addr(), xp,
                  (size_t) (r * n) * sizeof(double) );
  }
  else
  {
    for ( idx_t i = 0; i < n; i++, xp += r )
    {
      for ( idx_t k = 0; k < r; k++ )
      {
        coords(k,i) = xp[k];
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getXCoords
//-----------------------------------------------------------------------


void StdPointSet::getXCoords

  ( const Vector&  coords,
    idx_t          idim ) const

{
  const Data_&   d  = * data_;

  const double*  JEM_RESTRICT  xp = d.points.addr ( idim );
  double*        JEM_RESTRICT  cp = coords  .addr ();

  const idx_t    r  = d.rank;
  const idx_t    n  = d.pointCount;
  const idx_t    st = coords.stride ();


  d.checkDim  ( idim );
  d.checkSize ( coords, n );

  if      ( st + r == 2_idx )
  {
    std::memcpy ( cp, xp, (size_t) n * sizeof(double) );
  }
  else if ( st == 1 )
  {
    idx_t  i, j;

    for ( i = j = 0; i < n; i++, j += r )
    {
      cp[i] = xp[j];
    }
  }
  else
  {
    const idx_t  k = n * st;
    idx_t        i, j;

    for ( i = j = 0; i < k; i += st, j += r )
    {
      cp[i] = xp[j];
    }
  }
}


//-----------------------------------------------------------------------
//   getSomeCoords
//-----------------------------------------------------------------------


void StdPointSet::getSomeCoords

  ( const Matrix&     coords,
    const IdxVector&  ipoints ) const

{
  const Data_&   d = * data_;

  const double*  JEM_RESTRICT  xp = d.points.addr ();

  const idx_t    r = d.rank;
  const idx_t    n = ipoints.size ();


  d.checkShape ( coords, n );

#ifndef NDEBUG

  for ( idx_t i = 0; i < n; i++ )
  {
    d.checkPoint ( ipoints[i] );
  }

#endif

  if ( coords.stride(1) == (idx_t) r &&
       coords.stride(0) + ipoints.stride() == 2_idx && r < 4_idx )
  {
    const idx_t* JEM_RESTRICT  ip = ipoints.addr ();
    double*      JEM_RESTRICT  cp = coords .addr ();

    if      ( r == 1_idx )
    {
JEM_IVDEP

      for ( idx_t i = 0; i < n; i++ )
      {
        cp[i] = xp[ip[i]];
      }
    }
    else if ( r == 2_idx )
    {
JEM_IVDEP

      for ( idx_t i = 0; i < n; i++, cp += 2 )
      {
        idx_t  j = 2 * ip[i];

        cp[0]  = xp[j + 0];
        cp[1]  = xp[j + 1];
      }
    }
    else if ( r == 3_idx )
    {
JEM_IVDEP

      for ( idx_t i = 0; i < n; i++, cp += 3 )
      {
        idx_t  j = 3 * ip[i];

        cp[0]  = xp[j + 0];
        cp[1]  = xp[j + 1];
        cp[2]  = xp[j + 2];
      }
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  j = r * ipoints[i];

      for ( idx_t k = 0; k < r; k++, j++ )
      {
        coords(k,i) = xp[j];
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getSomeXCoords
//-----------------------------------------------------------------------


void StdPointSet::getSomeXCoords

  ( const Vector&     coords,
    const IdxVector&  ipoints,
    idx_t             idim ) const

{
  const Data_&   d = * data_;

  const double*  JEM_RESTRICT  xp = d.points.addr ( idim );

  const idx_t    r = d.rank;
  const idx_t    n = ipoints.size ();


  d.checkDim  ( idim );
  d.checkSize ( coords, n );

#ifndef NDEBUG

  for ( idx_t i = 0; i < n; i++ )
  {
    d.checkPoint ( ipoints[i] );
  }

#endif

  if ( coords.stride() + ipoints.stride() == 2_idx )
  {
    const idx_t* JEM_RESTRICT  ip = ipoints.addr ();
    double*      JEM_RESTRICT  cp = coords .addr ();

    if ( r == 1 )
    {
      for ( idx_t i = 0; i < n; i++ )
      {
        cp[i] = xp[ip[i]];
      }
    }
    else
    {
      for ( idx_t i = 0; i < n; i++ )
      {
        cp[i] = xp[ip[i] * r];
      }
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      coords[i] = xp[ipoints[i] * r];
    }
  }
}


//-----------------------------------------------------------------------
//   setCoords
//-----------------------------------------------------------------------


void StdPointSet::setCoords ( const Matrix& coords )
{
  Data_&       d = * data_;

  double*      JEM_RESTRICT  xp = d.points.addr ();

  const idx_t  r = d.rank;
  const idx_t  n = d.pointCount;


  d.checkShape ( coords, n );

  if ( coords.stride(0) == 1_idx &&
       coords.stride(1) == (idx_t) r )
  {
    std::memcpy ( xp, coords.addr(),
                  (size_t) (n * r) * sizeof(double) );
  }
  else
  {
    for ( idx_t i = 0; i < n; i++, xp += r )
    {
      for ( idx_t k = 0; k < r; k++ )
      {
        xp[k] = coords(k,i);
      }
    }
  }

  newGeomEvent.emit ( *this );
  changeEvent .emit ( *this );
}


//-----------------------------------------------------------------------
//   setXCoords
//-----------------------------------------------------------------------


void StdPointSet::setXCoords

  ( const Vector&  coords,
    idx_t          idim )

{
  Data_&         d  = * data_;

  double*        JEM_RESTRICT  xp = d.points.addr ( idim );
  const double*  JEM_RESTRICT  cp = coords  .addr ();

  const idx_t    r  = d.rank;
  const idx_t    n  = d.pointCount;
  const idx_t    st = coords.stride ();


  d.checkDim  ( idim );
  d.checkSize ( coords, n );

  if      ( st + r == 2_idx )
  {
    std::memcpy ( xp, cp, (size_t) n * sizeof(double) );
  }
  else if ( st == 1_idx )
  {
    idx_t  i, j;

    for ( i = j = 0; i < n; i++, j += r )
    {
      xp[j] = cp[i];
    }
  }
  else
  {
    const idx_t  k = n * st;
    idx_t        i, j;

    for ( i = j = 0; i < k; i += st, j += r )
    {
      xp[j] = cp[i];
    }
  }

  newGeomEvent.emit ( *this );
  changeEvent .emit ( *this );
}


//-----------------------------------------------------------------------
//   setSomeCoords
//-----------------------------------------------------------------------


void StdPointSet::setSomeCoords

  ( const IdxVector&  ipoints,
    const Matrix&     coords )

{
  Data_&       d = * data_;

  double*      JEM_RESTRICT  xp = d.points.addr ();

  const idx_t  r = d.rank;
  const idx_t  n = ipoints.size ();


  d.checkShape ( coords, n );

#ifndef NDEBUG

  for ( idx_t i = 0; i < n; i++ )
  {
    d.checkPoint ( ipoints[i] );
  }

#endif

  if ( coords.stride(1) == (idx_t) r &&
       coords.stride(0) + ipoints.stride() == 2_idx && r < 4_idx )
  {
    const idx_t*  JEM_RESTRICT  ip = ipoints.addr ();
    const double* JEM_RESTRICT  cp = coords .addr ();

    if      ( r == 1_idx )
    {
      for ( idx_t i = 0; i < n; i++ )
      {
        xp[ip[i]] = cp[i];
      }
    }
    else if ( r == 2_idx )
    {
      for ( idx_t i = 0; i < n; i++, cp += 2 )
      {
        idx_t  j = 2 * ip[i];

        xp[j + 0] = cp[0];
        xp[j + 1] = cp[1];
      }
    }
    else if ( r == 3_idx )
    {
      for ( idx_t i = 0; i < n; i++, cp += 3 )
      {
        idx_t  j = 3 * ip[i];

        xp[j + 0] = cp[0];
        xp[j + 1] = cp[1];
        xp[j + 2] = cp[2];
      }
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      idx_t  j = r * ipoints[i];

      for ( idx_t k = 0; k < r; k++, j++ )
      {
        xp[j] = coords(k,i);
      }
    }
  }

  d.events.emit ( Data_::NEW_GEOM, this );
}


//-----------------------------------------------------------------------
//   setSomeXCoords
//-----------------------------------------------------------------------


void StdPointSet::setSomeXCoords

  ( const IdxVector&  ipoints,
    const Vector&     coords,
    idx_t             idim )

{
  Data_&       d = * data_;

  double*      JEM_RESTRICT  xp = d.points.addr ( idim );

  const idx_t  r = d.rank;
  const idx_t  n = ipoints.size ();


  d.checkDim  ( idim );
  d.checkSize ( coords, n );

#ifndef NDEBUG

  for ( idx_t i = 0; i < n; i++ )
  {
    d.checkPoint ( ipoints[i] );
  }

#endif

  if ( coords.stride() + ipoints.stride() == 2_idx )
  {
    const idx_t*  JEM_RESTRICT  ip = ipoints.addr ();
    const double* JEM_RESTRICT  cp = coords .addr ();

    if ( r == 1_idx )
    {
      for ( idx_t i = 0; i < n; i++ )
      {
        xp[ip[i]] = cp[i];
      }
    }
    else
    {
      for ( idx_t i = 0; i < n; i++ )
      {
        xp[ip[i] * r] = cp[i];
      }
    }
  }
  else
  {
    for ( idx_t i = 0; i < n; i++ )
    {
      xp[ipoints[i] * r] = coords[i];
    }
  }

  d.events.emit ( Data_::NEW_GEOM, this );
}


//-----------------------------------------------------------------------
//   emitEvents_
//-----------------------------------------------------------------------


void StdPointSet::emitEvents_ ( int events )
{
  if ( events & Data_::NEW_SIZE )
  {
    newSizeEvent.emit ( data_->pointCount, *this );
  }

  if ( events & Data_::NEW_GEOM )
  {
    newGeomEvent.emit ( *this );
  }

  if ( events )
  {
    changeEvent.emit ( *this );
  }
}


JIVE_END_PACKAGE( util )
