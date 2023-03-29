
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


#include <jem/base/utilities.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/tuple/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/util/Flex.h>
#include <jem/util/List.h>
#include <jem/util/FastAllocator.h>
#include <jem/gl/render.h>
#include <jem/gl/utilities.h>
#include <jem/gl/Vertex.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions1.h>
#include <jem/gl/ShaderDraw.h>
#include <jem/gl/Orb.h>


JEM_DEFINE_CLASS( jem::gl::Orb );


JEM_BEGIN_PACKAGE( gl )


using jem::util::List;
using jem::util::Flex;
using jem::util::FastAllocator;

typedef Tuple<idx_t,3>  Tuple3i;


//=======================================================================
//   class Orb::Edge_
//=======================================================================


class Orb::Edge_ : public Collectable
{
 public:

  typedef List
    < idx_t,
      FastAllocator>      IList;
  typedef
    IList::Iterator       Iterator;


  inline                  Edge_     ();

  inline idx_t            size      () const;
  inline Iterator         begin     ();
  inline Iterator         end       ();

  inline void             pushBack

    ( idx_t                 index );

  inline void             insert

    ( Iterator              pos,
      idx_t                 index );


 protected:

  virtual                ~Edge_     ();


 private:

  IList                   list_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline Orb::Edge_::Edge_ ()
{}


Orb::Edge_::~Edge_ ()
{}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t Orb::Edge_::size () const
{
  return list_.size ();
}


//-----------------------------------------------------------------------
//   begin
//-----------------------------------------------------------------------


inline Orb::Edge_::Iterator Orb::Edge_::begin ()
{
  return list_.begin ();
}


//-----------------------------------------------------------------------
//   end
//-----------------------------------------------------------------------


inline Orb::Edge_::Iterator Orb::Edge_::end ()
{
  return list_.end ();
}


//-----------------------------------------------------------------------
//   pushBack
//-----------------------------------------------------------------------


inline void Orb::Edge_::pushBack ( idx_t index )
{
  JEM_ASSERT ( index >= 0 );

  list_.pushBack ( index );
};


//-----------------------------------------------------------------------
//   insert
//-----------------------------------------------------------------------


inline void Orb::Edge_::insert

  ( Iterator  pos,
    idx_t     index )

{
  list_.insert ( pos, index );
};


//=======================================================================
//   class Orb::Patch_
//=======================================================================


class Orb::Patch_
{
 public:

  typedef
    Edge_::Iterator       EdgeIterator;

  static const bool       CLOCKWISE = true;


  void                    init

    ( const Tuple3i&        triangle,
      bool                  clockwise );

  void                    refine

    ( Flex<Vertex3f>&       vertices,
      Flex<Tuple3i>&        triangles,
      idx_t                 depth,
      bool                  clockwise );


 public:

  Ref<Edge_>              e01;
  Ref<Edge_>              e12;
  Ref<Edge_>              e20;

  EdgeIterator            start01;
  EdgeIterator            end01;
  EdgeIterator            start12;
  EdgeIterator            end12;
  EdgeIterator            start20;
  EdgeIterator            end20;


 private:

  static EdgeIterator     refineEdge_

    ( idx_t&                iv0,
      idx_t&                iv1,
      Edge_&                edge,
      EdgeIterator          start,
      EdgeIterator          end,
      Flex<Vertex3f>&       vertices );

};


//-----------------------------------------------------------------------
//   init
//----------------------------------------------------------------------


void Orb::Patch_::init

  ( const Tuple3i&  triangle,
    bool            clockwise )

{
  if ( e01->size() == 0 )
  {
    if ( clockwise )
    {
      e01->pushBack ( triangle[1] );
      e01->pushBack ( triangle[0] );
    }
    else
    {
      e01->pushBack ( triangle[0] );
      e01->pushBack ( triangle[1] );
    }
  }
  if ( e12->size() == 0 )
  {
    if ( clockwise )
    {
      e12->pushBack ( triangle[0] );
      e12->pushBack ( triangle[2] );
    }
    else
    {
      e12->pushBack ( triangle[1] );
      e12->pushBack ( triangle[2] );
    }
  }
  if ( e20->size() == 0 )
  {
    if ( clockwise )
    {
      e20->pushBack ( triangle[2] );
      e20->pushBack ( triangle[1] );
    }
    else
    {
      e20->pushBack ( triangle[2] );
      e20->pushBack ( triangle[0] );
    }
  }

  start01 = e01->begin ();
  end01   = e01->end   ();
  start12 = e12->begin ();
  end12   = e12->end   ();
  start20 = e20->begin ();
  end20   = e20->end   ();
}


//-----------------------------------------------------------------------
//   refine
//-----------------------------------------------------------------------


void Orb::Patch_::refine

  ( Flex<Vertex3f>&  vertices,
    Flex<Tuple3i>&   triangles,
    idx_t            depth,
    bool             clockwise )

{
  if ( depth == 0 )
  {
    return;
  }

  EdgeIterator   iter3, iter4, iter5;

  idx_t          iv0, iv1, iv2;
  idx_t          iv3, iv4, iv5;

  Patch_         p0, p1, p2, p3;
  Tuple3i        ts;


  triangles.popBack ();

  iter3 = refineEdge_ ( iv0, iv3, *e01, start01, end01, vertices );
  iter4 = refineEdge_ ( iv1, iv4, *e12,	start12, end12, vertices );
  iter5 = refineEdge_ ( iv2, iv5, *e20, start20, end20, vertices );

  if ( clockwise )
  {
    ts[0] = iv3;
    ts[1] = iv0;
    ts[2] = iv5;
  }
  else
  {
    ts[0] = iv0;
    ts[1] = iv3;
    ts[2] = iv5;
  }

  triangles.pushBack ( ts );

  p0.e01 = e01;
  p0.e12 = newInstance<Edge_> ();
  p0.e20 = e20;

  p0.e12->pushBack ( iv3 );
  p0.e12->pushBack ( iv5 );

  p0.start01 = start01;
  p0.end01   = iter3;
  p0.start12 = p0.e12->begin ();
  p0.end12   = p0.e12->end   ();
  p0.start20 = iter5;
  p0.end20   = end20;

  ++p0.end01;

  p0.refine ( vertices, triangles, depth - 1, clockwise );

  if ( clockwise )
  {
    ts[0] = iv1;
    ts[1] = iv3;
    ts[2] = iv4;
  }
  else
  {
    ts[0] = iv3;
    ts[1] = iv1;
    ts[2] = iv4;
  }

  triangles.pushBack ( ts );

  p1.e01 = e01;
  p1.e12 = e12;
  p1.e20 = newInstance<Edge_> ();

  p1.e20->pushBack ( iv4 );
  p1.e20->pushBack ( iv3 );

  p1.start01 = iter3;
  p1.end01   = end01;
  p1.start12 = start12;
  p1.end12   = iter4;
  p1.start20 = p1.e20->begin ();
  p1.end20   = p1.e20->end   ();

  ++p1.end12;

  p1.refine ( vertices, triangles, depth - 1, clockwise );

  if ( clockwise )
  {
    ts[0] = iv4;
    ts[1] = iv5;
    ts[2] = iv2;
  }
  else
  {
    ts[0] = iv5;
    ts[1] = iv4;
    ts[2] = iv2;
  }

  triangles.pushBack ( ts );

  p2.e01 = newInstance<Edge_> ();
  p2.e12 = e12;
  p2.e20 = e20;

  p2.e01->pushBack ( iv5 );
  p2.e01->pushBack ( iv4 );

  p2.start01 = p2.e01->begin ();
  p2.end01   = p2.e01->end   ();
  p2.start12 = iter4;
  p2.end12   = end12;
  p2.start20 = start20;
  p2.end20   = iter5;

  ++p2.end20;

  p2.refine ( vertices, triangles, depth - 1, clockwise );

  if ( clockwise )
  {
    ts[0] = iv3;
    ts[1] = iv5;
    ts[2] = iv4;
  }
  else
  {
    ts[0] = iv5;
    ts[1] = iv3;
    ts[2] = iv4;
  }

  triangles.pushBack ( ts );

  p3.e01 = p0.e12;
  p3.e12 = p2.e01;
  p3.e20 = p1.e20;

  p3.start01 = p0.e12->begin ();
  p3.end01   = p0.e12->end   ();
  p3.start12 = p2.e01->begin ();
  p3.end12   = p2.e01->end   ();
  p3.start20 = p1.e20->begin ();
  p3.end20   = p1.e20->end   ();

  p3.refine ( vertices, triangles, depth - 1, ! clockwise );

  return;
}


//-----------------------------------------------------------------------
//   refineEdge_
//-----------------------------------------------------------------------


Orb::Edge_::Iterator Orb::Patch_::refineEdge_

  ( idx_t&           iv0,
    idx_t&           iv1,
    Edge_&           edge,
    EdgeIterator     start,
    EdgeIterator     end,
    Flex<Vertex3f>&  vertices )

{
  EdgeIterator  iter = end;

  --iter;

  iv1  = *iter;
  iter =  start;
  iv0  = *iter;

  Vertex3f  v0 = vertices[iv0];
  Vertex3f  v1 = vertices[iv1];

  idx_t     nv = (idx_t) distance ( start, end );

  if ( nv > 2 )
  {
    for ( idx_t i = (nv - 1) / 2; i > 0; i-- )
    {
      ++iter;
    }

    iv1 = *iter;
  }
  else
  {
    Vertex3f  v2 = v0 + v1;

    iv1 = (idx_t) vertices.size ();

    normalize         ( v2 );
    vertices.pushBack ( v2 );
    edge.insert       ( ++iter, iv1 );

    --iter;
  }

  return iter;
}


//=======================================================================
//   class Orb
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  Orb::PART0       =  1 << 0;
const int  Orb::PART1       =  1 << 1;
const int  Orb::PART2       =  1 << 2;
const int  Orb::PART3       =  1 << 3;
const int  Orb::PART4       =  1 << 4;
const int  Orb::PART5       =  1 << 5;
const int  Orb::PART6       =  1 << 6;
const int  Orb::PART7       =  1 << 7;

const int  Orb::DOME0       =  PART0 | PART1 | PART2 | PART3;
const int  Orb::DOME1       =  PART4 | PART5 | PART6 | PART7;
const int  Orb::DOME2       =  PART0 | PART1 | PART4 | PART5;
const int  Orb::DOME3       =  PART2 | PART3 | PART6 | PART7;
const int  Orb::DOME4       =  PART0 | PART3 | PART4 | PART7;
const int  Orb::DOME5       =  PART1 | PART2 | PART5 | PART6;

const int  Orb::EVEN_PARTS  =  PART1 | PART3 | PART4 | PART6;
const int  Orb::ODD_PARTS   =  PART0 | PART2 | PART5 | PART7;

const int  Orb::ALL_PARTS   = (1 << 8) - 1;
const int  Orb::MAX_QUALITY =  8;

#ifndef _MSC_EXTENSIONS
const int  Orb::CACHE_SIZE_;
#endif


//-----------------------------------------------------------------------
//   constructor and destructor
//-----------------------------------------------------------------------


Orb::Orb ( int quality )
{
  for ( int i = 0; i <= CACHE_SIZE_; i++ )
  {
    mcache_[i].meshed  = false;
    mcache_[i].quality = -1;
  }

  current_ = mcache_;

  setQuality ( quality );
}


Orb::~Orb ()
{}


//-----------------------------------------------------------------------
//   setQuality
//-----------------------------------------------------------------------


void Orb::setQuality ( int quality )
{
  quality = clamped ( quality, MAX_QUALITY );

  if ( current_->quality != quality )
  {
    current_ = mcache_ + min ( quality, CACHE_SIZE_ );

    if ( current_->quality != quality )
    {
      current_->quality = quality;
    }
  }
}


//-----------------------------------------------------------------------
//   changeQuality
//-----------------------------------------------------------------------


int Orb::changeQuality ( int delta )
{
  setQuality ( current_->quality + delta );

  return current_->quality;
}


//-----------------------------------------------------------------------
//   drawOrb
//-----------------------------------------------------------------------


void Orb::drawOrb

  ( GLContext&  gtx,
    GLfloat     radius,
    int         parts ) const

{
  JEM_PRECHECK ( radius > 0.0F );

  Self*  self = const_cast<Self*> ( this );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    self->drawOrbV3_ ( gtx, radius, parts );
  }
  else
  {
    self->drawOrbV1_ ( gtx, radius, parts );
  }
}


void Orb::drawOrb

  ( GLContext&    gtx,
    GLfloat       radius,
    const Color&  color,
    int           parts ) const

{
  JEM_PRECHECK ( radius > 0.0F );

  Self*  self = const_cast<Self*> ( this );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    self->drawOrbV3_ ( gtx, radius, color, parts );
  }
  else
  {
    self->drawOrbV1_ ( gtx, radius, color, parts );
  }
}


void Orb::drawOrb

  ( GLContext&    gtx,
    GLfloat       radius,
    const Color&  color1,
    const Color&  color2,
    int           parts ) const

{
  JEM_PRECHECK ( radius > 0.0F );

  Self*  self = const_cast<Self*> ( this );

  if ( gtx.checkVersion( GLVersion::VERSION_3_3 ) )
  {
    self->drawOrbV3_ ( gtx, radius, color1, color2, parts );
  }
  else
  {
    self->drawOrbV1_ ( gtx, radius, color1, color2, parts );
  }
}


//-----------------------------------------------------------------------
//   drawUnitOrb
//-----------------------------------------------------------------------


void Orb::drawUnitOrb

  ( GLContext&  gtx,
    int         parts ) const

{
  drawOrb ( gtx, 1.0, parts );
}


//-----------------------------------------------------------------------
//   drawOrbV1_
//-----------------------------------------------------------------------


void Orb::drawOrbV1_

  ( GLContext&    gtx,
    GLfloat       radius,
    const Color&  color,
    int           parts )

{
  AttribScope  scope ( gtx, GLContext::COLOR );

  gtx.setColor ( color );
  drawOrbV1_   ( gtx, radius, parts );
}


void Orb::drawOrbV1_

  ( GLContext&    gtx,
    GLfloat       radius,
    const Color&  color1,
    const Color&  color2,
    int           parts )

{
  AttribScope  scope ( gtx, GLContext::COLOR );

  gtx.setColor ( color1 );
  drawOrbV1_   ( gtx, radius, parts & EVEN_PARTS );
  gtx.setColor ( color2 );
  drawOrbV1_   ( gtx, radius, parts & ODD_PARTS );
}


void Orb::drawOrbV1_

  ( GLContext&  gtx,
    GLfloat     radius,
    int         parts )

{
  if ( ! current_->meshed )
  {
    initMesh_ ( *current_, current_->quality );
  }

  const GLfloat*  vx    = current_->vertices .addr ();
  const idx_t*    tr    = current_->triangles.addr ();
  const idx_t     n8    = current_->triangles.size (1) / 8;
  GLFunctions1*   funcs = gtx.findFunctions<GLFunctions1> ();


  if ( ! funcs )
  {
    return;
  }

  funcs->glBegin ( GL_TRIANGLES );

  for ( idx_t p = 0; p < 8; p++ )
  {
    if ( parts & 1 )
    {
      for ( idx_t i = n8; i > 0; i--, tr += 3 )
      {
        idx_t  iv0 = tr[0];
        idx_t  iv1 = tr[1];
        idx_t  iv2 = tr[2];

        funcs->glNormal3 ( vx + iv0 );
        funcs->glVertex  ( vx[iv0 + 0] * radius,
                           vx[iv0 + 1] * radius,
                           vx[iv0 + 2] * radius );

        funcs->glNormal3 ( vx + iv1 );
        funcs->glVertex  ( vx[iv1 + 0] * radius,
                           vx[iv1 + 1] * radius,
                           vx[iv1 + 2] * radius );

        funcs->glNormal3 ( vx + iv2 );
        funcs->glVertex  ( vx[iv2 + 0] * radius,
                           vx[iv2 + 1] * radius,
                           vx[iv2 + 2] * radius );
      }
    }
    else
    {
      tr += (3 * n8);
    }

    parts = parts >> 1;
  }

  funcs->glEnd ();
}


//-----------------------------------------------------------------------
//   drawOrbV3_
//-----------------------------------------------------------------------


void Orb::drawOrbV3_

  ( GLContext&    gtx,
    GLfloat       radius,
    const Color&  color,
    int           parts )

{
  Color  color0 = gtx.getColor ();

  gtx.setColor ( color );
  drawOrbV3_   ( gtx, radius, parts );
  gtx.setColor ( color0 );
}


void Orb::drawOrbV3_

  ( GLContext&    gtx,
    GLfloat       radius,
    const Color&  color1,
    const Color&  color2,
    int           parts )

{
  if ( parts == ALL_PARTS )
  {
    const Vertex3f  origin = Vertex3f ( 0.0F, 0.0F, 0.0F );

    ShaderDraw::drawOrb ( gtx, origin, radius, color1, color2 );
  }
  else
  {
    Color  color0 = gtx.getColor ();

    gtx.setColor ( color1 );

    drawOrbV3_   ( gtx, radius, parts & EVEN_PARTS );
    gtx.setColor ( color2 );
    drawOrbV3_   ( gtx, radius, parts & ODD_PARTS );
    gtx.setColor ( color0 );
  }
}


void Orb::drawOrbV3_

  ( GLContext&  gtx,
    GLfloat     radius,
    int         parts )

{
  if ( parts == ALL_PARTS )
  {
    const Vertex3f  origin = Vertex3f ( 0.0F, 0.0F, 0.0F);

    ShaderDraw::drawOrb ( gtx, origin, radius, gtx.getColor() );

    return;
  }

  if ( ! current_->meshed )
  {
    initMesh_ ( *current_, current_->quality );
  }

  GLfloat*        vertices = nullptr;
  GLfloat*        normals  = nullptr;
  const GLfloat*  vx       = current_->vertices .addr ();
  const idx_t*    tr       = current_->triangles.addr ();
  const idx_t     n8       = current_->triangles.size (1) / 8;
  const size_t    vsize    = 3 * sizeof(GLfloat);
  idx_t           vcount   = 0;


  scratch_.vertices.resize ( 72 * n8 );
  scratch_.normals .resize ( 72 * n8 );

  vertices  = scratch_.vertices.addr ();
  normals   = scratch_.normals .addr ();

  for ( idx_t p = 0; p < 8; p++ )
  {
    if ( parts & 1 )
    {
      for ( idx_t i = n8; i > 0; i--, tr += 3 )
      {
        std::memcpy ( vertices + (vcount + 0), vx + tr[0], vsize );
        std::memcpy ( vertices + (vcount + 3), vx + tr[1], vsize );
        std::memcpy ( vertices + (vcount + 6), vx + tr[2], vsize );

        std::memcpy ( normals  + (vcount + 0), vx + tr[0], vsize );
        std::memcpy ( normals  + (vcount + 3), vx + tr[1], vsize );
        std::memcpy ( normals  + (vcount + 6), vx + tr[2], vsize );

        vcount += 9;
      }
    }
    else
    {
      tr += (3 * n8);
    }

    parts = parts >> 1;
  }

  for ( idx_t i = 0; i < vcount; i++ )
  {
    vertices[i] = vertices[i] * radius;
  }

  ShaderDraw::draw ( gtx, vcount, vertices, normals,
                     gtx.getColor(), ShaderDraw::TRIANGLES );
}


//-----------------------------------------------------------------------
//   initMesh_
//----------------------------------------------------------------------


void Orb::initMesh_

  ( Mesh_&  mesh,
    int     quality )

{
  const bool      CLOCKWISE = Patch_::CLOCKWISE;

  Flex<Vertex3f>  vertices ( 3 );
  Flex<Tuple3i>   triangles;

  Vertex3f        vertex;
  Tuple3i         triang;
  Patch_          patch[8];

  idx_t           oindex[8];
  idx_t           nindex;
  idx_t           nv, nt, n8;


  nt = (8 * (1 << (2 * quality)));

  vertices .reserve ( nt );
  triangles.reserve ( nt );

  patch[0].e01 = newInstance<Edge_> ();
  patch[0].e12 = newInstance<Edge_> ();
  patch[0].e20 = newInstance<Edge_> ();
  patch[1].e01 = newInstance<Edge_> ();
  patch[1].e12 = patch[0].e12;
  patch[1].e20 = newInstance<Edge_> ();
  patch[2].e01 = newInstance<Edge_> ();
  patch[2].e12 = newInstance<Edge_> ();
  patch[2].e20 = patch[1].e20;
  patch[3].e01 = newInstance<Edge_> ();
  patch[3].e12 = patch[2].e12;
  patch[3].e20 = patch[0].e20;
  patch[4].e01 = patch[0].e01;
  patch[4].e12 = newInstance<Edge_> ();
  patch[4].e20 = newInstance<Edge_> ();
  patch[5].e01 = patch[1].e01;
  patch[5].e12 = patch[4].e12;
  patch[5].e20 = newInstance<Edge_> ();
  patch[6].e01 = patch[2].e01;
  patch[6].e12 = newInstance<Edge_> ();
  patch[6].e20 = patch[5].e20;
  patch[7].e01 = patch[3].e01;
  patch[7].e12 = patch[6].e12;
  patch[7].e20 = patch[4].e20;

  vertices[0][0] =  1.0F;
  vertices[0][1] =  0.0F;
  vertices[0][2] =  0.0F;
  vertices[1][0] =  0.0F;
  vertices[1][1] =  1.0F;
  vertices[1][2] =  0.0F;
  vertices[2][0] =  0.0F;
  vertices[2][1] =  0.0F;
  vertices[2][2] = -1.0F;

  triang[0] = 1;
  triang[1] = 0;
  triang[2] = 2;
  oindex[0] = 1;

  triangles.pushBack ( triang );
  patch[0] .init     ( triang, CLOCKWISE );
  patch[0] .refine   ( vertices, triangles, quality, CLOCKWISE );

  nindex    =  vertices.size ();
  oindex[1] =  nindex;
  vertex[0] = -1.0F;
  vertex[1] =  0.0F;
  vertex[2] =  0.0F;
  triang[0] =  nindex;
  triang[1] =  1;
  triang[2] =  2;

  triangles.pushBack ( triang );
  vertices .pushBack ( vertex );
  patch[1] .init     ( triang, ! CLOCKWISE );
  patch[1] .refine   ( vertices, triangles, quality, ! CLOCKWISE );

  nindex    =  vertices.size ();
  oindex[2] =  nindex;
  vertex[0] =  0.0F;
  vertex[1] = -1.0F;
  vertex[2] =  0.0F;
  triang[0] =  nindex;
  triang[1] =  oindex[1];
  triang[2] =  2;

  triangles.pushBack ( triang );
  vertices .pushBack ( vertex );
  patch[2] .init     ( triang, CLOCKWISE );
  patch[2] .refine   ( vertices, triangles, quality, CLOCKWISE );

  triang[0] = 0;
  triang[1] = oindex[2];
  triang[2] = 2;

  triangles.pushBack ( triang );
  patch[3] .init     ( triang, ! CLOCKWISE );
  patch[3] .refine   ( vertices, triangles, quality, ! CLOCKWISE  );

  nindex    = vertices.size();
  vertex[0] = 0.0F;
  vertex[1] = 0.0F;
  vertex[2] = 1.0F;
  triang[0] = 0;
  triang[1] = oindex[0];
  triang[2] = nindex;

  triangles.pushBack ( triang );
  vertices .pushBack ( vertex );
  patch[4] .init     ( triang, ! CLOCKWISE );
  patch[4] .refine   ( vertices, triangles, quality, ! CLOCKWISE );

  triang[0] = oindex[0];
  triang[1] = oindex[1];
  triang[2] = nindex;

  triangles.pushBack ( triang );
  patch[5] .init     ( triang, CLOCKWISE );
  patch[5] .refine   ( vertices, triangles, quality, CLOCKWISE  );

  triang[0] = oindex[1];
  triang[1] = oindex[2];
  triang[2] = nindex;

  triangles.pushBack ( triang );
  patch[6] .init     ( triang, ! CLOCKWISE );
  patch[6] .refine   ( vertices, triangles, quality, ! CLOCKWISE );

  triang[0] = oindex[2];
  triang[1] = 0;
  triang[2] = nindex;

  triangles.pushBack ( triang );
  patch[7] .init     ( triang, CLOCKWISE );
  patch[7] .refine   ( vertices, triangles, quality, CLOCKWISE );

  nv = vertices .size ();
  nt = triangles.size ();
  n8 = nt / 8;

  mesh.vertices .resize ( 3, nv );
  mesh.triangles.resize ( 3, nt );

  // Copy the vertex data.

  for ( idx_t i = 0; i < nv; i++ )
  {
    for ( idx_t j = 0; j < 3; j++ )
    {
      mesh.vertices(j,i) = vertices[i][j];
    }
  }

  // Determine the back-to-front order of the triangles. Note that the
  // triangles are sorted per part of the orb.

  IdxVector      iperm ( nt );
  GLfloatVector  depth ( nt );

  iperm = iarray ( nt );

  for ( idx_t i = 0; i < nt; i++ )
  {
    depth[i] = 0.0F;

    for ( idx_t j = 0; j < 3; j++ )
    {
      idx_t  iv = triangles[i][j];

      depth[i] += vertices[iv][2];
    }
  }

  for ( idx_t i = 0; i < 8; i++ )
  {
    idx_t  j = i * n8;

    sort ( iperm[slice(j,j + n8)], depth );
  }

  // Copy the triangle data.

  for ( idx_t i = 0; i < nt; i++ )
  {
    idx_t  k = iperm[i];

    for ( idx_t j = 0; j < 3; j++ )
    {
      mesh.triangles(j,i) = 3 * triangles[k][j];
    }
  }

  mesh.meshed  = true;
  mesh.quality = quality;
};


JEM_END_PACKAGE( gl )
