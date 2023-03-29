
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


#include <jem/base/Error.h>
#include <jem/base/ClassTemplate.h>
#include <jem/gl/utilities.h>
#include <jem/gl/Actions.h>
#include <jem/gl/Camera.h>
#include <jem/gl/CameraPath.h>
#include <jem/gl/Compound.h>
#include <jem/gl/Pivotable.h>
#include <jem/gl/RenderContext.h>
#include <jem/gl/Transformable.h>


JEM_DEFINE_CLASS( jem::gl::CameraPath );


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class CameraPath
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


CameraPath::CameraPath ( const Ref<Camera>& cam ) :

  camera_ ( cam )

{
  JEM_PRECHECK ( cam );

  time_[0]  = time_[1] = time_[2] = 0.0;
  duration_ = 0.0;
  started_  = false;
}


CameraPath::~CameraPath ()
{}


//-----------------------------------------------------------------------
//   takeGLAction
//-----------------------------------------------------------------------


bool CameraPath::takeGLAction

  ( RenderContext&     rtx,
    int                action,
    const Properties&  params )

{
  bool  retval = false;

  if ( ! started_ )
  {
    retval = Super::takeGLAction ( rtx, action, params );
  }
  else
  {
    // Make sure that this object is not deleted until this
    // function has completed.

    Ref<Self>  self = this;
    bool       cont = true;

    switch ( action )
    {
    case GLActions::REDRAW:
    case GLActions::REDRAW_FAST:

      cont = advance_ ( rtx );

      break;

    case GLActions::KEY_EVENT:
    case GLActions::MOUSE_EVENT:
    case GLActions::MOTION_EVENT:
    case GLActions::CLICK2_EVENT:

      cont = false;

      break;
    }

    retval = Super::takeGLAction ( rtx, action, params );

    if ( ! cont )
    {
      finish ();
    }
  }

  return retval;
}


//-----------------------------------------------------------------------
//   setPath
//-----------------------------------------------------------------------


void CameraPath::setPath ( const Box& end )
{
  JEM_PRECHECK ( ! end.isEmpty() );

  Box       start, mid;
  Vertex3d  a, b, c, d;


  if ( started_ )
  {
    finish ();
  }

  camera_->getVisibleBox ( start );

  start.getCorners ( a, b );
  end  .getCorners ( c, d );

  for ( int i = 0; i < 3; i++ )
  {
    a[i] = 0.5 * (a[i] + c[i]);
    b[i] = 0.5 * (b[i] + d[i]);
  }

  mid.setCorners ( a, b );

  path_[0] = start;
  path_[1] = mid;
  path_[2] = end;

  time_[0] = 0.0;
  time_[1] = 0.5;
  time_[2] = 1.0;
}


void CameraPath::setPath

  ( const Box&  mid,
    const Box&  end )

{
  JEM_PRECHECK ( ! mid.isEmpty() );
  JEM_PRECHECK ( ! end.isEmpty() );

  Box     start;
  double  d1, d2;


  if ( started_ )
  {
    finish ();
  }

  camera_->getVisibleBox ( start );

  d1 =      distance ( start, mid );
  d2 = d1 + distance ( mid,   end );

  path_[0] = start;
  path_[1] = mid;
  path_[2] = end;

  time_[0] = 0.0;
  time_[2] = 1.0;

  if ( d2 > 0.0 )
  {
    time_[1] = d1 / d2;
  }
  else
  {
    time_[1] = 0.5;
  }
}


//-----------------------------------------------------------------------
//   finish
//-----------------------------------------------------------------------


void CameraPath::finish ()
{
  if ( started_ )
  {
    Pivotable*  piv = dynamic_cast<Pivotable*> ( getParent() );

    if ( piv )
    {
      Ref<GfxNode>  child = child_;

      this->setChild_ ( nullptr );
      piv ->setChild  ( child );
    }
    else
    {
      Compound*  cmp = dynamic_cast<Compound*> ( getParent() );

      if ( cmp )
      {
        cmp->eraseChild ( this );
      }
      else
      {
        throw Error (
          JEM_FUNC,
          "failed to remove a CameraPath from the scene graph"
        );
      }
    }

    setChild_ ( nullptr );

    started_ = false;
  }
}


//-----------------------------------------------------------------------
//   animate()
//-----------------------------------------------------------------------


void CameraPath::animate ( double time )
{
  JEM_PRECHECK ( time >= 0.0 );

  Transformable*  tr   = nullptr;
  GfxNode*        node = nullptr;

  if ( started_ )
  {
    finish ();
  }

  // Terminate any active CameraPath object.

  node = camera_->getParent ();

  while ( node )
  {
    CameraPath*  path = dynamic_cast<Self*> ( node );

    if ( path )
    {
      path->finish ();

      break;
    }

    node = node->getParent ();
  }

  // Insert this CameraPath into the scene graph.

  node = camera_->getParent ();

  while ( node )
  {
    Pivotable*  piv = dynamic_cast<Pivotable*> ( node );

    if ( piv )
    {
      Ref<GfxNode>  child = piv->getChild ();

      piv ->setChild  ( this );
      this->setChild_ ( child );

      break;
    }

    Compound*  cmp = dynamic_cast<Compound*> ( node );

    if ( cmp )
    {
      cmp->addChild ( this );

      break;
    }

    node = node->getParent ();
  }

  if ( ! node )
  {
    return;
  }

  tr = toTransformable ( camera_ );

  if ( tr )
  {
    trans0_ = tr->getTransform ();
  }
  else
  {
    trans0_.setIdentity ();
  }

  start_    = Time::now ();
  started_  = true;
  duration_ = time;
}


void CameraPath::animate

  ( const Ref<Camera>&  cam,
    const Box&          end,
    double              time )

{
  Ref<Self>  cp = newInstance<Self> ( cam );

  cp->setPath ( end );
  cp->animate ( time );
}


void CameraPath::animate

  ( const Ref<Camera>&  cam,
    const Box&          mid,
    const Box&          end,
    double              time )

{
  Ref<Self>  cp = newInstance<Self> ( cam );

  cp->setPath ( mid, end );
  cp->animate ( time );
}


//-----------------------------------------------------------------------
//   advance_
//-----------------------------------------------------------------------


bool CameraPath::advance_ ( RenderContext& rtx )
{
  Transformable*  tr = toTransformable ( camera_ );
  Time            tm = Time::now() - start_;
  double          t  = tm.toDouble ();


  if ( tr )
  {
    // Restore the original camera transform because the boxes have
    // been specified in the original coordinate system.

    tr->setTransform ( trans0_ );
  }

  t = max ( t, 0.0 );

  if ( t >= duration_ )
  {
    camera_->lookAtBox    ( path_[2] );
    camera_->updateZRange ();
    rtx     .postRedraw   ( camera_ );

    return false;
  }
  else
  {
    Box        box;
    Vertex3d   a, b, c, d;
    double     u, v, dt;
    int        i, j;

    if ( duration_ > 0.0 )
    {
      t /= duration_;
    }

    // Make a nice, smooth velocity curve.

    t = 0.5 * (1.0 - std::cos( t * 180.0 * RADS_PER_DEGREE ));

    if ( t < time_[1] )
    {
      i = 0;
    }
    else
    {
      i = 1;
    }

    j  = i + 1;
    dt = time_[j] - time_[i];

    if ( dt > 0.0 )
    {
      u = (time_[j] - t) / dt;
    }
    else
    {
      u = 0.0;
    }

    v = 1.0 - u;

    path_[i].getCorners ( a, b );
    path_[j].getCorners ( c, d );

    for ( i = 0; i < 3; i++ )
    {
      a[i] = u * a[i] + v * c[i];
      b[i] = u * b[i] + v * d[i];
    }

    box.setCorners ( a, b );

    camera_->lookAtBox      ( box );
    camera_->updateZRange   ();
    rtx     .postFastRedraw ( camera_ );

    return true;
  }
}


JEM_END_PACKAGE( gl )

