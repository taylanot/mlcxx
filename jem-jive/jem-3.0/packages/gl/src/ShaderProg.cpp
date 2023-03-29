
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


#include <jem/base/CString.h>
#include <jem/base/StringBuffer.h>
#include <jem/io/File.h>
#include <jem/io/FileReader.h>
#include <jem/io/BufferedReader.h>
#include <jem/gl/ShaderProg.h>
#include <jem/gl/GLException.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class ShaderProg
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const int  ShaderProg::VALID_   = 1;
const int  ShaderProg::UNKNOWN_ = 2;
const int  ShaderProg::INVALID_ = 3;


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ShaderProg::ShaderProg ( const String& name ) :

  Super ( name )

{
  gtxID_  = 0;
  progID_ = 0;
  status_ = INVALID_;
}


ShaderProg::~ShaderProg ()
{}


//-----------------------------------------------------------------------
//   getContext
//-----------------------------------------------------------------------


String ShaderProg::getContext () const
{
  return makeContext ( "OpenGL shader program", myName_ );
}


//-----------------------------------------------------------------------
//   setSource
//-----------------------------------------------------------------------


void ShaderProg::setSource

  ( const String&  vertProg,
    const String&  fragProg,
    const String&  geomProg )

{
  vertProg_ = vertProg;
  fragProg_ = fragProg;
  geomProg_ = geomProg;

  if ( vertProg_.size() +
       fragProg_.size() +
       geomProg_.size() )
  {
    status_ = UNKNOWN_;
  }
  else
  {
    status_ = INVALID_;
  }
}


//-----------------------------------------------------------------------
//   compileSource
//-----------------------------------------------------------------------


void ShaderProg::compileSource

  ( GLContext&     gtx,
    const String&  vertProg,
    const String&  fragProg,
    const String&  geomProg )

{
  GLFuncs3Core&  funcs    = gtx.getFunctions<GLFuncs3Core> ();
  GLuint         progID   = getProgramID   ( gtx );

  String         vertCode = vertProg.stripWhite ();
  String         fragCode = fragProg.stripWhite ();
  String         geomCode = geomProg.stripWhite ();

  String         context  = getContext          ();


  status_ = INVALID_;

  try
  {
    if ( vertCode.size() )
    {
      GLuint  iobj = compileShader_ ( gtx, vertCode,
                                      GL_VERTEX_SHADER );

      funcs.glAttachShader ( progID, iobj );
      funcs.glDeleteShader ( iobj );
      checkGLError         ( gtx, context );
    }

    if ( fragCode.size() )
    {
      GLuint  iobj = compileShader_ ( gtx, fragCode,
                                      GL_FRAGMENT_SHADER );

      funcs.glAttachShader ( progID, iobj );
      funcs.glDeleteShader ( iobj );
      checkGLError         ( gtx, context );
    }

    if ( geomCode.size() )
    {
      GLuint  iobj = compileShader_ ( gtx, geomCode,
                                      GL_GEOMETRY_SHADER );

      funcs.glAttachShader ( progID, iobj );
      funcs.glDeleteShader ( iobj );
      checkGLError         ( gtx, context );
    }

    linkProgram_ ( gtx );
  }
  catch ( GLException& ex )
  {
    ex.setContext ( context );

    throw ex;
  }

  vertProg_ = vertProg;
  fragProg_ = fragProg;
  geomProg_ = geomProg;
  status_   = VALID_;
}


//-----------------------------------------------------------------------
//   compileFiles
//-----------------------------------------------------------------------


void ShaderProg::compileFiles

  ( GLContext&     gtx,
    const String&  vertPath,
    const String&  fragPath,
    const String&  geomPath )

{
  String  vertName = vertPath.stripWhite ();
  String  fragName = fragPath.stripWhite ();
  String  geomName = geomPath.stripWhite ();

  String  vertProg;
  String  fragProg;
  String  geomProg;


  if ( vertName.size() )
  {
    vertProg = readShaderFile_ ( vertName );
  }

  if ( fragName.size() )
  {
    fragProg = readShaderFile_ ( fragName );
  }

  if ( geomName.size() )
  {
    geomProg = readShaderFile_ ( geomName );
  }

  compileSource ( gtx, vertProg, fragProg, geomProg );
}


//-----------------------------------------------------------------------
//   bind
//-----------------------------------------------------------------------


bool ShaderProg::bind ( GLContext& gtx )
{
  bool  result = false;

  if ( status_ != INVALID_ )
  {
    GLFuncs3Core*  funcs = gtx.findFunctions<GLFuncs3Core> ();

    if ( funcs )
    {
      if ( status_ != VALID_ )
      {
        compileSource ( gtx, vertProg_, fragProg_, geomProg_ );
      }

      if ( status_ == VALID_ )
      {
        funcs->glUseProgram ( getProgramID( gtx ) );

        result = true;
      }
    }
  }

  return result;
}


//-----------------------------------------------------------------------
//   unbind
//-----------------------------------------------------------------------


void ShaderProg::unbind ( GLContext& gtx )
{
  GLFuncs3Core*  funcs = gtx.findFunctions<GLFuncs3Core> ();

  if ( funcs )
  {
    funcs->glUseProgram ( 0 );
  }
}


//-----------------------------------------------------------------------
//   getProgramID
//-----------------------------------------------------------------------


GLuint ShaderProg::getProgramID ( GLContext& gtx )
{
  GLuint  id = 0;

  if ( gtx.hasID( gtxID_ ) )
  {
    id = progID_;
  }
  else
  {
    if ( ! gtx.findGLObject( id, this ) )
    {
      id = gtx.newGLObject ( GLObject::PROGRAM, this, this );
    }

    gtxID_  = gtx.getID ();
    progID_ = id;

    if ( vertProg_.size() +
         fragProg_.size() +
         geomProg_.size() )
    {
      status_ = UNKNOWN_;
    }
    else
    {
      status_ = INVALID_;
    }

    uniLocs_.clear ();
  }

  return id;
}


//-----------------------------------------------------------------------
//   getUniformIdx
//-----------------------------------------------------------------------


GLint ShaderProg::getUniformIdx

  ( GLContext&     gtx,
    const String&  name )

{
  GLint  iloc = 0;

  if ( status_ == VALID_ )
  {
    if ( ! uniLocs_.find( iloc, name ) )
    {
      iloc = getUniformVar_ ( gtx, name );

      uniLocs_.set ( name, iloc );
    }
  }

  return iloc;
}


//-----------------------------------------------------------------------
//   linkProgram_
//-----------------------------------------------------------------------


void ShaderProg::linkProgram_ ( GLContext& gtx )
{
  GLFuncs3Core&  funcs  = gtx.getFunctions<GLFuncs3Core> ();
  GLuint         progID = getProgramID ( gtx );
  GLsizei        len    = 0;

  GLchar         buffer[1024];


  funcs.glLinkProgram ( progID );

  if ( funcs.glGetError() != GL_NO_ERROR )
  {
    throw GLException (
      getContext (),
      "failed to link the shader program"
    );
  }

  funcs.glGetProgramInfoLog ( progID, (GLsizei) sizeof(buffer),
                              &len, buffer );

  if ( len )
  {
    throw GLException (
      getContext (),
      String::format (
        "failed to link the shader program: %s",
        buffer
      )
    );
  }
}


//-----------------------------------------------------------------------
//   getUniformVar_
//-----------------------------------------------------------------------


GLint ShaderProg::getUniformVar_

  ( GLContext&     gtx,
    const String&  name )

{
  GLFuncs3Core&  funcs  = gtx.getFunctions<GLFuncs3Core> ();
  GLuint         progID = getProgramID ( gtx );
  GLint          iloc;

  iloc = funcs.glGetUniformLocation ( progID,
                                      makeCString( name ) );

  if ( funcs.glGetError() != GL_NO_ERROR )
  {
    throw GLException (
      getContext (),
      String::format (
        "could not get the location of the uniform shader "
        "variable `%s'",
        name
      )
    );
  }

  return iloc;
}


//-----------------------------------------------------------------------
//   compileShader_
//-----------------------------------------------------------------------


GLuint ShaderProg::compileShader_

  ( GLContext&     gtx,
    const String&  prog,
    int            type )

{
  GLFuncs3Core&  funcs   = gtx.getFunctions<GLFuncs3Core> ();
  String         context = getContext        ();
  const char*    source  =         prog.addr ();
  GLint          length  = (GLint) prog.size ();
  GLint          result  = 0;
  GLuint         handle  = 0;


  handle = funcs.glCreateShader ( (GLenum) type );

  checkGLError ( gtx, context );

  if ( ! handle )
  {
    throw GLException (
      context,
      "failed to create an OpenGL shader object"
    );
  }

  funcs.glShaderSource  ( handle, (GLsizei) 1, &source, &length );
  checkGLError          ( gtx, context );
  funcs.glCompileShader ( handle );
  funcs.glGetShader     ( handle, GL_COMPILE_STATUS, &result );

  if ( ! result )
  {
    char*    errTxt = nullptr;
    String   msg;
    String   kind;
    GLint    errLen0;
    GLsizei  errLen1;

    funcs.glGetShader ( handle, GL_INFO_LOG_LENGTH, &errLen0 );

    errTxt = new char[errLen0];

    try
    {
      funcs.glGetShaderInfoLog ( handle, (GLsizei) errLen0,
                                 &errLen1, errTxt );
    }
    catch ( ... )
    {
      delete errTxt;
      throw;
    }

    if      ( type == GL_VERTEX_SHADER )
    {
      kind = "the vertex shader";
    }
    else if ( type == GL_FRAGMENT_SHADER )
    {
      kind = "the fragment shader";
    }
    else if ( type == GL_GEOMETRY_SHADER )
    {
      kind = "the geometry shader";
    }
    else
    {
      kind = "a shader of an invalid type";
    }

    if ( errLen1 > 0 )
    {
      msg =

        String::format (
          "failed to compile %s: %s",
          kind,
          errTxt
        );
    }
    else
    {
      msg = "failed to compile " + kind;
    }

    delete errTxt;

    throw GLException ( context, msg );
  }

  return handle;
}


//-----------------------------------------------------------------------
//   readShaderFile_
//-----------------------------------------------------------------------


String ShaderProg::readShaderFile_ ( const String& path )
{
  using jem::newInstance;
  using jem::StringBuffer;
  using jem::io::File;
  using jem::io::Reader;
  using jem::io::FileReader;
  using jem::io::BufferedReader;

  Ref<Reader>   input;
  StringBuffer  source;


  if ( ! File::exists( path ) )
  {
    throw GLException (
      getContext (),
      String::format (
        "the shader source file `%s' does not exist",
        path
      )
    );
  }

  try
  {
    input = newInstance<BufferedReader> (
      newInstance<FileReader> ( path )
    );
  }
  catch ( const Exception& ex )
  {
    throw GLException (
      getContext (),
      String::format (
        "error opening the shader source file `%s': %s",
        path,
        ex.what ()
      )
    );
  }

  while ( true )
  {
    int  c = input->read ();

    if ( c < 0 )
    {
      break;
    }

    source += (char) c;
  }

  input->close ();

  return source.toString().stripWhite ();
}


JEM_END_PACKAGE( gl )
