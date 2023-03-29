
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

#ifndef JEM_GL_SHADERPROG_H
#define JEM_GL_SHADERPROG_H

#include <jem/base/NamedObject.h>
#include <jem/util/Properties.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/GLFunctions3.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class ShaderProg
//-----------------------------------------------------------------------


class ShaderProg : public NamedObject
{
 public:

  typedef NamedObject     Super;
  typedef ShaderProg      Self;

  explicit                ShaderProg

    ( const String&         name);

  virtual String          getContext      () const override;

  void                    setSource

    ( const String&         vertProg,
      const String&         fragProg,
      const String&         geomProg = "" );

  void                    compileSource

    ( GLContext&            gtx,
      const String&         vertProg,
      const String&         fragProg,
      const String&         geomProg = "" );

  void                    compileFiles

    ( GLContext&            gtx,
      const String&         vertPath,
      const String&         fragPath,
      const String&         geomPath = "" );

  bool                    bind

    ( GLContext&            gtx );

  void                    unbind

    ( GLContext&            gtx );

  GLuint                  getProgramID

    ( GLContext&            gtx );

  GLint                   getUniformIdx

    ( GLContext&            gtx,
      const String&         name );

  template <class T>
    inline GLint          setUniform

    ( GLContext&            gtx,
      const String&         name,
      const T&              value );

  template <class T>
    inline GLint          setUniform2

    ( GLContext&            gtx,
      const String&         name,
      const T&              value );

  template <class T>
    inline GLint          setUniform2

    ( GLContext&            gtx,
      const String&         name,
      const T*              value );

  template <class T>
    inline GLint          setUniform3

    ( GLContext&            gtx,
      const String&         name,
      const T&              value );

  template <class T>
    inline GLint          setUniform3

    ( GLContext&            gtx,
      const String&         name,
      const T*              value );

  template <class T>
    inline GLint          setUniform4

    ( GLContext&            gtx,
      const String&         name,
      const T&              value );

  template <class T>
    inline GLint          setUniform4

    ( GLContext&            gtx,
      const String&         name,
      const T*              value );

  template <class T>
    inline GLint          setUniMatrix3

    ( GLContext&            gtx,
      const String&         name,
      const T*              value );

  template <class T>
    inline GLint          setUniMatrix4

    ( GLContext&            gtx,
      const String&         name,
      const T*              value );


 protected:

  virtual                ~ShaderProg      ();


 private:

  void                    linkProgram_

    ( GLContext&            gtx );

  GLint                   getUniformVar_

    ( GLContext&            gtx,
      const String&         name );

  GLuint                  compileShader_

    ( GLContext&            gtx,
      const String&         prog,
      int                   type );

  String                  readShaderFile_

    ( const String&         fileName );


 private:

  static const int        VALID_;
  static const int        UNKNOWN_;
  static const int        INVALID_;

  String                  vertProg_;
  String                  fragProg_;
  String                  geomProg_;

  jem::util::Properties   uniLocs_;

  int                     gtxID_;
  GLuint                  progID_;
  int                     status_;

};


//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   setUniform
//-----------------------------------------------------------------------


template <class T>

  inline GLint ShaderProg::setUniform

  ( GLContext&     gtx,
    const String&  name,
    const T&       value )

{
  GLint  idx = getUniformIdx ( gtx, name );

  gtx.getFunctions<GLFuncs3Core> ()
    .glUniform ( idx, value );

  return idx;
}


//-----------------------------------------------------------------------
//   setUniform2
//-----------------------------------------------------------------------


template <class T>

  inline GLint ShaderProg::setUniform2

  ( GLContext&     gtx,
    const String&  name,
    const T&       value )

{
  GLint  idx = getUniformIdx ( gtx, name );

  gtx.getFunctions<GLFuncs3Core> ()
    .glUniform2 ( idx, 1, value.addr() );

  return idx;
}


template <class T>

  inline GLint ShaderProg::setUniform2

  ( GLContext&     gtx,
    const String&  name,
    const T*       value )

{
  GLint  idx = getUniformIdx ( gtx, name );

  gtx.getFunctions<GLFuncs3Core> ()
    .glUniform2 ( idx, 1, value );

  return idx;
}


//-----------------------------------------------------------------------
//   setUniform3
//-----------------------------------------------------------------------


template <class T>

  inline GLint ShaderProg::setUniform3

  ( GLContext&     gtx,
    const String&  name,
    const T&       value )

{
  GLint  idx = getUniformIdx ( gtx, name );

  gtx.getFunctions<GLFuncs3Core> ()
    .glUniform3 ( idx, 1, value.addr() );

  return idx;
}


template <class T>

  inline GLint ShaderProg::setUniform3

  ( GLContext&     gtx,
    const String&  name,
    const T*       value )

{
  GLint  idx = getUniformIdx ( gtx, name );

  gtx.getFunctions<GLFuncs3Core> ()
    .glUniform3 ( idx, 1, value );

  return idx;
}


//-----------------------------------------------------------------------
//   setUniform4
//-----------------------------------------------------------------------


template <class T>

  inline GLint ShaderProg::setUniform4

  ( GLContext&     gtx,
    const String&  name,
    const T&       value )

{
  GLint  idx = getUniformIdx ( gtx, name );

  gtx.getFunctions<GLFuncs3Core> ()
    .glUniform4 ( idx, 1, value.addr() );

  return idx;
}


template <class T>

  inline GLint ShaderProg::setUniform4

  ( GLContext&     gtx,
    const String&  name,
    const T*       value )

{
  GLint  idx = getUniformIdx ( gtx, name );

  gtx.getFunctions<GLFuncs3Core> ()
    .glUniform4 ( idx, 1, value );

  return idx;
}


//-----------------------------------------------------------------------
//   setUniMatrix3
//-----------------------------------------------------------------------


template <class T>

  inline GLint ShaderProg::setUniMatrix3

  ( GLContext&     gtx,
    const String&  name,
    const T*       value )

{
  GLint  idx = getUniformIdx ( gtx, name );

  gtx.getFunctions<GLFuncs3Core> ()
    .glUniformMatrix3 ( idx, 1, GL_FALSE, value );

  return idx;
}


//-----------------------------------------------------------------------
//   setUniMatrix4
//-----------------------------------------------------------------------


template <class T>

  inline GLint ShaderProg::setUniMatrix4

  ( GLContext&     gtx,
    const String&  name,
    const T*       value )

{
  GLint  idx = getUniformIdx ( gtx, name );

  gtx.getFunctions<GLFuncs3Core> ()
    .glUniformMatrix4 ( idx, 1, GL_FALSE, value );

  return idx;
}


JEM_END_PACKAGE( gl )

#endif
