
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

#ifndef JEM_IO_OBJECTINPUTSTREAM_H
#define JEM_IO_OBJECTINPUTSTREAM_H

#include <jem/io/ObjectInput.h>
#include <jem/io/InputStream.h>
#include <jem/io/DataStreamOptions.h>
#include <jem/io/ObjectStreamParams.h>


JEM_BEGIN_PACKAGE( io )


class TextOutput;


//-----------------------------------------------------------------------
//   class ObjectInputStream
//-----------------------------------------------------------------------


class ObjectInputStream : public  InputStream,
                          public  ObjectInput,
                          public  DataStreamOptions,
                          private ObjectStreamParams
{
 public:

  JEM_DECLARE_CLASS     ( ObjectInputStream, InputStream );


  explicit                ObjectInputStream

    ( Ref<InputStream>      in );

  virtual void            close             () override;

  virtual idx_t           poll

    ( const Time&           timeout )          override;

  virtual idx_t           read

    ( void*                 buf,
      idx_t                 n )                override;

  virtual bool            decodeBool        () override;
  virtual byte            decodeByte        () override;
  virtual char            decodeChar        () override;
  virtual short           decodeShort       () override;
  virtual int             decodeInt         () override;
  virtual long            decodeLong        () override;
  virtual lint            decodeLint        () override;
  virtual idx_t           decodeIdx         () override;
  virtual float           decodeFloat       () override;
  virtual double          decodeDouble      () override;

  virtual void            decode

    ( bool*                 buf,
      idx_t                 len )              override;

  virtual void            decode

    ( byte*                 buf,
      idx_t                 len )              override;

  virtual void            decode

    ( char*                 buf,
      idx_t                 len )              override;

  virtual void            decode

    ( short*                buf,
      idx_t                 len )              override;

  virtual void            decode

    ( int*                  buf,
      idx_t                 len )              override;

  virtual void            decode

    ( long*                 buf,
      idx_t                 len )              override;

  virtual void            decode

    ( lint*                 buf,
      idx_t                 len )              override;

  virtual void            decode

    ( idx_t*                buf,
      idx_t                 len )              override;

  virtual void            decode

    ( float*                buf,
      idx_t                 len )              override;

  virtual void            decode

    ( double*               buf,
      idx_t                 len )              override;

  virtual Ref<Object>     decodeObject      () override;

  virtual void            decodeReference

    ( const Ref<Object>&    obj );

  void                    printTrace

    ( TextOutput&           out )              const;

  Options                 getOptions        () const;

  void                    setOption

    ( Option                option,
      bool                  yesno = true );

  void                    setOptions

    ( Options               options );


 protected:

  virtual                ~ObjectInputStream ();


 private:

  void                    decodeClass_      ();
  Ref<Object>             decodeObject_     ();


 private:

  class                   Data_;

  Ref<Data_>              data_;

};


JEM_END_PACKAGE( io )

#endif
