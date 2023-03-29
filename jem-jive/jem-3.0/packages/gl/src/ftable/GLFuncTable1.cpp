
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


#include <jem/gl/ftable/GLFuncTable1.h>
#include "GLFuncLoader.h"


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class GLFuncTable1Core
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GLFuncTable1Core::GLFuncTable1Core

  ( GLFuncLoader*  loader )

{
  JEM_LOAD_FUNC ( glViewport               );
  JEM_LOAD_FUNC ( glDepthRange             );
  JEM_LOAD_FUNC ( glIsEnabled              );
  JEM_LOAD_FUNC ( glGetTexImage            );
  JEM_LOAD_FUNC ( glGetString              );
  JEM_LOAD_FUNC ( glGetIntegerv            );
  JEM_LOAD_FUNC ( glGetFloatv              );
  JEM_LOAD_FUNC ( glGetError               );
  JEM_LOAD_FUNC ( glGetDoublev             );
  JEM_LOAD_FUNC ( glGetBooleanv            );
  JEM_LOAD_FUNC ( glGetTexParameteriv      );
  JEM_LOAD_FUNC ( glGetTexParameterfv      );
  JEM_LOAD_FUNC ( glGetTexLevelParameteriv );
  JEM_LOAD_FUNC ( glGetTexLevelParameterfv );
  JEM_LOAD_FUNC ( glReadPixels             );
  JEM_LOAD_FUNC ( glReadBuffer             );
  JEM_LOAD_FUNC ( glPixelStorei            );
  JEM_LOAD_FUNC ( glPixelStoref            );
  JEM_LOAD_FUNC ( glDepthFunc              );
  JEM_LOAD_FUNC ( glLogicOp                );
  JEM_LOAD_FUNC ( glBlendFunc              );
  JEM_LOAD_FUNC ( glFlush                  );
  JEM_LOAD_FUNC ( glFinish                 );
  JEM_LOAD_FUNC ( glEnable                 );
  JEM_LOAD_FUNC ( glDisable                );
  JEM_LOAD_FUNC ( glDepthMask              );
  JEM_LOAD_FUNC ( glColorMask              );
  JEM_LOAD_FUNC ( glClearDepth             );
  JEM_LOAD_FUNC ( glClearColor             );
  JEM_LOAD_FUNC ( glClear                  );
  JEM_LOAD_FUNC ( glDrawBuffer             );
  JEM_LOAD_FUNC ( glTexImage2D             );
  JEM_LOAD_FUNC ( glTexImage1D             );
  JEM_LOAD_FUNC ( glTexParameteriv         );
  JEM_LOAD_FUNC ( glTexParameteri          );
  JEM_LOAD_FUNC ( glTexParameterfv         );
  JEM_LOAD_FUNC ( glTexParameterf          );
  JEM_LOAD_FUNC ( glScissor                );
  JEM_LOAD_FUNC ( glPolygonMode            );
  JEM_LOAD_FUNC ( glPointSize              );
  JEM_LOAD_FUNC ( glLineWidth              );
  JEM_LOAD_FUNC ( glHint                   );
  JEM_LOAD_FUNC ( glFrontFace              );
  JEM_LOAD_FUNC ( glCullFace               );
}


GLFuncTable1Core::~GLFuncTable1Core ()
{}


//=======================================================================
//   class GLFuncTable1Compat
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


GLFuncTable1Compat::GLFuncTable1Compat

  ( GLFuncLoader*  loader )

{
  JEM_LOAD_FUNC ( glTranslatef     );
  JEM_LOAD_FUNC ( glTranslated     );
  JEM_LOAD_FUNC ( glScalef         );
  JEM_LOAD_FUNC ( glScaled         );
  JEM_LOAD_FUNC ( glRotatef        );
  JEM_LOAD_FUNC ( glRotated        );
  JEM_LOAD_FUNC ( glPushMatrix     );
  JEM_LOAD_FUNC ( glPopMatrix      );
  JEM_LOAD_FUNC ( glOrtho          );
  JEM_LOAD_FUNC ( glMatrixMode     );
  JEM_LOAD_FUNC ( glLoadMatrixd    );
  JEM_LOAD_FUNC ( glLoadMatrixf    );
  JEM_LOAD_FUNC ( glLoadIdentity   );
  JEM_LOAD_FUNC ( glTexEnviv       );
  JEM_LOAD_FUNC ( glTexEnvi        );
  JEM_LOAD_FUNC ( glTexEnvfv       );
  JEM_LOAD_FUNC ( glTexEnvf        );
  JEM_LOAD_FUNC ( glGetMaterialiv  );
  JEM_LOAD_FUNC ( glGetMaterialfv  );
  JEM_LOAD_FUNC ( glGetLightiv     );
  JEM_LOAD_FUNC ( glGetLightfv     );
  JEM_LOAD_FUNC ( glGetClipPlane   );
  JEM_LOAD_FUNC ( glDrawPixels     );
  JEM_LOAD_FUNC ( glCopyPixels     );
  JEM_LOAD_FUNC ( glPixelTransferi );
  JEM_LOAD_FUNC ( glPixelTransferf );
  JEM_LOAD_FUNC ( glAlphaFunc      );
  JEM_LOAD_FUNC ( glPushAttrib     );
  JEM_LOAD_FUNC ( glPopAttrib      );
  JEM_LOAD_FUNC ( glAccum          );
  JEM_LOAD_FUNC ( glClearAccum     );
//  JEM_LOAD_FUNC ( glPushName       );
//  JEM_LOAD_FUNC ( glPopName        );
  JEM_LOAD_FUNC ( glSelectBuffer   );
  JEM_LOAD_FUNC ( glShadeModel     );
  JEM_LOAD_FUNC ( glMaterialiv     );
  JEM_LOAD_FUNC ( glMateriali      );
  JEM_LOAD_FUNC ( glMaterialfv     );
  JEM_LOAD_FUNC ( glMaterialf      );
  JEM_LOAD_FUNC ( glLineStipple    );
  JEM_LOAD_FUNC ( glLightModeliv   );
  JEM_LOAD_FUNC ( glLightModeli    );
  JEM_LOAD_FUNC ( glLightModelfv   );
  JEM_LOAD_FUNC ( glLightModelf    );
  JEM_LOAD_FUNC ( glLightiv        );
  JEM_LOAD_FUNC ( glLighti         );
  JEM_LOAD_FUNC ( glLightfv        );
  JEM_LOAD_FUNC ( glLightf         );
  JEM_LOAD_FUNC ( glColorMaterial  );
  JEM_LOAD_FUNC ( glClipPlane      );
  JEM_LOAD_FUNC ( glVertex4s       );
  JEM_LOAD_FUNC ( glVertex4i       );
  JEM_LOAD_FUNC ( glVertex4f       );
  JEM_LOAD_FUNC ( glVertex4d       );
  JEM_LOAD_FUNC ( glVertex3s       );
  JEM_LOAD_FUNC ( glVertex3i       );
  JEM_LOAD_FUNC ( glVertex3f       );
  JEM_LOAD_FUNC ( glVertex3d       );
  JEM_LOAD_FUNC ( glVertex2s       );
  JEM_LOAD_FUNC ( glVertex2i       );
  JEM_LOAD_FUNC ( glVertex2f       );
  JEM_LOAD_FUNC ( glVertex2d       );
  JEM_LOAD_FUNC ( glTexCoord2s     );
  JEM_LOAD_FUNC ( glTexCoord2i     );
  JEM_LOAD_FUNC ( glTexCoord2f     );
  JEM_LOAD_FUNC ( glTexCoord2d     );
  JEM_LOAD_FUNC ( glTexCoord1s     );
  JEM_LOAD_FUNC ( glTexCoord1i     );
  JEM_LOAD_FUNC ( glTexCoord1f     );
  JEM_LOAD_FUNC ( glTexCoord1d     );
  JEM_LOAD_FUNC ( glRects          );
  JEM_LOAD_FUNC ( glRecti          );
  JEM_LOAD_FUNC ( glRectf          );
  JEM_LOAD_FUNC ( glRectd          );
  JEM_LOAD_FUNC ( glRasterPos4s    );
  JEM_LOAD_FUNC ( glRasterPos4i    );
  JEM_LOAD_FUNC ( glRasterPos4f    );
  JEM_LOAD_FUNC ( glRasterPos4d    );
  JEM_LOAD_FUNC ( glRasterPos3s    );
  JEM_LOAD_FUNC ( glRasterPos3i    );
  JEM_LOAD_FUNC ( glRasterPos3f    );
  JEM_LOAD_FUNC ( glRasterPos3d    );
  JEM_LOAD_FUNC ( glRasterPos2s    );
  JEM_LOAD_FUNC ( glRasterPos2i    );
  JEM_LOAD_FUNC ( glRasterPos2f    );
  JEM_LOAD_FUNC ( glRasterPos2d    );
  JEM_LOAD_FUNC ( glNormal3s       );
  JEM_LOAD_FUNC ( glNormal3i       );
  JEM_LOAD_FUNC ( glNormal3f       );
  JEM_LOAD_FUNC ( glNormal3d       );
  JEM_LOAD_FUNC ( glNormal3b       );
  JEM_LOAD_FUNC ( glEnd            );
  JEM_LOAD_FUNC ( glColor4us       );
  JEM_LOAD_FUNC ( glColor4ui       );
  JEM_LOAD_FUNC ( glColor4ub       );
  JEM_LOAD_FUNC ( glColor4s        );
  JEM_LOAD_FUNC ( glColor4i        );
  JEM_LOAD_FUNC ( glColor4f        );
  JEM_LOAD_FUNC ( glColor4d        );
  JEM_LOAD_FUNC ( glColor4b        );
  JEM_LOAD_FUNC ( glColor3us       );
  JEM_LOAD_FUNC ( glColor3ui       );
  JEM_LOAD_FUNC ( glColor3ub       );
  JEM_LOAD_FUNC ( glColor3s        );
  JEM_LOAD_FUNC ( glColor3i        );
  JEM_LOAD_FUNC ( glColor3f        );
  JEM_LOAD_FUNC ( glColor3d        );
  JEM_LOAD_FUNC ( glColor3b        );
  JEM_LOAD_FUNC ( glBitmap         );
  JEM_LOAD_FUNC ( glBegin          );
}


GLFuncTable1Compat::~GLFuncTable1Compat ()
{}


JEM_END_PACKAGE( gl )
