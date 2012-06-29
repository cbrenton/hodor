#ifndef NYUPARSER_H
#define NYUPARSER_H

#include "parse/tokens.h"
#include "Scene.h"
//#include "geom/Transformation.h"

class NYUParser{
   private:
      Tokenizer * tokenizer;

      // helper functions
      void ParseLeftAngle();
      void ParseRightAngle();
      double ParseDouble();
      void ParseLeftCurly();
      void ParseRightCurly();
      void ParseComma();
      void ParseVector(glm::vec3 & v);
      void ParseRGBFColor(color & c, float & f);
      void ParseRGBColor(color & c, float & f);
      void ParseColor(color & c, float & f);
      void PrintColor(color & c, float & f);
      void ParseMatrix();
      void ParseTransform(Geometry & s);
      void ParsePigment(Pigment & p);
      void PrintPigment(Pigment & p);
      void ParseFinish(Finish & f);
      void PrintFinish(Finish & f);
      // void ParseInterior( struct Interior* interior);
      void ParseInterior(float & ior);
      // void InitModifiers(struct ModifierStruct* modifiers);
      // void ParseModifiers(struct ModifierStruct* modifiers);
      // void PrintModifiers(struct ModifierStruct* modifiers);
      void ParseModifiers(Geometry & s);
      void PrintModifiers(Geometry & s);
      // void ParseCamera();
      Camera ParseCamera();
      void ParsePolygon();
      // void ParseSphere();
      Sphere * ParseSphere();
      // void ParseBox();
      Box * ParseBox();
      void ParseCylinder();
      // void ParseCone();
      // Cone * ParseCone();
      void ParseQuadric();
      Triangle * ParseTriangle(); // added by mbecker
      Plane * ParsePlane(); // added by mbecker
      // void ParseLightSource();
      Light * ParseLightSource();
      void ParseGlobalSettings();

   public:
      void parse(std::fstream & input, Scene & s);
      // Parse(FILE* infile);
};

#endif
