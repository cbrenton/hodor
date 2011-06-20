#include "nyuparser.h"

using namespace std;

/* a collection of functions for syntax verification */

void NYUParser::ParseLeftAngle() {
  //GetToken();
  Token t = tokenizer->GetToken();
  if(t.id != T_LEFT_ANGLE ) tokenizer->Error("Expected <");
}

void NYUParser::ParseRightAngle() { 
  //GetToken();
  Token t = tokenizer->GetToken();
  if(t.id != T_RIGHT_ANGLE ) tokenizer->Error("Expected >");  
}

double NYUParser::ParseDouble() { 
  //GetToken();
  Token t = tokenizer->GetToken();
  if(t.id != T_DOUBLE ) tokenizer->Error("Expected a number");
  return t.double_value;
}

void NYUParser::ParseLeftCurly() { 
  //GetToken();
  Token t = tokenizer->GetToken();
  if(t.id != T_LEFT_CURLY ) tokenizer->Error("Expected {");
}

void NYUParser::ParseRightCurly() { 
  //GetToken();
  Token t = tokenizer->GetToken();
  if(t.id != T_RIGHT_CURLY ) tokenizer->Error("Expected }");
}

void NYUParser::ParseComma() { 
  //GetToken();
  Token t = tokenizer->GetToken();
  if(t.id != T_COMMA ) tokenizer->Error("Expected ,");
}

void NYUParser::ParseVector(vect3 & v) { 
  ParseLeftAngle();
  v.x = ParseDouble();
  ParseComma();
  v.y = ParseDouble();
  ParseComma();
  v.z = ParseDouble();
  ParseRightAngle();
}

void NYUParser::ParseRGBFColor(color & c, float & f) { 
  ParseLeftAngle();
  c.r = ParseDouble();
  ParseComma();
  c.g = ParseDouble();
  ParseComma();
  c.b = ParseDouble();
  ParseComma();
  f = ParseDouble();
  ParseRightAngle();
}

void NYUParser::ParseRGBColor(color & c, float & f) { 
  ParseLeftAngle();
  c.r = ParseDouble();
  ParseComma();
  c.g = ParseDouble();
  ParseComma();
  c.b = ParseDouble();
  f = 0.0;
  ParseRightAngle();
}

void NYUParser::ParseColor(color & c, float & f) { 
  //GetToken();
  Token t = tokenizer->GetToken();
  if(t.id == T_RGB) 
    ParseRGBColor(c,f);
  else if ( t.id == T_RGBF )
    ParseRGBFColor(c,f);
  else tokenizer->Error("Expected rgb or rgbf");
}

void NYUParser::PrintColor(color & c, float & f) { 
  //printf("rgbf <%.3g,%.3g,%.3g,%.3g>", c->r, c->g, c->b, c->f);
  cout << "rgbf <" << c.r << "," << c.g << "," << c.b << "," << f << ">" << endl;
}


//void NYUParser::ParseMatrix( struct Matrix4d* m) {
void NYUParser::ParseMatrix() {
  cout << "Error: Cannot Parse Matrix" << endl;
  exit(0);
  /*
  SetIdentity4d(m);
  ParseLeftAngle();
  m->val[4*0] = ParseDouble(); ParseComma(); 
  m->val[4*1] = ParseDouble(); ParseComma();
  m->val[4*2] = ParseDouble(); ParseComma();

  m->val[4*0+1] = ParseDouble(); ParseComma(); 
  m->val[4*1+1] = ParseDouble(); ParseComma();
  m->val[4*2+1] = ParseDouble(); ParseComma();

  m->val[4*0+2] = ParseDouble(); ParseComma(); 
  m->val[4*1+2] = ParseDouble(); ParseComma();
  m->val[4*2+2] = ParseDouble(); ParseComma();

  m->val[4*0+3] = ParseDouble(); ParseComma(); 
  m->val[4*1+3] = ParseDouble(); ParseComma();
  m->val[4*2+3] = ParseDouble(); 
  ParseRightAngle();
  */
}


void NYUParser::ParseTransform(Shape & s) { 
  /* if there is nothing to parse, this is not our problem: 
     this should be handled by the caller */
  vect3 v;
  //struct Matrix4d m;
  Token t;
  Transformation * trans;
  //while(1) {
  for(;;){
    //GetToken();
    t = tokenizer->GetToken();
    switch( t.id ) { 
    case T_SCALE: 
      ParseVector(v);
      trans = new Scale(v.x,v.y,v.z);
      s.addTransformation(trans);
      //ScaleMatrix4d(transf, &v);
      break;
    case T_ROTATE:
      ParseVector(v);
      trans = new Rotation(v.x,v.y,v.z);
      s.addTransformation(trans);
      //RotateMatrix4d(transf, &v);
      break;
    case T_TRANSLATE:
      ParseVector(v);
      trans = new Translation(v.x,v.y,v.z);
      s.addTransformation(trans);
      //TranslateMatrix4d(transf, &v);
      break;
      /* once we run into an unknown token, we assume there are no 
         more  transforms to parse and we return to caller */
    case T_MATRIX:
      ParseMatrix();
      //MultMatrix4d(transf, &m, transf); 
      break;
    default: tokenizer->UngetToken(); return; 
    }    
  }
}

void NYUParser::ParsePigment(Pigment & p) { 
  Token t;
  ParseLeftCurly();
  //while(1) {
  for(;;){
    //GetToken();
    t = tokenizer->GetToken();
    if( t.id == T_COLOR)
      ParseColor(p.c,p.f);
    else if( t.id == T_RIGHT_CURLY) return;
    else tokenizer->Error("error parsing pigment: unexpected token");
  }    
}

void NYUParser::PrintPigment(Pigment & p) {
  printf("\tpigment { color ");
  PrintColor(p.c,p.f);
  printf("}");
}

void NYUParser::ParseFinish(Finish & f) { 
  Token t;
  ParseLeftCurly();
  //while(1) { 
  for(;;){
    //GetToken();
    t = tokenizer->GetToken();
    switch(t.id) {
    case T_AMBIENT:
      f.ambient = ParseDouble();
      break;
    case T_DIFFUSE:
      f.diffuse = ParseDouble();
      break;
    case T_PHONG:
      //finish->phong = ParseDouble();
      cout << "Error: phong not supported" << endl;
      break;
    case T_PHONG_SIZE:
      //finish->phong_size = ParseDouble();
      cout << "Error: phong size not supported" << endl;
      break;
    case T_REFLECTION:
      f.reflection = ParseDouble();
      break;
    case T_METALLIC:
      //finish->metallic = ( ParseDouble()!= 0.0 ? 1: 0);
      cout << "Error: metallic not supported" << endl;
      break;
    //added by mbecker
    case T_REFRACTION:
      f.refraction = ParseDouble();
      break;
    //added by mbecker
    case T_SPECULAR:
      f.specular = ParseDouble();
      break;
    case T_ROUGHNESS:
      f.roughness = ParseDouble();
      break;
    case T_RIGHT_CURLY: return;
    default: tokenizer->Error("Error parsing finish: unexpected token");
    }   
  } 
}

void NYUParser::PrintFinish(Finish & f) {
  /*
  printf("\tfinish { ambient %.3g diffuse %.3g phong %.3g phong_size %.3g reflection %.3g metallic %d }\n", 
         finish->ambient, finish->diffuse, 
         finish->phong, finish->phong_size, 
         finish->reflection, finish->metallic);
  */
  cout << "\tfinish {";
  cout << " ambient " << f.ambient;
  cout << " diffuse " << f.diffuse;
  cout << " specular " << f.specular;
  cout << " roughness " << f.roughness;
  cout << " reflection " << f.reflection;
  cout << " refraction " << f.refraction;
  cout << " ior " << f.ior << " }" << endl;
}

//void NYUParser::ParseInterior( struct Interior* interior) {
void NYUParser::ParseInterior(float & ior) {
  Token t;
  ParseLeftCurly();
  //while(1) {
  for(;;){
    //GetToken();
    t = tokenizer->GetToken();
    if( t.id == T_RIGHT_CURLY) return;
    else if (t.id == T_IOR) { 
      //interior->ior = ParseDouble();
      ior = ParseDouble();
    } else tokenizer->Error("Error parsing interior: unexpected token\n");
  }
}

/*
void NYUParser::InitModifiers(struct ModifierStruct* modifiers) { 

  SetIdentity4d(&(modifiers->transform));

  modifiers->pigment.color.r = 0;
  modifiers->pigment.color.g = 0;
  modifiers->pigment.color.b = 0;
  modifiers->pigment.color.f = 0;

  modifiers->finish.ambient    = 0.1;
  modifiers->finish.diffuse    = 0.6;
  modifiers->finish.phong      = 0.0;
  modifiers->finish.phong_size = 40.0;
  modifiers->finish.reflection = 0;

  modifiers->interior.ior = 1.0; 
}
*/


void NYUParser::ParseModifiers(Shape & s) {
  Token t; 
  //while(1) {
  for(;;){
    //GetToken();
    t = tokenizer->GetToken();
    switch(t.id) { 
    case  T_SCALE:
    case T_ROTATE:
    case T_TRANSLATE:
    case T_MATRIX: 
      tokenizer->UngetToken();
      //ParseTransform(&(modifiers->transform));
      ParseTransform(s);
      break;
    case T_PIGMENT:
      //ParsePigment(&(modifiers->pigment));
      ParsePigment(s.p);
      break;
    case T_FINISH:
      //ParseFinish(&(modifiers->finish));
      ParseFinish(s.f);
      break;
    case T_INTERIOR:
      //ParseInterior(&(modifiers->interior));
      ParseInterior(s.f.ior);
      break;      
    default: tokenizer->UngetToken(); return;
    }
  }
}

void NYUParser::PrintModifiers(Shape & s) {
  /*
  printf("\tmatrix "); PrintMatrix4d(modifiers->transform);
  printf("\n"); 
  PrintPigment(&(modifiers->pigment));
  printf("\n"); 
  PrintFinish(&(modifiers->finish));
  printf("\tinterior { ior %.3g }\n", modifiers->interior.ior);
  */
  PrintPigment(s.p);
  cout << "\n" << endl;
  PrintFinish(s.f);
  cout << "\n" << endl;
}


Camera NYUParser::ParseCamera() { 
  /* these are variables where we store the information about the
     camera; they can be used in the end of the function to 
     assign fields in the camera object */

  vect3 location, right, up, look_at; 
  double angle;
  //struct Matrix4d  transform; 
  bool done = false;
  Token t;
  //SetIdentity4d(&transform);

  /* default values */
  /*
  SetVect(&location, 0.0, 0.0,0.0);   
  SetVect(&look_at, 0.0, 0.0,1.0);   
  SetVect(&right, 1.0, 0.0, 0.0);
  SetVect(&up, 0.0,1.0,0.0);
  angle = 60.0*M_PI/180.0;
  */
  
  ParseLeftCurly();

  /* parse camera parameters */  
  while(!done) {     
    //GetToken();
    t = tokenizer->GetToken();
    switch(t.id) { 
    case T_LOCATION:  ParseVector(location);  break;
    case T_RIGHT:     ParseVector(right);     break;
    case T_UP:        ParseVector(up);        break;
    case T_LOOK_AT:   ParseVector(look_at);   break;
    case T_ANGLE:     angle = M_PI*ParseDouble()/180.0;   break;
    default: done = true; tokenizer->UngetToken(); break;
    }
  }

  //ParseTransform(&transform); //NOTE: may need this!
  ParseRightCurly();

  /* TODO: assignment to the camera object fields should happen here;
     for now, we just print the values */
  
  //printf("camera { \n");
  //printf("\tlocation ");   PrintVect(location);   printf("\n");
  //printf("\tright ");      PrintVect(right);      printf("\n");
  //printf("\tup ");         PrintVect(up);         printf("\n");
  //printf("\tangle %.3g\n", angle*180.0/M_PI);
  //printf("\tlook_at ");   PrintVect(look_at);   printf("\n");
  //printf("\tmatrix "); PrintMatrix4d(transform);
  //printf("\n}\n");
  return Camera(location,up,right,look_at);
}

void NYUParser::ParsePolygon() { 
  cout << "Error: Cannot parse polygon" << endl;
  exit(0);
  /*
  // these three variables store information about the polygon 
  int num_vertices;
  struct Vector* vertices;
  int vert_cnt;
  int done = FALSE;
  int i;
  struct ModifierStruct modifiers;
  InitModifiers(&modifiers);


  num_vertices = 0; 
  vertices = 0;

  ParseLeftCurly();
  num_vertices = (int)ParseDouble();

  if( num_vertices < 3 ) tokenizer->Error("Polygon must have at least 3 vertices");

  vertices = (struct Vector*)malloc( sizeof(struct Vector)*num_vertices);
  ParseComma();

  for( vert_cnt = 0; vert_cnt < num_vertices; vert_cnt++) { 
    ParseVector(&(vertices[vert_cnt]));
    if( vert_cnt < num_vertices-1 ) ParseComma();
  }
  ParseModifiers(&modifiers);
  ParseRightCurly();

  // TODO: assignment to the polygon object fields should happen here;
  // for now, we just print the values 

  printf("polygon {\n");
  printf("\t%d,\n\t", num_vertices);
  for( i = 0; i < num_vertices-1; i++) {
    PrintVect(vertices[i]); printf(",");
  }
  PrintVect(vertices[num_vertices-1]); 
  printf("\n");
  PrintModifiers(&modifiers);
  printf("\n}\n");
  */
}

Sphere * NYUParser::ParseSphere() { 
  //vect3 center; 
  //double radius; 
  //struct ModifierStruct modifiers;
  Sphere * s = new Sphere();
  //InitModifiers(&modifiers);
  //SetVect(&center, 0,0,0);
  //radius = 1.0;

  ParseLeftCurly();
  ParseVector(s->center);
  ParseComma();
  s->radius = ParseDouble();

  ParseModifiers(*s);
  ParseRightCurly();

 
  /* TODO: assignment to the sphere object fields should happen here;
     for now, we just print the values */

  //printf("sphere {\n\t");
  //PrintVect(center); printf(", %.3g\n", radius);
  //PrintModifiers(&modifiers);
  //printf("\n}\n");
  return s;
}

Box * NYUParser::ParseBox() {
  //struct Vector corner1, corner2; 
  //struct ModifierStruct modifiers;
  //vect3 corner1, corner2;
  //InitModifiers(&modifiers);
  
  Box * b = new Box();

  //SetVect(&corner1, 0,0,0);
  //SetVect(&corner2, 0,0,0);

  ParseLeftCurly();
  ParseVector(b->corner1);
  ParseComma();
  ParseVector(b->corner2);
  ParseModifiers(*b);
  ParseRightCurly();
 
  /* TODO: assignment to the box object fields should happen here;
     for now, we just print the values */

  //printf("box {\n\t");
  //PrintVect(corner1); printf(", "); PrintVect(corner2);
  //PrintModifiers(&modifiers);
  //printf("\n}\n");
  return b;
}

void NYUParser::ParseCylinder() {
  cout << "Error: Cannot parse cylinder" << endl;
  exit(0);
  /*
  //struct Vector base_point , cap_point ;
  vect3 base_point, cap_point;
  double radius;
  //struct ModifierStruct modifiers;
  //InitModifiers(&modifiers);

  //SetVect(&base_point, 0,0,0);
  //SetVect(&cap_point, 0,0,0);
  radius = 0;

  ParseLeftCurly();
  ParseVector(base_point);
  ParseComma();
  ParseVector(cap_point);
  ParseComma();
  radius = ParseDouble();
  ParseModifiers(&modifiers);
  ParseRightCurly();
 
  // TODO: assignment to the cylinder object fields should happen here;
  // for now, we just print the values

  printf("cylinder {\n\t");
  PrintVect(base_point); printf(", "); 
  PrintVect(cap_point); printf(", %.3g\n", radius); 
  PrintModifiers(&modifiers);
  printf("\n}\n");
  */
}

Cone * NYUParser::ParseCone() {
  //struct Vector base_point, cap_point;
  vect3 base_point, cap_point;
  double base_radius, cap_radius; 
  //struct ModifierStruct modifiers;
  //InitModifiers(&modifiers);
  
  Cone * c = new Cone();

  //SetVect(&base_point, 0,0,0);
  //SetVect(&cap_point, 0,0,0);
  base_radius = 0;
  cap_radius = 0;

  ParseLeftCurly();
  ParseVector(base_point);
  ParseComma();
  base_radius = ParseDouble();
  ParseComma();
  ParseVector(cap_point);
  ParseComma();  
  cap_radius = ParseDouble();
  ParseModifiers(*c);
  ParseRightCurly();

  /* TODO: assignment to the cone object fields should happen here;
     for now, we just print the values */

  //printf("cone {\n\t");
  //PrintVect(base_point); printf(", %.3g, ", base_radius); 
  //PrintVect(cap_point); printf(", %.3g\n",  cap_radius); 
  //PrintModifiers(&modifiers);
  //printf("\n}\n");
}

void NYUParser::ParseQuadric() {
  cout << "Error: Cannot parse cylinder" << endl;
  exit(0);
  /*
  //struct Vector ABC, DEF, GHI;
  vect3 ABC, DEF, GHI; 
  double J;
  //struct ModifierStruct modifiers;
  //InitModifiers(&modifiers);

  //SetVect(&ABC, 0,0,0);
  //SetVect(&DEF, 0,0,0);
  //SetVect(&GHI, 0,0,0);
  J = 0;

  ParseLeftCurly();
  ParseVector(ABC);
  ParseComma();
  ParseVector(DEF);
  ParseComma();
  ParseVector(GHI);
  ParseComma();
  J = ParseDouble();
  ParseModifiers(&modifiers);
  ParseRightCurly();
  
  printf("quadric {\n\t");
  PrintVect(ABC); printf(", "); 
  PrintVect(DEF); printf(", "); 
  PrintVect(GHI); printf(", %.3g\n", J);   
  PrintModifiers(&modifiers);
  printf("\n}\n");
  */
}

//added by mbecker
Triangle * NYUParser::ParseTriangle(){
  Triangle * t = new Triangle();

  ParseLeftCurly();
  ParseVector(t->corner1);
  ParseComma();
  ParseVector(t->corner2);
  ParseComma();
  ParseVector(t->corner3);
  
  ParseModifiers(*t);
  ParseRightCurly();
  return t;
}

Plane * NYUParser::ParsePlane(){
  Plane * p = new Plane();

  ParseLeftCurly();
  ParseVector(p->normal);
  ParseComma();
  p->distance = ParseDouble();

  ParseModifiers(*p);
  ParseRightCurly();
  return p;
}

Light * NYUParser::ParseLightSource() { 
    //struct Color c;
    //struct Vector pos;
    Token t;
    color c;
    float f;
    vect3 pos;
    
    c.r = 0; c.g = 0; c.b = 0; f =0; 
    pos.x = 0; pos.y = 0; pos.z = 0;
    ParseLeftCurly();
    ParseVector(pos);
    t = tokenizer->GetToken();
    if(t.id != T_COLOR) tokenizer->Error("Error parsing light source: missing color");
    ParseColor(c,f);
    ParseRightCurly();
    
    //printf("light_source {\n");
    //printf("\t"); PrintVect(pos); printf("\n");
    //printf("\t"); PrintColor(c); 
    //printf("\n}\n");
    return new Light(pos.x,pos.y,pos.z,c.r,c.g,c.b);
} 

void NYUParser::ParseGlobalSettings() {
	cout << "Error: Cannot parse global settings" << endl;
	exit(0);
	/* 
    struct Color ambient;
    ambient.r = 0; ambient.g = 0; ambient.b = 0; ambient.f =0;     
    ParseLeftCurly();
    while(1) { 
      GetToken();
      if(Token.id == T_AMBIENT_LIGHT) {
        ParseLeftCurly();
        GetToken();
        if(Token.id != T_COLOR) 
          Error("Error parsing light source: missing color");
        ParseColor(&ambient);
        ParseRightCurly();
      } else if(Token.id == T_RIGHT_CURLY) { 
        break;
      } else         
        Error("error parsing default settings: unexpected token");
    }
    printf("global_settings {\n");
    printf("\tambient_light {"); PrintColor(&ambient);
    printf("\n}\n"); 
    */   
}

/* main parsing function calling functions to parse each object;  */

void NYUParser::parse(std::fstream & input, Scene & s) {
  tokenizer = new Tokenizer(input);
  Token t = tokenizer->GetToken();
  Sphere * sphere;
  Box * box;
  Cone * cone;
  
  //added by mbecker
  Triangle * triangle;
  Plane * plane;
  
  //GetToken();
  //while(Token.id != T_EOF) {
  while(t.id != T_EOF) {  
    switch(t.id) { 
    case T_CAMERA:
      s.camera = ParseCamera();
      break;
    case T_POLYGON:
      ParsePolygon();
      break;
    case T_SPHERE:
      //ParseSphere();
      sphere = ParseSphere();
      s.shapes.push_back(sphere);
      s.spheres.push_back(sphere);
      break;
    case T_BOX:
      //ParseBox();
      box = ParseBox();
      s.shapes.push_back(box);
      s.boxes.push_back(box);
      break;
    case T_CYLINDER:
      ParseCylinder();
      break;
    case T_CONE:
      //ParseCone();
      cone = ParseCone();
      s.shapes.push_back(cone);
      s.cones.push_back(cone);
      break;
    case T_QUADRIC:
      ParseQuadric();
      break;
    case T_TRIANGLE:
      triangle = ParseTriangle();
      s.shapes.push_back(triangle);
      s.triangles.push_back(triangle);
      break;
    case T_PLANE:
      plane = ParsePlane();
      s.shapes.push_back(plane);
      s.planes.push_back(plane);
      break;
    case T_LIGHT_SOURCE:
      //ParseLightSource();
      s.lights.push_back(ParseLightSource());
      break;
    case T_GLOBAL_SETTINGS:
      ParseGlobalSettings();
      break;
    default:
      tokenizer->Error("Unknown statement");
    }
    //GetToken();
    t = tokenizer->GetToken();
  }
  delete tokenizer;
}
