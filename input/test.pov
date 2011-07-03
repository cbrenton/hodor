camera {
  location  <0, 0, 14>
  up        <0,  1,  0>
  right     <1.33333, 0,  0>
  look_at   <0, 0, 0>
}

light_source {<-100, 100, 100> color rgb <1.5, 1.5, 1.5>}

sphere { <3, 3, -2>, 1.5
  pigment { color rgb <1.0, 0.0, 0.0>}
  finish {ambient 0.2 diffuse 0.4}
  translate <0, 0, 0>
}

sphere { <2, 2, -1>, 1.75
  pigment { color rgb <0.0, 1.0, 0.0>}
  finish {ambient 0.2 diffuse 0.4}
  translate <0, 0, 0>
}

sphere { <0, 0, 0>, 2
  pigment { color rgb <0.0, 0.0, 1.0>}
  finish {ambient 0.2 diffuse 0.4}
  translate <0, 0, 0>
}
