function logical(a, b, c) {
  var and = a && b;
  var or  = a || b;
  var nc  = c ?? b;
  return and || or || nc;
}
logical(1, 2, null);
