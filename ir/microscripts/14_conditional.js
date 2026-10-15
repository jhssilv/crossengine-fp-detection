function conditional(a, b) {
  var bigger;
  if (a > b) {
    bigger = a;
  } else {
    bigger = b;
  }
  var abs = a >= 0 ? a : -a;
  return bigger + abs;
}
conditional(3, 7);
