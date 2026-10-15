function typeCoercion(x) {
  var n   = Number(x);
  var s   = String(x);
  var pos = +x;
  var b   = Boolean(x);
  return n + pos;
}
typeCoercion("42");
