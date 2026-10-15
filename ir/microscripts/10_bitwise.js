function bitwise(a, b) {
  var or  = a | b;
  var and = a & b;
  var xor = a ^ b;
  var shl = a << b;
  var shr = a >> b;
  var usr = a >>> b;
  return or + and + xor + shl + shr + usr;
}
bitwise(0xDEAD, 4);
