function typeofCheck(x) {
  var t    = typeof x;
  var isS  = typeof x === "string";
  var isN  = typeof x === "number";
  var isU  = typeof x === "undefined";
  var isF  = typeof x === "function";
  return t;
}
typeofCheck("hello");
