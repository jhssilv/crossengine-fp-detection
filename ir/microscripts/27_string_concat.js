function stringConcat(a, b) {
  var s1 = a + b;
  var s2 = "prefix_" + b;
  var s3 = a + "_suffix";
  var s4 = s1 + s2 + s3;
  return s4;
}
stringConcat("foo", "bar");
