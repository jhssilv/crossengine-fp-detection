function instanceofCheck(x) {
  var isArr = x instanceof Array;
  var isObj = x instanceof Object;
  return isArr && isObj;
}
instanceofCheck([1, 2, 3]);
instanceofCheck({});
