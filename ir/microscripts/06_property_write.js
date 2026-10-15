function propertyWrite(obj, val) {
  obj.name = val;
  obj.count = obj.count + 1;
  return obj;
}
propertyWrite({ name: "", count: 0 }, "hello");
