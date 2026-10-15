function propertyRead(obj) {
  var a = obj.name;
  var b = obj.value;
  var c = obj.count;
  return a;
}
propertyRead({ name: "fp", value: 42, count: 1 });
