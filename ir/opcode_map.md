# V8 ↔ SpiderMonkey Opcode Mapping

Sources: `navegadores/v8-tcc/v8/src/interpreter/bytecodes.h` and
`navegadores/spidermonkey-tcc/gecko-dev/js/src/vm/Opcodes.h`.

Architecture note: V8 is **accumulator-based** (register machine); SM is
**stack-based**. A V8 binary op like `Add r1` (add r1 to accumulator) maps to
SM's `GetLocal 0; GetLocal 1; Add` — one extra push per operand. The IR
dictionary captures semantic equivalence, not 1:1 instruction count.

Priority column: **P1** = appears constantly in fingerprinting code, micro-script required;
**P2** = appears occasionally, micro-script useful; **P3** = rarely in FP code, skip.

---

## Full Category Map

| Priority | JS Construct | V8 (Ignition) | SpiderMonkey |
|---|---|---|---|
| P1 | **Literals** | `LdaSmi`, `LdaConstant`, `LdaZero`, `LdaTrue`, `LdaFalse`, `LdaNull`, `LdaUndefined` | `Zero`, `One`, `Int8`, `Int32`, `Uint16`, `Uint24`, `Double`, `String`, `True`, `False`, `Null`, `Undefined` |
| P1 | **Local var read/write** | `Ldar`, `Star`, `Star0`–`Star15`, `Mov` | `GetLocal`, `SetLocal` |
| P1 | **Global var read/write** | `LdaGlobal`, `StaGlobal` | `GetGName`, `SetGName`, `GetName`, `SetName`, `BindUnqualifiedGName` |
| P1 | **Closure/upvar read/write** | `LdaContextSlot`, `StaContextSlot`, `LdaCurrentContextSlot`, `StaCurrentContextSlot` | `GetAliasedVar`, `SetAliasedVar` |
| P1 | **Named property read** | `GetNamedProperty` | `GetProp` |
| P1 | **Named property write** | `SetNamedProperty`, `DefineNamedOwnProperty` | `SetProp`, `InitProp`, `StrictSetProp` |
| P1 | **Keyed property read** | `GetKeyedProperty` | `GetElem` |
| P1 | **Keyed property write** | `SetKeyedProperty`, `StaInArrayLiteral` | `SetElem`, `InitElemArray`, `InitElemInc` |
| P1 | **Arithmetic** | `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Exp` + `*Smi` variants | `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Pow` |
| P1 | **Bitwise** | `BitwiseOr`, `BitwiseAnd`, `BitwiseXor`, `ShiftLeft`, `ShiftRight`, `ShiftRightLogical` + `*Smi` variants | `BitOr`, `BitAnd`, `BitXor`, `Lsh`, `Rsh`, `Ursh` |
| P1 | **Unary** | `Negate`, `BitwiseNot`, `Inc`, `Dec`, `LogicalNot`, `ToBooleanLogicalNot` | `Neg`, `BitNot`, `Inc`, `Dec`, `Not`, `Pos` |
| P1 | **Comparisons** | `TestEqual`, `TestEqualStrict`, `TestLessThan`, `TestGreaterThan`, `TestLessThanOrEqual`, `TestGreaterThanOrEqual` | `Eq`, `Ne`, `StrictEq`, `StrictNe`, `Lt`, `Le`, `Gt`, `Ge` |
| P1 | **`typeof`** | `TypeOf`, `TestTypeOf` | `Typeof`, `TypeofEq`, `TypeofExpr` |
| P1 | **Conditional jumps** | `JumpIfTrue`, `JumpIfFalse`, `JumpIfToBooleanTrue`, `JumpIfToBooleanFalse`, `JumpIfNull`, `JumpIfUndefined`, `JumpIfUndefinedOrNull` | `JumpIfTrue`, `JumpIfFalse`, `JumpTarget` |
| P1 | **Loops** | `JumpLoop`, `Jump` | `Goto`, `LoopHead` |
| P1 | **Logical short-circuit** | `JumpIfToBooleanFalse` + reload (&&), `JumpIfToBooleanTrue` + reload (\|\|) | `And`, `Or`, `Coalesce` |
| P1 | **Method call** | `CallProperty0`, `CallProperty1`, `CallProperty2`, `CallProperty` | `Call`, `CallIgnoresRv`, `CallContent` |
| P1 | **Function call** | `CallUndefinedReceiver`, `CallUndefinedReceiver0/1/2`, `CallAnyReceiver` | `Call`, `CallIgnoresRv` |
| P1 | **Constructor call** | `Construct`, `ConstructWithSpread` | `New`, `SpreadNew` |
| P1 | **Return** | `Return` | `Return`, `RetRval` |
| P1 | **Closure creation** | `CreateClosure`, `CreateFunctionContext`, `CreateFunctionContextWithCells` | `Lambda` |
| P1 | **Object literal** | `CreateObjectLiteral`, `CreateEmptyObjectLiteral` | `NewObject`, `NewInit`, `Object` |
| P1 | **Array literal** | `CreateArrayLiteral`, `CreateEmptyArrayLiteral` | `NewArray` |
| P1 | **try/catch/throw** | `Throw`, `ReThrow` | `Throw`, `Exception`, `Try` |
| P2 | **Type coercions** | `ToNumber`, `ToString`, `ToBoolean`, `ToNumeric`, `ToName` | `ToNumeric`, `ToString`, `Pos` |
| P2 | **`instanceof`** | `TestInstanceOf` | `Instanceof` |
| P2 | **String concat (specialized)** | `Add_StringConstant_Internalize` | *(no direct equivalent — uses `Add`)* |
| P3 | **`for...in`** | `ForInEnumerate`, `ForInPrepare`, `ForInNext`, `ForInStep` | `Iter`, `MoreIter`, `IsNoIter`, `EndIter`, `CloseIter` |
| P3 | **Generators / async** | `SuspendGenerator`, `ResumeGenerator`, `SwitchOnGeneratorState` | `Generator`, `Yield`, `Resume`, `Await` |
| P3 | **`void` operator** | `LdaUndefined` (after discard) | `Void` |
| P3 | **`delete`** | `DeletePropertyStrict`, `DeletePropertySloppy` | `DelProp`, `StrictDelProp`, `DelElem` |
| P3 | **`in` operator** | `TestIn` | `In` |
| P3 | **TDZ checks** | `ThrowReferenceErrorIfHole` | `CheckLexical`, `CheckAliasedLexical` |
| P3 | **`let`/`const` init** | `StaCurrentContextSlot` | `InitLexical`, `InitAliasedLexical` |
| P3 | **`arguments` / rest** | `CreateMappedArguments`, `CreateRestParameter` | `Arguments`, `Rest` |
| P3 | **Module** | `LdaModuleVariable`, `StaModuleVariable` | `GetImport`, `ImportMeta` |
| P3 | **Private fields** | `GetPrivateField`, `SetPrivateField` | `CheckPrivateField`, `NewPrivateName` |

---

## Notes

- V8 `*Smi` variants (`AddSmi`, `SubSmi`, `MulSmi`, etc.) are integer fast-paths with no SM equivalent; they map to the same IR token as their generic counterparts.
- SM `And`/`Or`/`Coalesce` are jump instructions (they short-circuit and *keep* TOS); V8 encodes the same semantics as a conditional jump + a reload on the fall-through path.
- SM `FunctionThis` (pushes `this` inside a function) has no V8 equivalent opcode — V8 loads `this` via a register or context slot.
- SM `Call` vs V8 `CallProperty*` / `CallUndefinedReceiver*`: V8 splits on receiver presence and arg count; SM uses a single `Call` with `argc` operand.
- `StrictConstantEq` / `StrictConstantNe` in SM are optimized comparisons against a constant; treated as `StrictEq`/`StrictNe` in the IR.
