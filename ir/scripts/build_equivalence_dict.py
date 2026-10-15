#!/usr/bin/env python3
"""
Build the cross-engine IR equivalence dictionary.

Reads ir/microscript_logs/<stem>.json (output of run_microscripts.py),
validates the static opcode→IR mapping derived from opcode_map.md, and
writes ir/equivalence_dict.json.

Output schema:
  {
    "version": 1,
    "v8_to_ir":  {"Add": "BINARY_ADD", ...},
    "sm_to_ir":  {"Add": "BINARY_ADD", ...},
    "ir_tokens": ["BINARY_ADD", ...],
    "coverage":  {
      "v8_seen": [...], "v8_mapped_pct": 97.1, "v8_unmapped": [...],
      "sm_seen": [...], "sm_mapped_pct": 94.3, "sm_unmapped": [...]
    }
  }

Usage:
  python ir/build_equivalence_dict.py           # needs microscript_logs/
  python ir/build_equivalence_dict.py --dry-run # print coverage only, no write
"""
import json
import sys
from pathlib import Path
from collections import Counter

LOGS_DIR = Path(__file__).resolve().parent.parent / "microscript_logs"
OUT_PATH = Path(__file__).resolve().parent.parent / "equivalence_dict.json"

# ── Static mapping: V8 opcode → IR token ──────────────────────────────────────
# Source: navegadores/v8-tcc/v8/src/interpreter/bytecodes.h + opcode_map.md
V8_TO_IR: dict[str, str] = {
    # ── Literals ──
    "LdaSmi":               "LOAD_INT",
    "LdaZero":              "LOAD_INT",
    "LdaConstant":          "LOAD_CONST",
    "LdaTrue":              "LOAD_BOOL",
    "LdaFalse":             "LOAD_BOOL",
    "LdaNull":              "LOAD_NULL",
    "LdaUndefined":         "LOAD_UNDEFINED",
    "LdaTheHole":           "LOAD_UNDEFINED",

    # ── Local registers (accumulator machine) ──
    "Ldar":                 "LOAD_LOCAL",
    "Star":                 "STORE_LOCAL",
    "Mov":                  "STORE_LOCAL",
    **{f"Star{i}": "STORE_LOCAL" for i in range(16)},

    # ── Global variables ──
    "LdaGlobal":            "LOAD_GLOBAL",
    "LdaGlobalInsideTypeof":"LOAD_GLOBAL",
    "StaGlobal":            "STORE_GLOBAL",

    # ── Closure / context variables ──
    "LdaContextSlot":               "LOAD_CLOSURE_VAR",
    "LdaCurrentContextSlot":        "LOAD_CLOSURE_VAR",
    "LdaImmutableContextSlot":      "LOAD_CLOSURE_VAR",
    "LdaImmutableCurrentContextSlot":"LOAD_CLOSURE_VAR",
    "LdaContextSlotNoCell":         "LOAD_CLOSURE_VAR",
    "LdaCurrentContextSlotNoCell":  "LOAD_CLOSURE_VAR",
    "StaContextSlot":               "STORE_CLOSURE_VAR",
    "StaCurrentContextSlot":        "STORE_CLOSURE_VAR",
    "StaContextSlotNoCell":         "STORE_CLOSURE_VAR",
    "StaCurrentContextSlotNoCell":  "STORE_CLOSURE_VAR",
    "LdaLookupSlot":                "LOAD_CLOSURE_VAR",
    "LdaLookupGlobalSlot":          "LOAD_GLOBAL",
    "LdaLookupContextSlot":         "LOAD_CLOSURE_VAR",
    "LdaLookupContextSlotNoCell":   "LOAD_CLOSURE_VAR",
    "LdaLookupSlotInsideTypeof":    "LOAD_CLOSURE_VAR",
    "LdaLookupGlobalSlotInsideTypeof":"LOAD_GLOBAL",
    "LdaLookupContextSlotInsideTypeof":"LOAD_CLOSURE_VAR",
    "LdaLookupContextSlotNoCellInsideTypeof":"LOAD_CLOSURE_VAR",
    "StaLookupSlot":                "STORE_CLOSURE_VAR",

    # ── Named property access ──
    "GetNamedProperty":       "GET_NAMED_PROP",
    "GetNamedPropertyFromSuper":"GET_NAMED_PROP",
    "SetNamedProperty":       "SET_NAMED_PROP",
    "DefineNamedOwnProperty": "DEFINE_OBJ_PROP",

    # ── Keyed property access ──
    "GetKeyedProperty":          "GET_KEYED_PROP",
    "GetEnumeratedKeyedProperty":"GET_KEYED_PROP",
    "SetKeyedProperty":          "SET_KEYED_PROP",
    "StaInArrayLiteral":         "SET_ARR_ELEM",
    "DefineKeyedOwnProperty":    "SET_ARR_ELEM",
    "DefineKeyedOwnPropertyInLiteral":"SET_ARR_ELEM",

    # ── Arithmetic ──
    "Add": "BINARY_ADD",  "AddSmi": "BINARY_ADD",
    "Sub": "BINARY_SUB",  "SubSmi": "BINARY_SUB",
    "Mul": "BINARY_MUL",  "MulSmi": "BINARY_MUL",
    "Div": "BINARY_DIV",  "DivSmi": "BINARY_DIV",
    "Mod": "BINARY_MOD",  "ModSmi": "BINARY_MOD",
    "Exp": "BINARY_POW",  "ExpSmi": "BINARY_POW",
    "Add_StringConstant_Internalize": "BINARY_ADD",  # V8 string concat fast-path

    # ── Bitwise ──
    "BitwiseOr":           "BIT_OR",   "BitwiseOrSmi":  "BIT_OR",
    "BitwiseAnd":          "BIT_AND",  "BitwiseAndSmi": "BIT_AND",
    "BitwiseXor":          "BIT_XOR",  "BitwiseXorSmi": "BIT_XOR",
    "ShiftLeft":           "BIT_SHL",  "ShiftLeftSmi":  "BIT_SHL",
    "ShiftRight":          "BIT_SHR",  "ShiftRightSmi": "BIT_SHR",
    "ShiftRightLogical":   "BIT_USHR", "ShiftRightLogicalSmi": "BIT_USHR",

    # ── Unary ──
    "Negate":              "UNARY_NEG",
    "BitwiseNot":          "UNARY_BITNOT",
    "LogicalNot":          "UNARY_NOT",
    "ToBooleanLogicalNot": "UNARY_NOT",
    "Inc":                 "INC",
    "Dec":                 "DEC",

    # ── Comparisons ──
    "TestEqual":              "CMP_EQ",
    "TestEqualStrict":        "CMP_SEQ",
    "TestLessThan":           "CMP_LT",
    "TestGreaterThan":        "CMP_GT",
    "TestLessThanOrEqual":    "CMP_LTE",
    "TestGreaterThanOrEqual": "CMP_GTE",
    "TestInstanceOf":         "INSTANCEOF",
    "TestIn":                 "IN_OP",
    "TestNull":               "TEST_NULL",
    "TestUndefined":          "TEST_UNDEFINED",
    "TestReferenceEqual":     "CMP_SEQ",
    "TestUndetectable":       "TEST_UNDETECTABLE",
    "TestTypeOf":             "TYPEOF_EQ",

    # ── typeof ──
    "TypeOf": "TYPEOF",

    # ── Jumps / control flow ──
    "Jump":                         "JMP",
    "JumpConstant":                 "JMP",
    "JumpLoop":                     "JMP_LOOP",
    "JumpIfTrue":                   "JMP_IF_TRUE",
    "JumpIfTrueConstant":           "JMP_IF_TRUE",
    "JumpIfFalse":                  "JMP_IF_FALSE",
    "JumpIfFalseConstant":          "JMP_IF_FALSE",
    "JumpIfToBooleanTrue":          "LOGICAL_OR",
    "JumpIfToBooleanTrueConstant":  "LOGICAL_OR",
    "JumpIfToBooleanFalse":         "LOGICAL_AND",
    "JumpIfToBooleanFalseConstant": "LOGICAL_AND",
    "JumpIfNull":                   "JMP_IF_NULL",
    "JumpIfNullConstant":           "JMP_IF_NULL",
    "JumpIfNotNull":                "JMP_IF_NOT_NULL",
    "JumpIfNotNullConstant":        "JMP_IF_NOT_NULL",
    "JumpIfUndefined":              "JMP_IF_UNDEFINED",
    "JumpIfUndefinedConstant":      "JMP_IF_UNDEFINED",
    "JumpIfNotUndefined":           "JMP_IF_NOT_UNDEFINED",
    "JumpIfNotUndefinedConstant":   "JMP_IF_NOT_UNDEFINED",
    "JumpIfUndefinedOrNull":        "LOGICAL_NC",
    "JumpIfUndefinedOrNullConstant":"LOGICAL_NC",
    "JumpIfJSReceiver":             "JMP_IF_OBJ",
    "JumpIfJSReceiverConstant":     "JMP_IF_OBJ",
    "SwitchOnSmiNoFeedback":        "SWITCH",

    # ── Calls ──
    "CallProperty":          "CALL_METHOD",
    "CallProperty0":         "CALL_METHOD",
    "CallProperty1":         "CALL_METHOD",
    "CallProperty2":         "CALL_METHOD",
    "CallUndefinedReceiver": "CALL_FUNC",
    "CallUndefinedReceiver0":"CALL_FUNC",
    "CallUndefinedReceiver1":"CALL_FUNC",
    "CallUndefinedReceiver2":"CALL_FUNC",
    "CallAnyReceiver":       "CALL_FUNC",
    "CallWithSpread":        "CALL_FUNC",
    "CallRuntime":           "CALL_RUNTIME",
    "CallRuntimeForPair":    "CALL_RUNTIME",
    "CallJSRuntime":         "CALL_RUNTIME",
    "InvokeIntrinsic":       "CALL_RUNTIME",

    # ── Construct ──
    "Construct":              "CALL_NEW",
    "ConstructWithSpread":    "CALL_NEW",
    "ConstructForwardAllArgs":"CALL_NEW",

    # ── Return ──
    "Return": "RETURN",

    # ── Closures / scopes ──
    "CreateClosure":                 "CREATE_CLOSURE",
    "CreateFunctionContext":         "CREATE_SCOPE",
    "CreateFunctionContextWithCells":"CREATE_SCOPE",
    "CreateBlockContext":            "CREATE_SCOPE",
    "CreateCatchContext":            "CREATE_SCOPE",
    "CreateWithContext":             "CREATE_SCOPE",
    "CreateEvalContext":             "CREATE_SCOPE",
    "PushContext":                   "PUSH_SCOPE",
    "PopContext":                    "POP_SCOPE",
    "CreateMappedArguments":         "CREATE_ARGUMENTS",
    "CreateUnmappedArguments":       "CREATE_ARGUMENTS",
    "CreateRestParameter":           "CREATE_REST",

    # ── Object / Array literals ──
    "CreateObjectLiteral":      "CREATE_OBJ",
    "CreateEmptyObjectLiteral": "CREATE_OBJ",
    "CloneObject":              "CREATE_OBJ",
    "CreateArrayLiteral":       "CREATE_ARR",
    "CreateEmptyArrayLiteral":  "CREATE_ARR",
    "CreateArrayFromIterable":  "CREATE_ARR",
    "CreateRegExpLiteral":      "CREATE_REGEXP",

    # ── Type coercions ──
    "ToNumber":  "TO_NUMBER",
    "ToNumeric": "TO_NUMBER",
    "ToString":  "TO_STRING",
    "ToBoolean": "TO_BOOL",
    "ToName":    "TO_NAME",
    "ToObject":  "TO_OBJ",

    # ── Exceptions ──
    "Throw":   "THROW",
    "ReThrow": "THROW",
    "SetPendingMessage": "SET_EXCEPTION",
    "ThrowReferenceErrorIfHole":       "TDZ_CHECK",
    "ThrowSuperNotCalledIfHole":       "TDZ_CHECK",
    "ThrowSuperAlreadyCalledIfNotHole":"TDZ_CHECK",
    "ThrowIfNotSuperConstructor":      "TDZ_CHECK",

    # ── Iterators ──
    "ForInEnumerate":        "FOR_IN_START",
    "ForInPrepare":          "FOR_IN_PREP",
    "ForInNext":             "FOR_IN_NEXT",
    "ForInStep":             "FOR_IN_STEP",
    "JumpIfForInDone":       "FOR_IN_DONE",
    "JumpIfForInDoneConstant":"FOR_IN_DONE",
    "GetIterator":           "GET_ITER",

    # ── Generators / async (V8) ──
    "SuspendGenerator":                    "GEN_YIELD",
    "ResumeGenerator":                     "GEN_RESUME",
    "SwitchOnGeneratorState":              "GEN_RESUME",

    # ── Modules ──
    "LdaModuleVariable":                   "LOAD_CLOSURE_VAR",
    "StaModuleVariable":                   "STORE_CLOSURE_VAR",

    # ── Classes ──
    "FindNonDefaultConstructorOrConstruct": "CALL_NEW",

    # ── Misc ──
    "Wide":          "WIDE_PREFIX",
    "ExtraWide":     "WIDE_PREFIX",
    "GetSuperConstructor":  "GET_SUPER",
    "GetTemplateObject":    "GET_TEMPLATE",
    "DeletePropertyStrict": "DELETE_PROP",
    "DeletePropertySloppy": "DELETE_PROP",
    "SetPrototypeProperties":"SET_PROTO",
}

# ── Static mapping: SpiderMonkey opcode → IR token ────────────────────────────
# Source: navegadores/spidermonkey-tcc/gecko-dev/js/src/vm/Opcodes.h + opcode_map.md
SM_TO_IR: dict[str, str] = {
    # ── Literals ──
    "Zero":     "LOAD_INT",
    "One":      "LOAD_INT",
    "Int8":     "LOAD_INT",
    "Int32":    "LOAD_INT",
    "Uint16":   "LOAD_INT",
    "Uint24":   "LOAD_INT",
    "Double":   "LOAD_FLOAT",
    "String":   "LOAD_STR",
    "True":     "LOAD_BOOL",
    "False":    "LOAD_BOOL",
    "Null":     "LOAD_NULL",
    "Undefined":"LOAD_UNDEFINED",
    "BigInt":   "LOAD_BIGINT",
    "Symbol":   "LOAD_SYMBOL",
    "Hole":     "LOAD_UNDEFINED",
    "Uninitialized":"LOAD_UNDEFINED",

    # ── Stack / local vars (stack machine) ──
    "GetLocal": "LOAD_LOCAL",
    "SetLocal": "STORE_LOCAL",
    "Dup2":     "DUP",
    "GetArg":   "LOAD_ARG",
    "SetArg":   "STORE_ARG",
    "GetFrameArg": "LOAD_ARG",
    "GetActualArg":"LOAD_ARG",
    "Dup":      "DUP",
    "DupAt":    "DUP",
    "Pop":      "POP",
    "PopN":     "POP",
    "Swap":     "SWAP",
    "Pick":     "SWAP",
    "Unpick":   "SWAP",

    # ── Global / named vars ──
    "GetGName":             "LOAD_GLOBAL",
    "SetGName":             "STORE_GLOBAL",
    "StrictSetGName":       "STORE_GLOBAL",
    "GetName":              "LOAD_GLOBAL",
    "SetName":              "STORE_GLOBAL",
    "StrictSetName":        "STORE_GLOBAL",
    "GetBoundName":         "LOAD_GLOBAL",
    "BindUnqualifiedGName": "BIND_GLOBAL",
    "BindUnqualifiedName":  "BIND_NAME",
    "BindName":             "BIND_NAME",
    "BindVar":              "BIND_NAME",
    "GetIntrinsic":         "LOAD_GLOBAL",
    "SetIntrinsic":         "STORE_GLOBAL",

    # ── Closure / aliased vars ──
    "GetAliasedVar":        "LOAD_CLOSURE_VAR",
    "SetAliasedVar":        "STORE_CLOSURE_VAR",
    "GetAliasedDebugVar":   "LOAD_CLOSURE_VAR",
    "InitAliasedLexical":   "STORE_CLOSURE_VAR",
    "CheckAliasedLexical":  "TDZ_CHECK",
    "InitLexical":          "STORE_LOCAL",
    "CheckLexical":         "TDZ_CHECK",
    "InitGLexical":         "STORE_GLOBAL",

    # ── this ──
    "FunctionThis":  "LOAD_THIS",
    "GlobalThis":    "LOAD_GLOBAL_THIS",
    "NonSyntacticGlobalThis": "LOAD_GLOBAL_THIS",

    # ── Named property access ──
    "GetProp":        "GET_NAMED_PROP",
    "SetProp":        "SET_NAMED_PROP",
    "StrictSetProp":  "SET_NAMED_PROP",
    "GetPropSuper":   "GET_NAMED_PROP",
    "SetPropSuper":   "SET_NAMED_PROP",
    "StrictSetPropSuper":"SET_NAMED_PROP",
    "InitProp":       "DEFINE_OBJ_PROP",
    "InitHiddenProp": "DEFINE_OBJ_PROP",
    "InitPropGetter": "DEFINE_OBJ_PROP",
    "InitPropSetter": "DEFINE_OBJ_PROP",
    "InitHiddenPropGetter":"DEFINE_OBJ_PROP",
    "InitHiddenPropSetter": "DEFINE_OBJ_PROP",
    "HasOwn":         "IN_OP",

    # ── Keyed property access ──
    "InitLockedProp":    "DEFINE_OBJ_PROP",
    "GetElem":           "GET_KEYED_PROP",
    "SetElem":           "SET_KEYED_PROP",
    "StrictSetElem":     "SET_KEYED_PROP",
    "GetElemSuper":      "GET_KEYED_PROP",
    "SetElemSuper":      "SET_NAMED_PROP",
    "StrictSetElemSuper":"SET_NAMED_PROP",
    "InitElem":          "SET_ARR_ELEM",
    "InitElemArray":     "SET_ARR_ELEM",
    "InitElemInc":       "SET_ARR_ELEM",
    "InitHiddenElem":    "SET_ARR_ELEM",
    "InitHiddenElemGetter":"SET_ARR_ELEM",
    "InitHiddenElemSetter":"SET_ARR_ELEM",
    "InitLockedElem":    "SET_ARR_ELEM",

    # ── Arithmetic ──
    "Add": "BINARY_ADD",
    "Sub": "BINARY_SUB",
    "Mul": "BINARY_MUL",
    "Div": "BINARY_DIV",
    "Mod": "BINARY_MOD",
    "Pow": "BINARY_POW",

    # ── Bitwise ──
    "BitOr":  "BIT_OR",
    "BitAnd": "BIT_AND",
    "BitXor": "BIT_XOR",
    "Lsh":    "BIT_SHL",
    "Rsh":    "BIT_SHR",
    "Ursh":   "BIT_USHR",

    # ── Unary ──
    "Neg":    "UNARY_NEG",
    "BitNot": "UNARY_BITNOT",
    "Not":    "UNARY_NOT",
    "Pos":    "TO_NUMBER",
    "Inc":    "INC",
    "Dec":    "DEC",
    "Void":   "VOID_OP",

    # ── Comparisons ──
    "Eq":               "CMP_EQ",
    "Ne":               "CMP_NEQ",
    "StrictEq":         "CMP_SEQ",
    "StrictNe":         "CMP_SNEQ",
    "StrictConstantEq": "CMP_SEQ",
    "StrictConstantNe": "CMP_SNEQ",
    "Lt":               "CMP_LT",
    "Le":               "CMP_LTE",
    "Gt":               "CMP_GT",
    "Ge":               "CMP_GTE",
    "Instanceof":       "INSTANCEOF",
    "In":               "IN_OP",
    "IsNullOrUndefined":"TEST_NULL_OR_UNDEFINED",

    # ── typeof ──
    "Typeof":     "TYPEOF",
    "TypeofEq":   "TYPEOF_EQ",
    "TypeofExpr": "TYPEOF",

    # ── Jumps / control flow ──
    "Goto":        "JMP",
    "JumpIfTrue":  "JMP_IF_TRUE",
    "JumpIfFalse": "JMP_IF_FALSE",
    "LoopHead":    "JMP_LOOP",
    "JumpTarget":  "JUMP_TARGET",
    "And":         "LOGICAL_AND",
    "Or":          "LOGICAL_OR",
    "Coalesce":    "LOGICAL_NC",
    "Case":        "SWITCH_CASE",
    "Default":     "SWITCH_DEFAULT",
    "TableSwitch": "SWITCH",

    # ── Calls ──
    "Call":            "CALL_FUNC",
    "CallContent":     "CALL_FUNC",
    "CallIgnoresRv":   "CALL_FUNC",
    "CallIter":        "CALL_ITER",
    "CallContentIter": "CALL_ITER",
    "SpreadCall":      "CALL_FUNC",
    "SpreadEval":      "CALL_FUNC",
    "StrictSpreadEval":"CALL_FUNC",
    "Eval":            "CALL_FUNC",
    "StrictEval":      "CALL_FUNC",
    "SuperCall":       "CALL_SUPER",
    "SpreadSuperCall": "CALL_SUPER",
    "SuperFun":        "GET_SUPER",

    # ── Construct ──
    "New":        "CALL_NEW",
    "SpreadNew":  "CALL_NEW",
    "NewContent": "CALL_NEW",

    # ── Return ──
    "Return":  "RETURN",
    "RetRval": "RETURN",
    "SetRval": "RETURN",
    "GetRval": "RETURN",

    # ── Closures / scopes ──
    "Lambda":            "CREATE_CLOSURE",
    "FunWithProto":      "CREATE_CLOSURE",
    "PushLexicalEnv":    "CREATE_SCOPE",
    "PopLexicalEnv":     "POP_SCOPE",
    "DebugLeaveLexicalEnv":"POP_SCOPE",
    "PushVarEnv":        "CREATE_SCOPE",
    "EnterWith":         "CREATE_SCOPE",
    "LeaveWith":         "POP_SCOPE",
    "PushClassBodyEnv":  "CREATE_SCOPE",
    "FreshenLexicalEnv": "PUSH_SCOPE",
    "RecreateLexicalEnv":"PUSH_SCOPE",
    "EnvCallee":         "LOAD_CALLEE",
    "Arguments":         "CREATE_ARGUMENTS",
    "ArgumentsLength":   "GET_ARGS_LEN",
    "Rest":              "CREATE_REST",

    # ── Object / Array literals ──
    "NewObject":  "CREATE_OBJ",
    "NewInit":    "CREATE_OBJ",
    "Object":     "CREATE_OBJ",
    "ObjWithProto":"CREATE_OBJ",
    "NewArray":   "CREATE_ARR",
    "CallSiteObj":"GET_TEMPLATE",
    "RegExp":     "CREATE_REGEXP",

    # ── Type coercions ──
    "ToNumeric":    "TO_NUMBER",
    "ToString":     "TO_STRING",
    "ToPropertyKey":"TO_NAME",
    "CheckObjCoercible":"TO_OBJ",

    # ── Exceptions ──
    "Throw":        "THROW",
    "ThrowMsg":     "THROW",
    "ThrowSetConst":"THROW",
    "ThrowWithStack":"THROW",
    "Exception":    "CATCH_EXCEPTION",
    "ExceptionAndStack":"CATCH_EXCEPTION",
    "Try":          "TRY_START",
    "Finally":      "TRY_FINALLY",

    # ── Iterators ──
    "Iter":              "FOR_IN_START",
    "MoreIter":          "FOR_IN_NEXT",
    "IsNoIter":          "FOR_IN_DONE",
    "EndIter":           "FOR_IN_END",
    "CloseIter":         "FOR_IN_END",
    "ToAsyncIter":       "GET_ITER",
    "OptimizeGetIterator":"GET_ITER",

    # ── Generators / async ──
    "Generator":         "GEN_START",
    "InitialYield":      "GEN_YIELD",
    "Yield":             "GEN_YIELD",
    "FinalYieldRval":    "GEN_YIELD",
    "AfterYield":        "GEN_RESUME",
    "Resume":            "GEN_RESUME",
    "ResumeKind":        "GEN_RESUME",
    "CheckResumeKind":   "GEN_RESUME",
    "Await":             "ASYNC_AWAIT",
    "AsyncAwait":        "ASYNC_AWAIT",
    "AsyncResolve":      "ASYNC_RESOLVE",
    "AsyncReject":       "ASYNC_REJECT",
    "CanSkipAwait":      "ASYNC_AWAIT",
    "MaybeExtractAwaitValue":"ASYNC_AWAIT",

    # ── Properties / proto ──
    "MutateProto":       "SET_PROTO",
    "InitHomeObject":    "INIT_HOME",
    "SetFunName":        "SET_FUN_NAME",
    "ObjWithProto":      "CREATE_OBJ",

    # ── Delete ──
    "DelProp":       "DELETE_PROP",
    "StrictDelProp": "DELETE_PROP",
    "DelElem":       "DELETE_PROP",
    "StrictDelElem": "DELETE_PROP",
    "DelName":       "DELETE_PROP",

    # ── Misc ──
    "Nop":                 "NOP",
    "Lineno":              "NOP",
    "NopDestructuring":    "NOP",
    "NopIsAssignOp":       "NOP",
    "ForceInterpreter":    "NOP",
    "Debugger":            "DEBUGGER",
    "Callee":              "LOAD_CALLEE",
    "IsConstructing":      "IS_CONSTRUCTING",
    "CheckClassHeritage":  "CLASS_CHECK",
    "CheckReturn":         "CLASS_CHECK",
    "CheckThis":           "CHECK_THIS",
    "CheckThisReinit":     "CHECK_THIS",
    "CheckIsObj":          "CLASS_CHECK",
    "CheckPrivateField":   "CLASS_CHECK",
    "NewPrivateName":      "CLASS_CHECK",
    "InitHiddenElemGetter":"DEFINE_OBJ_PROP",
    "InitHiddenElemSetter":"DEFINE_OBJ_PROP",
    "BuiltinObject":       "LOAD_BUILTIN",
    "GlobalOrEvalDeclInstantiation":"SCOPE_INIT",
    "IsGenClosing":        "GEN_RESUME",
    "IsNoIter":            "FOR_IN_DONE",
    "ImplicitThis":        "LOAD_THIS",
    "SuperBase":           "GET_SUPER",
    "CheckAliasedLexical": "TDZ_CHECK",
    "DynamicImport":       "IMPORT",
    "ImportMeta":          "IMPORT",
    "GetImport":           "IMPORT",
    "TryDestructuring":    "NOP",
    "NewTarget":           "LOAD_NEW_TARGET",
    "OptimizeSpreadCall":  "CALL_FUNC",
}


# ── Empirical validation ───────────────────────────────────────────────────────

def load_logs() -> dict[str, dict]:
    """Load all microscript log files."""
    logs = {}
    for path in sorted(LOGS_DIR.glob("*.json")):
        try:
            logs[path.stem] = json.loads(path.read_text())
        except Exception as e:
            print(f"  [WARN] Could not read {path.name}: {e}")
    return logs


def extract_opcodes(records: list[dict]) -> Counter:
    counts: Counter = Counter()
    for rec in records:
        for op in rec.get("bytecodes", "").split(","):
            op = op.strip()
            if op:
                counts[op] += 1
    return counts


def coverage_report(
    v8_seen: Counter, sm_seen: Counter
) -> tuple[list[str], list[str], float, float]:
    v8_unmapped = sorted(op for op in v8_seen if op not in V8_TO_IR)
    sm_unmapped = sorted(op for op in sm_seen if op not in SM_TO_IR)
    v8_pct = 100 * (1 - len(v8_unmapped) / max(len(v8_seen), 1))
    sm_pct = 100 * (1 - len(sm_unmapped) / max(len(sm_seen), 1))
    return v8_unmapped, sm_unmapped, v8_pct, sm_pct


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    dry_run = "--dry-run" in sys.argv

    all_ir_tokens = sorted(set(V8_TO_IR.values()) | set(SM_TO_IR.values()))

    if not LOGS_DIR.exists() or not any(LOGS_DIR.glob("*.json")):
        print("[WARN] No microscript logs found — writing static mapping only.")
        print(f"       Run  python ir/run_microscripts.py  first for empirical coverage.")
        v8_seen: Counter = Counter()
        sm_seen: Counter = Counter()
    else:
        logs = load_logs()
        print(f"Loaded {len(logs)} microscript logs.\n")

        v8_seen: Counter = Counter()
        sm_seen: Counter = Counter()
        for stem, data in logs.items():
            v8_c = extract_opcodes(data.get("v8", []))
            sm_c = extract_opcodes(data.get("sm", []))
            v8_n = sum(v8_c.values())
            sm_n = sum(sm_c.values())
            print(f"  {stem:35s}  V8={v8_n:4d} tok  SM={sm_n:4d} tok")
            v8_seen += v8_c
            sm_seen += sm_c

    v8_unmapped, sm_unmapped, v8_pct, sm_pct = coverage_report(v8_seen, sm_seen)

    print(f"\n── V8 coverage ──────────────────────────────────────────")
    print(f"  Unique opcodes seen : {len(v8_seen)}")
    print(f"  Mapped to IR        : {len(v8_seen) - len(v8_unmapped)} / {len(v8_seen)}  ({v8_pct:.1f}%)")
    if v8_unmapped:
        print(f"  Unmapped            : {v8_unmapped}")

    print(f"\n── SM coverage ──────────────────────────────────────────")
    print(f"  Unique opcodes seen : {len(sm_seen)}")
    print(f"  Mapped to IR        : {len(sm_seen) - len(sm_unmapped)} / {len(sm_seen)}  ({sm_pct:.1f}%)")
    if sm_unmapped:
        print(f"  Unmapped            : {sm_unmapped}")

    print(f"\n── IR vocabulary ────────────────────────────────────────")
    print(f"  Total IR tokens     : {len(all_ir_tokens)}")

    out = {
        "version": 1,
        "description": (
            "Cross-engine IR equivalence dictionary. "
            "Maps V8 (Ignition) and SpiderMonkey bytecode opcodes to a common IR token. "
            "Derived from opcode_map.md and validated with ir/microscripts/."
        ),
        "v8_to_ir":  V8_TO_IR,
        "sm_to_ir":  SM_TO_IR,
        "ir_tokens": all_ir_tokens,
        "coverage": {
            "v8_unique_seen":  len(v8_seen),
            "v8_mapped_pct":   round(v8_pct, 1),
            "v8_unmapped":     v8_unmapped,
            "sm_unique_seen":  len(sm_seen),
            "sm_mapped_pct":   round(sm_pct, 1),
            "sm_unmapped":     sm_unmapped,
        },
    }

    if dry_run:
        print("\n[dry-run] Not writing output.")
        return

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWritten → {OUT_PATH}")
    print("Next step: ir/translate_dataset.py  (Sprint 6)")


if __name__ == "__main__":
    main()
