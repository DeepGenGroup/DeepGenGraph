#ifndef FRISK_TYPES
#define FRISK_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "deepgengraph/Dialect/Frisk/IR/FriskDialect.td"

class Frisk_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Frisk_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Frisk_BarrierType : Frisk_Type<"Barrier", "barrier"> {
  let summary = "Barrier handle type";
  let description = [{
    Lightweight handle representing a synchronization barrier in Frisk.
  }];
}

// 假设这是在你的 Frisk_Dialect 中
def Frisk_TmaDescType : Frisk_Type<"TmaDesc", "tma_desc"> {
  let summary = "TMA descriptor type encapsulating global tensor metadata";
  let description = [{
    Frisk Dialect 中的 TMA 描述符类型。
    它封装了全局内存中张量的元数据（如 Shape、ElementType、Layout 等），
    用于指导硬件 TMA 单元进行高维数据的异步搬运。

    语法示例:
    `!frisk.tma_desc<memref<2048x2048xf16>>`
  }];

  // 定义类型参数：它所绑定的全局内存数据类型（通常是一个全局的 MemRef）
  let parameters = (ins
    "Type":$underlyingType
  );

  // 格式化打印与解析
  let assemblyFormat = "`<` $underlyingType `>`";

  // 添加一些辅助的 C++ 方法，方便在 C++ 源码中直接获取底层类型
  let extraClassDeclaration = [{
    // 获取底层绑定的 MemRef 类型
    mlir::MemRefType getUnderlyingMemRefType() const {
      return mlir::cast<mlir::MemRefType>(getUnderlyingType());
    }
  }];
}




#endif // FRISK_TYPES
