# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Helpers\ProductCode
### Namespace: [\YooKassa\Helpers](../namespaces/yookassa-helpers.md)
---
**Summary:**

Класс, представляющий модель ProductCode.

**Description:**

Класс для формирования тега 1162 на основе кода в формате Data Matrix.

---
### Examples
Вариант через метод

```php
$inputDataMatrix = '010463003407001221SxMGorvNuq6Wk91fgr92sdfsdfghfgjh';
$productCode = new \YooKassa\Helpers\ProductCode($inputDataMatrix);
$receiptItem = new \YooKassa\Model\Receipt\ReceiptItem();
$receiptItem->setProductCode($productCode);
$receiptItem->setMarkCodeInfo($productCode->getMarkCodeInfo());

var_dump($receiptItem);

```
Вариант через массив

```php
$inputDataMatrix = '010463003407001221SxMGorvNuq6Wk91fgr92sdfsdfghfgjh';
$receiptItem = new \YooKassa\Model\Receipt\ReceiptItem([
    'product_code' => (string) ($code = new \YooKassa\Helpers\ProductCode($inputDataMatrix)),
    'mark_code_info' => $code->getMarkCodeInfo(),
]);

var_dump($receiptItem);

```

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [PREFIX_DATA_MATRIX](../classes/YooKassa-Helpers-ProductCode.md#constant_PREFIX_DATA_MATRIX) |  |  |
| public | [PREFIX_UNKNOWN](../classes/YooKassa-Helpers-ProductCode.md#constant_PREFIX_UNKNOWN) |  |  |
| public | [PREFIX_EAN_8](../classes/YooKassa-Helpers-ProductCode.md#constant_PREFIX_EAN_8) |  |  |
| public | [PREFIX_EAN_13](../classes/YooKassa-Helpers-ProductCode.md#constant_PREFIX_EAN_13) |  |  |
| public | [PREFIX_ITF_14](../classes/YooKassa-Helpers-ProductCode.md#constant_PREFIX_ITF_14) |  |  |
| public | [PREFIX_FUR](../classes/YooKassa-Helpers-ProductCode.md#constant_PREFIX_FUR) |  |  |
| public | [PREFIX_EGAIS_20](../classes/YooKassa-Helpers-ProductCode.md#constant_PREFIX_EGAIS_20) |  |  |
| public | [PREFIX_EGAIS_30](../classes/YooKassa-Helpers-ProductCode.md#constant_PREFIX_EGAIS_30) |  |  |
| public | [TYPE_UNKNOWN](../classes/YooKassa-Helpers-ProductCode.md#constant_TYPE_UNKNOWN) |  |  |
| public | [TYPE_EAN_8](../classes/YooKassa-Helpers-ProductCode.md#constant_TYPE_EAN_8) |  |  |
| public | [TYPE_EAN_13](../classes/YooKassa-Helpers-ProductCode.md#constant_TYPE_EAN_13) |  |  |
| public | [TYPE_ITF_14](../classes/YooKassa-Helpers-ProductCode.md#constant_TYPE_ITF_14) |  |  |
| public | [TYPE_GS_10](../classes/YooKassa-Helpers-ProductCode.md#constant_TYPE_GS_10) |  |  |
| public | [TYPE_GS_1M](../classes/YooKassa-Helpers-ProductCode.md#constant_TYPE_GS_1M) |  |  |
| public | [TYPE_SHORT](../classes/YooKassa-Helpers-ProductCode.md#constant_TYPE_SHORT) |  |  |
| public | [TYPE_FUR](../classes/YooKassa-Helpers-ProductCode.md#constant_TYPE_FUR) |  |  |
| public | [TYPE_EGAIS_20](../classes/YooKassa-Helpers-ProductCode.md#constant_TYPE_EGAIS_20) |  |  |
| public | [TYPE_EGAIS_30](../classes/YooKassa-Helpers-ProductCode.md#constant_TYPE_EGAIS_30) |  |  |
| public | [AI_GTIN](../classes/YooKassa-Helpers-ProductCode.md#constant_AI_GTIN) |  |  |
| public | [AI_SERIAL](../classes/YooKassa-Helpers-ProductCode.md#constant_AI_SERIAL) |  |  |
| public | [AI_SUM](../classes/YooKassa-Helpers-ProductCode.md#constant_AI_SUM) |  |  |
| public | [MAX_PRODUCT_CODE_LENGTH](../classes/YooKassa-Helpers-ProductCode.md#constant_MAX_PRODUCT_CODE_LENGTH) |  |  |
| public | [MAX_MARK_CODE_LENGTH](../classes/YooKassa-Helpers-ProductCode.md#constant_MAX_MARK_CODE_LENGTH) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Helpers-ProductCode.md#method___construct) |  | ProductCode constructor. |
| public | [__toString()](../classes/YooKassa-Helpers-ProductCode.md#method___toString) |  | Приводит объект к строке. |
| public | [calcResult()](../classes/YooKassa-Helpers-ProductCode.md#method_calcResult) |  | Формирует тег 1162. |
| public | [getAppIdentifiers()](../classes/YooKassa-Helpers-ProductCode.md#method_getAppIdentifiers) |  | Возвращает массив дополнительных идентификаторов применения. |
| public | [getGtin()](../classes/YooKassa-Helpers-ProductCode.md#method_getGtin) |  | Возвращает глобальный номер товарной продукции. |
| public | [getMarkCodeInfo()](../classes/YooKassa-Helpers-ProductCode.md#method_getMarkCodeInfo) |  |  |
| public | [getPrefix()](../classes/YooKassa-Helpers-ProductCode.md#method_getPrefix) |  | Возвращает код типа маркировки. |
| public | [getResult()](../classes/YooKassa-Helpers-ProductCode.md#method_getResult) |  | Возвращает сформированный тег 1162. |
| public | [getSerial()](../classes/YooKassa-Helpers-ProductCode.md#method_getSerial) |  | Возвращает серийный номер товара. |
| public | [getType()](../classes/YooKassa-Helpers-ProductCode.md#method_getType) |  | Возвращает тип маркировки. |
| public | [isUsePrefix()](../classes/YooKassa-Helpers-ProductCode.md#method_isUsePrefix) |  | Возвращает флаг использования кода типа маркировки. |
| public | [setAppIdentifiers()](../classes/YooKassa-Helpers-ProductCode.md#method_setAppIdentifiers) |  | Устанавливает массив дополнительных идентификаторов применения. |
| public | [setGtin()](../classes/YooKassa-Helpers-ProductCode.md#method_setGtin) |  | Устанавливает глобальный номер товарной продукции. |
| public | [setMarkCodeInfo()](../classes/YooKassa-Helpers-ProductCode.md#method_setMarkCodeInfo) |  |  |
| public | [setPrefix()](../classes/YooKassa-Helpers-ProductCode.md#method_setPrefix) |  | Устанавливает код типа маркировки. |
| public | [setSerial()](../classes/YooKassa-Helpers-ProductCode.md#method_setSerial) |  | Устанавливает серийный номер товара. |
| public | [setType()](../classes/YooKassa-Helpers-ProductCode.md#method_setType) |  | Устанавливает тип маркировки. |
| public | [setUsePrefix()](../classes/YooKassa-Helpers-ProductCode.md#method_setUsePrefix) |  | Устанавливает флаг использования кода типа маркировки. |
| public | [validate()](../classes/YooKassa-Helpers-ProductCode.md#method_validate) |  | Проверяет заполненность необходимых свойств. |

---
### Details
* File: [lib/Helpers/ProductCode.php](../../lib/Helpers/ProductCode.php)
* Package: YooKassa\Model
* Class Hierarchy:
  * \YooKassa\Helpers\ProductCode

* See Also:
  * [](https://yookassa.ru/developers/api)
  * [](https://git.yoomoney.ru/projects/SDK/repos/yookassa-sdk-php/browse/lib/Helpers/ProductCode.php)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Constants
<a name="constant_PREFIX_DATA_MATRIX" class="anchor"></a>
###### PREFIX_DATA_MATRIX
```php
PREFIX_DATA_MATRIX = '444D' : string
```


<a name="constant_PREFIX_UNKNOWN" class="anchor"></a>
###### PREFIX_UNKNOWN
```php
PREFIX_UNKNOWN = '0000' : string
```


<a name="constant_PREFIX_EAN_8" class="anchor"></a>
###### PREFIX_EAN_8
```php
PREFIX_EAN_8 = '4508' : string
```


<a name="constant_PREFIX_EAN_13" class="anchor"></a>
###### PREFIX_EAN_13
```php
PREFIX_EAN_13 = '450D' : string
```


<a name="constant_PREFIX_ITF_14" class="anchor"></a>
###### PREFIX_ITF_14
```php
PREFIX_ITF_14 = '4909' : string
```


<a name="constant_PREFIX_FUR" class="anchor"></a>
###### PREFIX_FUR
```php
PREFIX_FUR = '5246' : string
```


<a name="constant_PREFIX_EGAIS_20" class="anchor"></a>
###### PREFIX_EGAIS_20
```php
PREFIX_EGAIS_20 = 'C514' : string
```


<a name="constant_PREFIX_EGAIS_30" class="anchor"></a>
###### PREFIX_EGAIS_30
```php
PREFIX_EGAIS_30 = 'C51E' : string
```


<a name="constant_TYPE_UNKNOWN" class="anchor"></a>
###### TYPE_UNKNOWN
```php
TYPE_UNKNOWN = 'unknown' : string
```


<a name="constant_TYPE_EAN_8" class="anchor"></a>
###### TYPE_EAN_8
```php
TYPE_EAN_8 = 'ean_8' : string
```


<a name="constant_TYPE_EAN_13" class="anchor"></a>
###### TYPE_EAN_13
```php
TYPE_EAN_13 = 'ean_13' : string
```


<a name="constant_TYPE_ITF_14" class="anchor"></a>
###### TYPE_ITF_14
```php
TYPE_ITF_14 = 'itf_14' : string
```


<a name="constant_TYPE_GS_10" class="anchor"></a>
###### TYPE_GS_10
```php
TYPE_GS_10 = 'gs_10' : string
```


<a name="constant_TYPE_GS_1M" class="anchor"></a>
###### TYPE_GS_1M
```php
TYPE_GS_1M = 'gs_1m' : string
```


<a name="constant_TYPE_SHORT" class="anchor"></a>
###### TYPE_SHORT
```php
TYPE_SHORT = 'short' : string
```


<a name="constant_TYPE_FUR" class="anchor"></a>
###### TYPE_FUR
```php
TYPE_FUR = 'fur' : string
```


<a name="constant_TYPE_EGAIS_20" class="anchor"></a>
###### TYPE_EGAIS_20
```php
TYPE_EGAIS_20 = 'egais_20' : string
```


<a name="constant_TYPE_EGAIS_30" class="anchor"></a>
###### TYPE_EGAIS_30
```php
TYPE_EGAIS_30 = 'egais_30' : string
```


<a name="constant_AI_GTIN" class="anchor"></a>
###### AI_GTIN
```php
AI_GTIN = '01' : string
```


<a name="constant_AI_SERIAL" class="anchor"></a>
###### AI_SERIAL
```php
AI_SERIAL = '21' : string
```


<a name="constant_AI_SUM" class="anchor"></a>
###### AI_SUM
```php
AI_SUM = '8005' : string
```


<a name="constant_MAX_PRODUCT_CODE_LENGTH" class="anchor"></a>
###### MAX_PRODUCT_CODE_LENGTH
```php
MAX_PRODUCT_CODE_LENGTH = 30 : int
```


<a name="constant_MAX_MARK_CODE_LENGTH" class="anchor"></a>
###### MAX_MARK_CODE_LENGTH
```php
MAX_MARK_CODE_LENGTH = 32 : int
```



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(null|string $codeDataMatrix = null, bool|string $usePrefix = true) : mixed
```

**Summary**

ProductCode constructor.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | codeDataMatrix  | Строка, расшифрованная из QR-кода |
| <code lang="php">bool OR string</code> | usePrefix  | Нужен ли код типа маркировки в результате |

**Returns:** mixed - 


<a name="method___toString" class="anchor"></a>
#### public __toString() : string

```php
public __toString() : string
```

**Summary**

Приводит объект к строке.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

**Returns:** string - 


<a name="method_calcResult" class="anchor"></a>
#### public calcResult() : string

```php
public calcResult() : string
```

**Summary**

Формирует тег 1162.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

**Returns:** string - Сформированный тег 1162


<a name="method_getAppIdentifiers" class="anchor"></a>
#### public getAppIdentifiers() : null|array

```php
public getAppIdentifiers() : null|array
```

**Summary**

Возвращает массив дополнительных идентификаторов применения.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

**Returns:** null|array - Массив дополнительных идентификаторов применения


<a name="method_getGtin" class="anchor"></a>
#### public getGtin() : string|null

```php
public getGtin() : string|null
```

**Summary**

Возвращает глобальный номер товарной продукции.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

**Returns:** string|null - Глобальный номер товарной продукции


<a name="method_getMarkCodeInfo" class="anchor"></a>
#### public getMarkCodeInfo() : ?\YooKassa\Model\Receipt\MarkCodeInfo

```php
public getMarkCodeInfo() : ?\YooKassa\Model\Receipt\MarkCodeInfo
```

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

**Returns:** ?\YooKassa\Model\Receipt\MarkCodeInfo - 


<a name="method_getPrefix" class="anchor"></a>
#### public getPrefix() : ?string

```php
public getPrefix() : ?string
```

**Summary**

Возвращает код типа маркировки.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

**Returns:** ?string - Код типа маркировки


<a name="method_getResult" class="anchor"></a>
#### public getResult() : string

```php
public getResult() : string
```

**Summary**

Возвращает сформированный тег 1162.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

**Returns:** string - Сформированный тег 1162


<a name="method_getSerial" class="anchor"></a>
#### public getSerial() : string|null

```php
public getSerial() : string|null
```

**Summary**

Возвращает серийный номер товара.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

**Returns:** string|null - Серийный номер товара


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает тип маркировки.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

**Returns:** string|null - Тип маркировки


<a name="method_isUsePrefix" class="anchor"></a>
#### public isUsePrefix() : bool

```php
public isUsePrefix() : bool
```

**Summary**

Возвращает флаг использования кода типа маркировки.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

**Returns:** bool - 


<a name="method_setAppIdentifiers" class="anchor"></a>
#### public setAppIdentifiers() : void

```php
public setAppIdentifiers(null|array $appIdentifiers) : void
```

**Summary**

Устанавливает массив дополнительных идентификаторов применения.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array</code> | appIdentifiers  | Массив дополнительных идентификаторов применения |

**Returns:** void - 


<a name="method_setGtin" class="anchor"></a>
#### public setGtin() : \YooKassa\Helpers\ProductCode

```php
public setGtin(string|null $gtin) : \YooKassa\Helpers\ProductCode
```

**Summary**

Устанавливает глобальный номер товарной продукции.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | gtin  | Глобальный номер товарной продукции |

**Returns:** \YooKassa\Helpers\ProductCode - 


<a name="method_setMarkCodeInfo" class="anchor"></a>
#### public setMarkCodeInfo() : void

```php
public setMarkCodeInfo(array|\YooKassa\Model\Receipt\MarkCodeInfo|string $markCodeInfo) : void
```

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\MarkCodeInfo OR string</code> | markCodeInfo  |  |

**Returns:** void - 


<a name="method_setPrefix" class="anchor"></a>
#### public setPrefix() : \YooKassa\Helpers\ProductCode

```php
public setPrefix(int|string|null $prefix) : \YooKassa\Helpers\ProductCode
```

**Summary**

Устанавливает код типа маркировки.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR string OR null</code> | prefix  | Код типа маркировки |

**Returns:** \YooKassa\Helpers\ProductCode - 


<a name="method_setSerial" class="anchor"></a>
#### public setSerial() : \YooKassa\Helpers\ProductCode

```php
public setSerial(string|null $serial) : \YooKassa\Helpers\ProductCode
```

**Summary**

Устанавливает серийный номер товара.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | serial  | Серийный номер товара |

**Returns:** \YooKassa\Helpers\ProductCode - 


<a name="method_setType" class="anchor"></a>
#### public setType() : void

```php
public setType(string $type) : void
```

**Summary**

Устанавливает тип маркировки.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | type  | Тип маркировки |

**Returns:** void - 


<a name="method_setUsePrefix" class="anchor"></a>
#### public setUsePrefix() : \YooKassa\Helpers\ProductCode

```php
public setUsePrefix(bool $usePrefix) : \YooKassa\Helpers\ProductCode
```

**Summary**

Устанавливает флаг использования кода типа маркировки.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool</code> | usePrefix  | Флаг использования кода типа маркировки |

**Returns:** \YooKassa\Helpers\ProductCode - 


<a name="method_validate" class="anchor"></a>
#### public validate() : bool

```php
public validate() : bool
```

**Summary**

Проверяет заполненность необходимых свойств.

**Details:**
* Inherited From: [\YooKassa\Helpers\ProductCode](../classes/YooKassa-Helpers-ProductCode.md)

**Returns:** bool - 



---

### Top Namespaces

* [\YooKassa](../namespaces/yookassa.md)

---

### Reports
* [Errors - 0](../reports/errors.md)
* [Markers - 0](../reports/markers.md)
* [Deprecated - 40](../reports/deprecated.md)

---

This document was automatically generated from source code comments on 2025-09-04 using [phpDocumentor](http://www.phpdoc.org/)

&copy; 2025 YooMoney